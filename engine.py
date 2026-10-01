"""
Battery Clinic AI - modelling engine (no UI code).

Stages
  1. load_data            read + validate + clean the CSV
  2. add_engineered       calculated attributes (config.ENGINEERED)
  3. select_engineered    keep an engineered attribute only if it improves cross-validated R²
  4. stability_run        every model x 5 shuffles -> mean / spread of test scores
  5. fit_final            refit the chosen model on all rows for prediction
  6. predict_frame        raw inputs -> health %
  7. insights             percentiles, drivers, conclusion
"""
import csv
import hashlib
import io
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

import config as C

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ============================================================ 1. loading
def file_fingerprint(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _to_number(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype(str).str.strip().str.lower()
    t = t.replace({"yes": "1", "no": "0", "true": "1", "false": "0", "y": "1", "n": "0"})
    t = t.str.replace(r"[,\s%$]", "", regex=True)
    return pd.to_numeric(t, errors="coerce")


def read_csv_any(source):
    """Read a CSV from a path or uploaded file, detecting separator and encoding."""
    raw = source.read() if hasattr(source, "read") else open(source, "rb").read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    try:
        sep = csv.Sniffer().sniff(text[:50_000], delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    return pd.read_csv(io.StringIO(text), sep=sep)


def align_columns(df):
    """Rename columns to the names in config (case/space-insensitive match)."""
    wanted = list(C.FEATURES) + [C.TARGET]
    lookup = {_norm(w): w for w in wanted}
    return df.rename(columns={c: lookup[_norm(c)] for c in df.columns if _norm(c) in lookup})


def load_data(path):
    df = align_columns(read_csv_any(path))
    report = {"rows_raw": len(df), "columns_raw": list(df.columns)}
    required = [c for c in C.FEATURES if c != C.OPTIONAL_FEATURE] + [C.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"The data file is missing required column(s): {missing}. "
                         f"Found: {list(df.columns)}")
    report["optional_available"] = C.OPTIONAL_FEATURE in df.columns
    keep = [c for c in list(C.FEATURES) + [C.TARGET] if c in df.columns]
    report["ignored_columns"] = [c for c in df.columns if c not in keep]
    df = df[keep].copy()
    for c in keep:
        df[c] = _to_number(df[c])
    n = len(df)
    df = df[df[C.TARGET].notna()]
    report["missing_target_removed"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = n - len(df)
    report["missing_values"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    report["rows_final"] = len(df)
    df = df.reset_index(drop=True)
    return df, report


def raw_inputs(include_optional):
    return [c for c in C.FEATURES if include_optional or c != C.OPTIONAL_FEATURE]


# ============================================================ 2. engineered attributes
def add_engineered(df):
    out = df.copy()
    age = out["Battery Age"].where(out["Battery Age"] > 0)
    hours = out["Daily Usage Hours"].where(out["Daily Usage Hours"] > 0)
    out["Cycles per Year"] = out["Cycle Count"] / age
    out["Heat Stress"] = out["Average Temperature"] * out["Daily Usage Hours"]
    out["Workload Intensity"] = (out["CPU Usage"] + out["GPU Usage"]) / 2
    out["Power per Usage Hour"] = out["Power Consumption"] / hours
    out["Gaming Load"] = out["Gaming User"] * out["GPU Usage"]
    return out


def column_stats(df):
    stats = {}
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        stats[c] = {"min": float(s.min()), "max": float(s.max()), "median": float(s.median()),
                    "mean": float(s.mean()), "std": float(s.std()), "mode": float(s.mode().iloc[0]),
                    "missing": int(df[c].isna().sum())}
    return stats


# ============================================================ models
def make_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=10, weights="distance"),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=0),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "Extra Trees": ExtraTreesRegressor(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0),
        "Gradient Boosting": GradientBoostingRegressor(random_state=0),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=0),
        "Support Vector Regression": SVR(C=10.0, epsilon=0.5),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                         random_state=0, verbosity=0, n_jobs=-1)
    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=0, verbose=-1)
    return models


def make_pipeline(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", clone(model))])


def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nz = y_true != 0
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE": float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100)}


# ============================================================ 3. engineered-feature selection
def select_engineered(df, base_cols, test_size):
    """Keep an engineered attribute only if it raises cross-validated R² by >= threshold.
    Done on the TRAINING part of the first shuffle only, so test rows never influence the choice."""
    eng = add_engineered(df)
    train_idx, _ = train_test_split(eng.index, test_size=test_size, random_state=C.STABILITY_SEEDS[0])
    tr = eng.loc[train_idx]
    y = tr[C.TARGET]
    cv = KFold(n_splits=C.CV_FOLDS_FOR_SELECTION, shuffle=True, random_state=0)
    ref = make_pipeline(HistGradientBoostingRegressor(random_state=0))

    def score(cols):
        return float(cross_val_score(ref, tr[cols], y, cv=cv, scoring="r2").mean())

    base = score(base_cols)
    rows, kept = [], []
    for name, (formula, sources, _) in C.ENGINEERED.items():
        if not all(s in base_cols for s in sources):
            continue
        s = score(base_cols + [name])
        gain = s - base
        rows.append({"Attribute": name, "Formula": formula, "CV R² without": base,
                     "CV R² with": s, "Gain": gain, "Kept": gain >= C.FEATURE_GAIN_THRESHOLD})
        if gain >= C.FEATURE_GAIN_THRESHOLD:
            kept.append(name)
    combined = None
    if len(kept) > 1:  # confirm the kept attributes also help together
        combined = score(base_cols + kept)
        best_single = max(r["CV R² with"] for r in rows if r["Kept"])
        if combined < best_single:
            best = max((r for r in rows if r["Kept"]), key=lambda r: r["CV R² with"])
            kept = [best["Attribute"]]
            for r in rows:
                r["Kept"] = r["Attribute"] == best["Attribute"]
    return kept, pd.DataFrame(rows), {"baseline": base, "combined": combined}


# ============================================================ 4. stability run
def stability_run(df, features, test_size, progress=None):
    eng = add_engineered(df)
    X, y = eng[features], eng[C.TARGET]
    models = make_models()
    per_seed, abs_errors = [], {m: [] for m in models}
    first_split, fitted_first = {}, {}
    all_splits = {m: [] for m in models}
    total = len(models) * len(C.STABILITY_SEEDS)
    step = 0
    for si, seed in enumerate(C.STABILITY_SEEDS):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        for name, model in models.items():
            step += 1
            if progress:
                progress(step / total, f"Shuffle {si + 1}/{len(C.STABILITY_SEEDS)} · {name}")
            pipe = make_pipeline(model)
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            secs = time.time() - t0
            pred = pipe.predict(X_te)
            m = metrics(y_te, pred)
            m.update({"Model": name, "Seed": seed, "Train R2": r2_score(y_tr, pipe.predict(X_tr)),
                      "Fit seconds": secs})
            per_seed.append(m)
            abs_errors[name].extend(np.abs(np.asarray(y_te) - pred).tolist())
            all_splits[name].append((np.asarray(y_te), np.asarray(pred)))
            if si == 0:
                first_split[name] = (y_te.values, pred, X_te.index.values)
                fitted_first[name] = pipe
    per_seed = pd.DataFrame(per_seed)
    g = per_seed.groupby("Model")
    table = pd.DataFrame({
        "Mean R²": g["R2"].mean(), "R² spread (±)": g["R2"].std(),
        "Worst R²": g["R2"].min(), "Best R²": g["R2"].max(),
        "MAE": g["MAE"].mean(), "RMSE": g["RMSE"].mean(), "MAPE %": g["MAPE"].mean(),
        "Train R²": g["Train R2"].mean(), "Fit time (s)": g["Fit seconds"].mean(),
    })
    table["Overfit gap"] = table["Train R²"] - table["Mean R²"]
    table["Stability score"] = table["Mean R²"] - table["R² spread (±)"]
    table["90% error within (±)"] = [float(np.quantile(abs_errors[m], 0.9)) for m in table.index]
    table = table.sort_values("Stability score", ascending=False)
    return {"table": table, "per_seed": per_seed, "first_split": first_split, "all_splits": all_splits,
            "fitted_first": fitted_first, "recommended": table.index[0],
            "test_rows": int(round(len(X) * test_size)), "train_rows": len(X) - int(round(len(X) * test_size))}


def importance_for(result, model_name, df, features, test_size, n_repeats=5):
    """Permutation importance of the chosen model on the first shuffle's test set."""
    eng = add_engineered(df)
    _, idx_te = train_test_split(eng.index, test_size=test_size, random_state=C.STABILITY_SEEDS[0])
    pipe = result["fitted_first"][model_name]
    X_te, y_te = eng.loc[idx_te, features], eng.loc[idx_te, C.TARGET]
    pi = permutation_importance(pipe, X_te, y_te, scoring="r2", n_repeats=n_repeats, random_state=0)
    return (pd.DataFrame({"Attribute": features, "Importance": pi.importances_mean, "Std": pi.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


# ============================================================ 5-6. final model & prediction
def fit_final(df, features, model_name):
    eng = add_engineered(df)
    pipe = make_pipeline(make_models()[model_name])
    pipe.fit(eng[features], eng[C.TARGET])
    return pipe


def predict_frame(model, raw_df, features):
    eng = add_engineered(raw_df)
    return np.clip(model.predict(eng[features]), 0, 100)


def health_band(value):
    for lower, label, colour in C.HEALTH_BANDS:
        if value >= lower:
            return label, colour
    return C.HEALTH_BANDS[-1][1], C.HEALTH_BANDS[-1][2]


# ============================================================ 7. insights
def percentile_of(series, value):
    s = series.dropna().values
    return float(((s < value).mean() + 0.5 * (s == value).mean()) * 100)


def compare_to_data(row, df, columns):
    eng_data, eng_row = add_engineered(df), add_engineered(pd.DataFrame([row]))
    out = []
    for c in columns:
        if c in C.FEATURES and C.FEATURES[c]["kind"] == "binary":
            continue
        v = float(eng_row[c].iloc[0]) if pd.notna(eng_row[c].iloc[0]) else np.nan
        if np.isnan(v):
            continue
        p = percentile_of(eng_data[c], v)
        level = "high" if p >= 80 else "low" if p <= 20 else "typical"
        out.append({"Attribute": c, "Value": v, "Percentile": p, "Level": level})
    return out


CAPACITY_GROUP = "Capacity (Design + Full Charge)"


def driver_groups(inputs):
    """Inputs tested one at a time, except Design + Full Charge Capacity, which only
    make sense together (their ratio is the health), so they are tested as a pair."""
    groups = {c: [c] for c in inputs}
    if C.OPTIONAL_FEATURE in inputs and "Design Capacity" in inputs:
        groups.pop(C.OPTIONAL_FEATURE)
        groups.pop("Design Capacity")
        groups[CAPACITY_GROUP] = ["Design Capacity", C.OPTIONAL_FEATURE]
    return groups


def drivers(model, row, features, inputs, stats):
    """Effect of an input = prediction(actual values) − prediction(that input set to the typical value).
    Typical = median (mode for Yes/No inputs). Positive -> raises health compared with a typical laptop."""
    base = float(predict_frame(model, pd.DataFrame([row]), features)[0])
    out = []
    for label, cols in driver_groups(inputs).items():
        alt = dict(row)
        for c in cols:
            alt[c] = stats[c]["mode"] if C.FEATURES[c]["kind"] == "binary" else stats[c]["median"]
        p = float(predict_frame(model, pd.DataFrame([alt]), features)[0])
        out.append({"Input": label, "Columns": cols, "Value": row[cols[0]] if len(cols) == 1 else None,
                    "Typical": alt[cols[0]] if len(cols) == 1 else None, "Effect": base - p})
    return sorted(out, key=lambda d: abs(d["Effect"]), reverse=True), base


def conclusion(pred, band, drv, comps):
    lines = [f"Estimated battery health is {pred:.1f}% ({band})."]
    negatives = [d for d in drv if d["Effect"] <= -C.INSIGHT_MIN_EFFECT]
    positives = [d for d in drv if d["Effect"] >= C.INSIGHT_MIN_EFFECT]
    if negatives:
        names = " and ".join(f"{d['Input']} ({d['Effect']:+.1f} points)" for d in negatives[:2])
        lines.append(f"The inputs pulling it down most, compared with a typical laptop in the data: {names}.")
    if positives:
        d = positives[0]
        lines.append(f"Working in its favour: {d['Input']} ({d['Effect']:+.1f} points).")
    if not negatives and not positives:
        lines.append("All inputs are close to typical values, so no single factor stands out.")
    # Mention an engineered attribute only if one of its source inputs is actually pulling health down
    neg_cols = {c for d in negatives[:3] for c in d["Columns"]}
    notable = [c for c in comps if c["Attribute"] in C.ENGINEERED and c["Level"] == "high"
               and set(C.ENGINEERED[c["Attribute"]][1]) & neg_cols]
    if notable:
        c = notable[0]
        lines.append(f"{c['Attribute']} is higher than {c['Percentile']:.0f}% of laptops in the dataset.")
    advice = [C.ADVICE.get(d["Input"], C.ADVICE.get(CAPACITY_GROUP, "")) for d in negatives[:2]]
    advice = [a for a in advice if a]
    return lines, advice


# ============================================================ upload validation
def validate_upload(raw_df, current_df=None):
    """Check an uploaded CSV before it may replace the current data. Returns errors (block), warnings and info."""
    df = align_columns(raw_df.copy())
    errors, warnings, info = [], [], []
    required = [c for c in C.FEATURES if c != C.OPTIONAL_FEATURE] + [C.TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")
    if C.OPTIONAL_FEATURE not in df.columns:
        warnings.append(f"No {C.OPTIONAL_FEATURE} column: the switch to include it will be disabled.")
    extra = [c for c in df.columns if c not in C.FEATURES and c != C.TARGET]
    if extra:
        info.append(f"Extra columns that will be ignored: {', '.join(map(str, extra))}")
    if errors:
        return {"errors": errors, "warnings": warnings, "info": info, "clean": None}
    keep = [c for c in list(C.FEATURES) + [C.TARGET] if c in df.columns]
    clean = df[keep].copy()
    for c in keep:
        before = clean[c].notna().sum()
        clean[c] = _to_number(clean[c])
        lost = before - clean[c].notna().sum()
        if lost:
            warnings.append(f"{c}: {lost} value(s) are not numbers and will be treated as missing.")
    n = len(clean)
    no_target = int(clean[C.TARGET].isna().sum())
    clean = clean[clean[C.TARGET].notna()]
    dups = int(clean.duplicated().sum())
    clean = clean.drop_duplicates()
    if no_target:
        warnings.append(f"{no_target} row(s) have no {C.TARGET} and will be skipped.")
    if dups:
        warnings.append(f"{dups} duplicate row(s) will be removed.")
    if len(clean) < C.MIN_ROWS_ERROR:
        errors.append(f"Only {len(clean)} usable rows; at least {C.MIN_ROWS_ERROR} are needed.")
    elif len(clean) < C.MIN_ROWS_WARNING:
        warnings.append(f"Only {len(clean)} usable rows; results will be less reliable below {C.MIN_ROWS_WARNING}.")
    for c in keep:
        s = clean[c].dropna()
        unit = C.FEATURES[c]["unit"] if c in C.FEATURES else C.TARGET_UNIT
        bad = int((s < 0).sum() + ((s > 100).sum() if unit == "%" else 0))
        if bad:
            warnings.append(f"{c}: {bad} impossible value(s) (negative{' or above 100' if unit == '%' else ''}).")
        miss = int(clean[c].isna().sum())
        if miss:
            info.append(f"{c}: {miss} missing value(s) will be filled with the median.")
    info.insert(0, f"{n:,} rows read, {len(clean):,} usable after cleaning.")
    compare = None
    if current_df is not None:
        rows = []
        for c in keep:
            if c in current_df.columns:
                rows.append({"Attribute": c, "Current median": float(current_df[c].median()),
                             "New median": float(clean[c].median()),
                             "Current range": f"{current_df[c].min():,.1f} – {current_df[c].max():,.1f}",
                             "New range": f"{clean[c].min():,.1f} – {clean[c].max():,.1f}"})
        compare = pd.DataFrame(rows)
    return {"errors": errors, "warnings": warnings, "info": info, "clean": clean, "compare": compare}


# ============================================================ insights v2
def typical_value(c, stats):
    return stats[c]["mode"] if C.FEATURES[c]["kind"] == "binary" else stats[c]["median"]


def explain(model, row, features, inputs, stats):
    """Waterfall from a typical laptop to this laptop.
    base = prediction with every input at its typical value; each step = driver effect;
    'other' = remainder so that base + steps + other = this laptop's prediction exactly."""
    drv, final = drivers(model, row, features, inputs, stats)
    typical_row = {c: typical_value(c, stats) for c in inputs}
    base = float(predict_frame(model, pd.DataFrame([typical_row]), features)[0])
    shown = [d for d in drv if abs(d["Effect"]) >= 0.05][:C.WATERFALL_STEPS]
    other = final - base - sum(d["Effect"] for d in shown)
    return {"base": base, "final": final, "steps": shown, "other": other, "drivers": drv}


def what_if(drv, final, stats):
    """For changeable inputs that lower health: prediction if that input were typical = final − effect."""
    out = []
    for d in drv:
        if d["Effect"] <= -C.INSIGHT_MIN_EFFECT and all(c in C.CHANGEABLE for c in d["Columns"]):
            c = d["Columns"][0]
            out.append({"Input": d["Input"], "Value": d["Value"], "Typical": typical_value(c, stats),
                        "New": min(final - d["Effect"], 100.0), "Gain": -d["Effect"]})
    return out[:3]


def grouped_comparisons(comps, corr):
    """Put each compared attribute into Concerns / Normal / In your favour.
    Direction comes from its correlation with health in the data: a high value of an attribute that
    is negatively linked with health is a concern, and so on."""
    groups = {"Concerns": [], "Normal": [], "In your favour": []}
    for c in comps:
        r = corr.get(c["Attribute"], 0.0)
        c["Corr"] = r
        if c["Level"] == "typical" or abs(r) < C.MIN_CORR_FOR_DIRECTION:
            groups["Normal"].append(c)
        elif (c["Level"] == "high") == (r < 0):
            groups["Concerns"].append(c)
        else:
            groups["In your favour"].append(c)
    return groups


# ============================================================ 8. EDA (all variables)
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

BAND_ORDER = [b[1] for b in sorted(C.HEALTH_BANDS, key=lambda b: -b[0])]


def eda_columns(df):
    """All variables for EDA: every input present, the calculated attributes, and the target."""
    eng = add_engineered(df)
    inputs = [c for c in C.FEATURES if c in eng.columns]
    return eng, inputs, list(C.ENGINEERED), C.TARGET


def role_of(c, include_optional, kept):
    if c == C.TARGET:
        return "Target"
    if c in C.ENGINEERED:
        return "Calculated (in model)" if c in kept else "Calculated (insights only)"
    if c == C.OPTIONAL_FEATURE:
        return "Optional input (in model)" if include_optional else "Optional input (excluded)"
    return "Input"


def skew_text(s):
    if pd.isna(s):
        return "–"
    a = abs(s)
    shape = "approximately symmetric" if a < 0.5 else "moderately skewed" if a < 1 else "highly skewed"
    if a < 0.5:
        return shape
    return f"{shape} to the {'right (a long tail of high values)' if s > 0 else 'left (a long tail of low values)'}"


def kurt_text(k):
    if pd.isna(k):
        return "–"
    if k > 1:
        return "heavy tails: more extreme values than a normal distribution"
    if k < -1:
        return "flat: values spread evenly, few extremes"
    return "close to a normal distribution"


def univariate(eng, cols):
    rows = []
    for c in cols:
        s = eng[c].dropna()
        binary = s.nunique() <= 2
        rows.append({"Variable": c, "Count": int(s.size), "Missing": int(eng[c].isna().sum()), "Unique": int(s.nunique()),
                     "Mean": s.mean(), "Median": s.median(), "Std": s.std(), "Min": s.min(), "Q1": s.quantile(.25),
                     "Q3": s.quantile(.75), "Max": s.max(),
                     "Skewness": np.nan if binary else s.skew(), "Kurtosis": np.nan if binary else s.kurt()})
    out = pd.DataFrame(rows)
    out["Shape"] = [("yes/no variable" if pd.isna(sk) else skew_text(sk)) for sk in out["Skewness"]]
    out["Tails"] = [("–" if pd.isna(k) else kurt_text(k)) for k in out["Kurtosis"]]
    return out


def outlier_table(eng, cols):
    rows = []
    for c in cols:
        s = eng[c].dropna()
        if s.nunique() <= 2:
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_iqr = int(((s < lo) | (s > hi)).sum())
        z = (s - s.mean()) / s.std() if s.std() > 0 else s * 0
        rows.append({"Variable": c, "Lower fence": lo, "Upper fence": hi, "IQR outliers": n_iqr,
                     "IQR outliers %": n_iqr / len(s) * 100, "|z| > 3": int((z.abs() > 3).sum()),
                     "Below fence": int((s < lo).sum()), "Above fence": int((s > hi).sum())})
    return pd.DataFrame(rows)


def vif_table(eng, cols):
    """Variance inflation factor: how much each input is explained by the others. VIF = 1 / (1 − R²)."""
    X = eng[cols].copy()
    X = X.fillna(X.median())
    rows = []
    for c in cols:
        others = [o for o in cols if o != c]
        r2 = LinearRegression().fit(X[others], X[c]).score(X[others], X[c]) if others else 0.0
        vif = np.inf if r2 >= 0.9999 else 1 / (1 - r2)
        level = "high" if vif >= 10 else "moderate" if vif >= 5 else "low"
        rows.append({"Variable": c, "R² from other inputs": r2, "VIF": vif, "Multicollinearity": level})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def high_pairs(eng, cols, threshold=0.5):
    cm = eng[cols].corr()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = cm.loc[a, b]
            if abs(r) >= threshold:
                rows.append({"Variable A": a, "Variable B": b, "Correlation": r,
                             "Strength": "strong" if abs(r) >= 0.7 else "moderate"})
    return pd.DataFrame(rows).sort_values("Correlation", key=abs, ascending=False) if rows else pd.DataFrame(
        columns=["Variable A", "Variable B", "Correlation", "Strength"])


def binned_target(eng, col, bins=8):
    """Average battery health across the range of one variable (quantile bins; each value for yes/no)."""
    s = eng[[col, C.TARGET]].dropna()
    if s[col].nunique() <= 10:
        g = s.groupby(col)[C.TARGET].agg(["mean", "count"]).reset_index()
        g["label"] = g[col].map(lambda v: ("Yes" if v >= 0.5 else "No") if s[col].nunique() == 2 else f"{v:g}")
        g["mid"] = g[col]
    else:
        q = pd.qcut(s[col], bins, duplicates="drop")
        g = s.groupby(q, observed=True)[C.TARGET].agg(["mean", "count"]).reset_index()
        g["label"] = [f"{iv.left:,.3g}–{iv.right:,.3g}" for iv in g[col]]
        g["mid"] = [iv.mid for iv in g[col]]
    return g.rename(columns={"mean": "Mean health", "count": "Laptops"})


def band_labels(values):
    return [health_band(v)[0] for v in values]


def band_share(values):
    labels = pd.Series(band_labels(values))
    counts = labels.value_counts().reindex(BAND_ORDER, fill_value=0)
    return pd.DataFrame({"Band": BAND_ORDER, "Laptops": counts.values, "Share %": counts.values / max(len(labels), 1) * 100})


def band_metrics(splits):
    """Precision / recall of the health band, pooled over the test sets of all shuffles."""
    y = np.concatenate([a for a, _ in splits])
    p = np.clip(np.concatenate([b for _, b in splits]), 0, 100)
    yt, yp = band_labels(y), band_labels(p)
    cm = confusion_matrix(yt, yp, labels=BAND_ORDER)
    prec, rec, f1, sup = precision_recall_fscore_support(yt, yp, labels=BAND_ORDER, zero_division=0)
    pred_count = cm.sum(axis=0)
    rows = [{"Band": b, "Precision": (prec[i] if pred_count[i] else np.nan), "Recall": (rec[i] if sup[i] else np.nan),
             "F1": (f1[i] if sup[i] and pred_count[i] else np.nan), "Actual laptops": int(sup[i]),
             "Predicted as this band": int(pred_count[i])} for i, b in enumerate(BAND_ORDER)]
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else np.nan
    return {"table": pd.DataFrame(rows), "confusion": pd.DataFrame(cm, index=BAND_ORDER, columns=BAND_ORDER),
            "accuracy": acc, "n": int(cm.sum())}


def variable_decisions(df, include_optional, kept, selection, importance=None):
    """One row per variable: is it used in the model, and the evidence that justifies the decision."""
    eng, inputs, calc, target = eda_columns(df)
    corr = eng[inputs + calc + [target]].corr()[target]
    used_inputs = [c for c in inputs if c != C.OPTIONAL_FEATURE or include_optional]
    vif = vif_table(eng, used_inputs).set_index("Variable")["VIF"]   # among the inputs the model actually uses
    outl = outlier_table(eng, inputs + calc).set_index("Variable")["IQR outliers %"]
    imp = importance.set_index("Attribute")["Importance"] if importance is not None and len(importance) else pd.Series(dtype=float)
    sel = selection.set_index("Attribute") if selection is not None and len(selection) else pd.DataFrame()
    ratio_r = None
    if C.OPTIONAL_FEATURE in eng.columns and "Design Capacity" in eng.columns:
        ratio_r = float(np.corrcoef((eng[C.OPTIONAL_FEATURE] / eng["Design Capacity"] * 100).fillna(0), eng[target])[0, 1])
    rows = []
    for c in inputs + calc + [target]:
        r = float(corr.get(c, np.nan)) if c != target else 1.0
        strength = ("–" if c == target else "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.3
                    else "weak" if abs(r) >= 0.1 else "near zero")
        v = vif.get(c, np.nan)
        im = imp.get(c, np.nan)
        role = role_of(c, include_optional, kept)
        in_model = (c in kept) if c in calc else (c != target and (c != C.OPTIONAL_FEATURE or include_optional))
        notes = []
        if c == target:
            decision, why = "Predicted", "The value the model predicts (current capacity as % of design capacity)."
        elif c == C.OPTIONAL_FEATURE:
            decision = "Excluded by default" if not include_optional else "Included (user choice)"
            why = (f"Target leakage: {c} ÷ Design Capacity × 100 correlates r = {ratio_r:.3f} with {target}, so the model would "
                   "reproduce a ratio instead of learning from usage." if ratio_r is not None else "Target leakage.")
        elif c in calc:
            if c in sel.index:
                g = sel.loc[c, "Gain"]
                decision = "Used in model" if c in kept else "Not used in model"
                why = (f"Selection test: adding it changed cross-validated R² by {g:+.4f} "
                       f"({'≥' if c in kept else '<'} the {C.FEATURE_GAIN_THRESHOLD} threshold). "
                       + ("Kept because it improves accuracy." if c in kept else
                          "Its information is already in the original inputs, so it is used in the insights only."))
            else:
                decision, why = "Not used in model", "Not tested: a source input is excluded."
        else:
            decision = "Used in model"
            if strength == "strong":
                why = f"Strong link with {target} (r = {r:+.2f}): a core predictor."
            elif strength == "moderate":
                why = f"Moderate link with {target} (r = {r:+.2f}): a useful predictor."
            elif strength == "weak":
                why = (f"Weak link on its own (r = {r:+.2f}); kept because it can still add information in combination "
                       "with other inputs.")
            else:
                why = (f"Near-zero link with {target} (r = {r:+.2f}). Kept as a reference input (it gives scale to the capacity "
                       "values); the model relies on it very little — a candidate for removal in a simpler model.")
            if not np.isnan(im):
                notes.append("model importance ≈ 0 (the model barely uses it)" if im < 0.0005 else f"model importance {im:.3f}")
        if not np.isnan(v) and v >= 5:
            notes.append(f"VIF {v:.1f}: overlaps strongly with other inputs")
        if c in outl.index and outl[c] > 5:
            notes.append(f"{outl[c]:.1f}% outliers by the IQR rule (kept: genuine laptops)")
        rows.append({"Variable": c, "Role": role, "In model": "Yes" if in_model else "No", "Decision": decision,
                     "r with health": r if c != target else np.nan, "Strength": strength,
                     "VIF": v, "Outliers %": outl.get(c, np.nan),
                     "Importance": (0.0 if (not np.isnan(im) and im < 0.0005) else im),   # tiny negatives are noise
                     "Justification": why + (" " + "; ".join(notes)[0].upper() + "; ".join(notes)[1:] + "." if notes else "")})
    return pd.DataFrame(rows)
