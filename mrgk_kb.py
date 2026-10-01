"""
Mr. GK knowledge base: in-depth, structured answers for common questions, filled with live numbers from the app.

Each topic has trigger phrases (with weights), the guide sections it relates to, and an answer function that
receives the live app state. answer(question, state) picks the best topic, or builds an attribute profile when the
question names an attribute, and returns (markdown, section_ids) or (None, []).
"""
import re

import config as C

BANDS = sorted(C.HEALTH_BANDS, key=lambda b: -b[0])


# ------------------------------------------------------------------ helpers
def _f(v, d=2):
    try:
        return f"{float(v):,.{d}f}"
    except (TypeError, ValueError):
        return "–"


def _m(state, key, d=2):
    return _f(state.get("metrics", {}).get(key), d)


def _unit(c):
    return C.FEATURES[c]["unit"] if c in C.FEATURES else C.ENGINEERED_UNITS.get(c, "")


def _top_models(state, n=3):
    rows = list(state.get("table", {}).items())[:n]
    return "\n".join(f"{i + 1}. **{name}** — mean R² {_f(r['Mean R²'], 3)} ± {_f(r['R² spread (±)'], 3)}, "
                     f"MAE {_f(r['MAE'])} points, stability score {_f(r['Stability score'], 4)}"
                     for i, (name, r) in enumerate(rows))


def _strength(r):
    a = abs(r)
    return "strong" if a >= 0.7 else "moderate" if a >= 0.3 else "weak" if a >= 0.1 else "negligible"


def _pred_line(state):
    p = state.get("prediction")
    if not p:
        return ""
    return f"\n\nFor your latest prediction ({_f(p['pred'], 1)}%, {p['band']})"


def _sections(*ids):
    return list(ids)


# ------------------------------------------------------------------ topics
def a_fcc(s):
    corr = s.get("corr", {}).get(C.OPTIONAL_FEATURE)
    return f"""**In short:** Full Charge Capacity almost *is* the answer. Battery Health is essentially Full Charge Capacity ÷ Design Capacity × 100, so using it as an input lets the model copy a ratio instead of learning why batteries wear.

**How it works**
- Battery health is defined as the charge a battery can hold today compared with when it was new.
- Full Charge Capacity is exactly "the charge it can hold today", and Design Capacity is "when it was new".
- With both as inputs, the model learns the division and scores very highly — but it tells you nothing a calculator couldn't.

**The calculation**
`Health ≈ Full Charge Capacity ÷ Design Capacity × 100`
Example from the data: 44,961.9 ÷ 57,884 × 100 = **77.7%**, recorded health **76.9%**.

**In your app right now**
- {C.OPTIONAL_FEATURE} is **{'included' if s.get('include') else 'excluded'}**; the model uses **{len(s.get('inputs', []))} inputs**.
- Its correlation with health on its own is {_f(corr, 2) if corr is not None else '–'} (it only becomes near-perfect when divided by Design Capacity).
- The EDA page shows the ratio against health: the points fall almost on a straight line.

**What it means for you**
- Keep it **off** when the goal is to estimate health from *usage* (age, cycles, temperature, workload) — the more useful question when capacity is unknown.
- Switch it **on** (Model setup → Apply) only if users will always know their current full charge capacity; scores will jump, but the explanation of *why* gets weaker.""", _sections("p-fcc", "attr-fcc", "no-ratio")


def a_recommend(s):
    t = list(s.get("table", {}).items())
    tie = ""
    if len(t) > 1:
        gap = t[0][1]["Stability score"] - t[1][1]["Stability score"]
        if gap < 0.001:
            tie = (f"\n- The top two are **practically tied**: their stability scores differ by only {gap:.6f}. "
                   "Either is an equally sound choice; the simpler one is easier to explain.")
    return f"""**In short:** every model is trained and tested on 5 different shuffles of the data, and the one with the best **stability score = mean R² − spread** is recommended (⭐). It rewards models that are both accurate *and* consistent.

**How it works**
1. The data is split {s.get('split', '')} into training and test rows — 5 times, each with a different shuffle.
2. Each of the {len(t)} models is trained on the training rows and scored (R²) on the unseen test rows every time.
3. The 5 scores are averaged (mean R²) and their standard deviation measured (spread).
4. Models are ranked by mean R² − spread; the top one gets the ⭐.

**The calculation**
`Stability score = mean(R²₁…R²₅) − standard deviation(R²₁…R²₅)`
A model scoring 0.930 ± 0.003 (0.927) beats one scoring 0.932 ± 0.010 (0.922): slightly better on average, but less predictable.

**In your app right now**
{_top_models(s)}
- Active model: **{s.get('model')}** · recommended: **{s.get('recommended')}**{tie}

**What it means for you**
The ⭐ is a strong default, not a rule. On Model performance you can compare any model with the active one and press **Apply** to switch.""", _sections("p-recommend", "p-stability", "page-models")


def a_r2(s):
    return f"""**In short:** R² (R-squared) is the share of the differences in battery health that the model explains. 1.0 is perfect; 0 means no better than always guessing the average health.

**How it works**
- Imagine predicting every laptop at the average health: the total squared miss is the baseline.
- R² measures how much of that baseline miss the model removes.

**The calculation**
`R² = 1 − Σ(actual − predicted)² ÷ Σ(actual − average)²`

**In your app right now**
- Active model **{s.get('model')}**: mean R² **{_m(s, 'Mean R²', 3)}** ± {_m(s, 'R² spread (±)', 3)} across 5 test shuffles.
- That means it explains about **{float(s.get('metrics', {}).get('Mean R²', 0)) * 100:.0f}%** of the variation in battery health on laptops it has never seen.

**What it means for you**
- Above 0.9 is very good for this kind of data; compare models on the same split only.
- R² says how well the pattern is captured, not how big an error is in points — for that, look at **MAE** (±{_m(s, 'MAE')} points here).
- A very high R² with Full Charge Capacity switched on mostly reflects the capacity ratio, not understanding of wear.""", _sections("p-metrics", "g-r2")


def a_errors(s):
    return f"""**In short:** MAE, RMSE and MAPE all measure how far predictions are from the true health, in slightly different ways. Lower is better.

**The calculations**
- **MAE** (mean absolute error) = average of |actual − predicted| → the typical miss in health points.
- **RMSE** (root mean squared error) = √(average of (actual − predicted)²) → like MAE but punishes big misses more.
- **MAPE** = average of |actual − predicted| ÷ actual × 100 → the miss relative to the true value, in %.

**In your app right now** ({s.get('model')}, averaged over 5 shuffles)
- MAE **{_m(s, 'MAE')}** points · RMSE **{_m(s, 'RMSE')}** points · MAPE **{_m(s, 'MAPE %')}%**
- 90% of test predictions were within **±{_m(s, '90% error within (±)')}** points.

**How to read them together**
- RMSE close to MAE → errors are even; RMSE much larger than MAE → occasional big misses.
- The Predict page uses MAE for the "likely between … and …" range: prediction ± {_m(s, 'MAE')} points.

**What it means for you**
A prediction of 76.9% should be read as "about 77%, give or take {_m(s, 'MAE', 1)} points", not as an exact figure.""", _sections("p-metrics", "p-error")


def a_overfit(s):
    worst = max(s.get("table", {}).items(), key=lambda kv: kv[1]["Overfit gap"], default=(None, None))
    return f"""**In short:** overfitting is when a model memorises its training data, including noise, and does worse on new laptops. The **overfit gap** measures it.

**The calculation**
`Overfit gap = Train R² − mean test R²`
Above about **0.05** the model fits training data noticeably better than unseen data.

**In your app right now**
- Active model **{s.get('model')}**: train R² {_m(s, 'Train R²', 3)}, test R² {_m(s, 'Mean R²', 3)}, gap **{_m(s, 'Overfit gap', 3)}**.
- Largest gap: **{worst[0]}** ({_f(worst[1]['Overfit gap'], 3) if worst[1] else '–'}) — typical for flexible models such as trees and K-Nearest Neighbors (which scores 1.0 on training data by design).

**What it means for you**
A small gap means the reported test scores are a fair picture of real-world accuracy. The stability ranking already favours models that generalise, because it uses test scores only.""", _sections("p-metrics", "g-overfit")


def a_split(s):
    return f"""**In short:** the split decides how many rows the models learn from (training) and how many are hidden to test them (test). Options: 70/30, 75/25, 80/20 (default), 85/15.

**How it works**
- Training rows teach the model; test rows are never seen during training and measure accuracy on new laptops.
- More training data can help the model slightly; more test data makes the score more reliable.

**In your app right now**
- Split **{s.get('split')}**: {s.get('train_rows', 0):,} training rows and {s.get('test_rows', 0):,} test rows per shuffle, repeated 5 times.

**What it means for you**
- Changing the ratio mostly changes how the model is **tested**, not how good it is; scores move a little partly by chance.
- That is why every split is evaluated on 5 shuffles and the average is reported.
- To change it: sidebar → Model setup → choose a split → **Apply** (all models retrain, about 20–40 seconds).""", _sections("p-split", "p-stability")


def a_shuffles(s):
    return """**In short:** instead of one random split, the app repeats the split 5 times with fixed shuffles (random states 11, 23, 42, 57, 89) and averages the scores. You never pick the random state yourself — deliberately.

**Why**
- A single split can be lucky or unlucky; averaging 5 gives a stable, fair score and shows how much it varies (the spread).
- If users could change the random state freely, they could keep trying until a score looks great. The model wouldn't be better — the shuffle would just be lucky. Fixed shuffles remove that temptation.

**The calculation**
`Mean R² = (R²₁ + … + R²₅) ÷ 5` · `Spread = standard deviation of the five`

**What it means for you**
The "Score on every shuffle" chart on Model performance shows all 5 scores per model: tight boxes mean consistent models.""", _sections("p-stability")


def a_models(s):
    t = s.get("table", {})
    return f"""**In short:** {len(t)} regression models are compared, from simple straight-line models to tree ensembles. Each has strengths:

- **Linear / Ridge Regression** — straight-line relationships; fast and easy to explain. Ridge adds a small penalty that stabilises correlated inputs.
- **K-Nearest Neighbors** — averages the 10 most similar laptops; simple but memorises training data.
- **Decision Tree** — one tree of if/else rules; very readable, but unstable.
- **Random Forest / Extra Trees** — many trees averaged; robust and handle interactions.
- **Gradient Boosting / Hist Gradient Boosting** — trees built one after another, each fixing the last one's errors; often very accurate.
- **Support Vector Regression** — fits a smooth function, ignoring errors within ±0.5 points.
- **XGBoost / LightGBM** (if installed) — optimised boosting libraries.

**In your app right now** (top 3)
{_top_models(s)}

**What it means for you**
If simple linear models are on top, the relationships in your data are mostly straight-line. With the real data, tree models may take the lead if there are curves or interactions. Settings are sensible defaults rather than tuned per run.""", _sections("p-models", "p-recommend")


def a_ridge_linear(s):
    t = s.get("table", {})
    rl = ""
    if "Ridge Regression" in t and "Linear Regression" in t:
        d = t["Ridge Regression"]["Stability score"] - t["Linear Regression"]["Stability score"]
        gap = "less than 0.00001" if abs(d) < 1e-5 else f"{abs(d):.5f}"
        rl = (f"\n- Stability scores now: Ridge {_f(t['Ridge Regression']['Stability score'], 4)} vs Linear "
              f"{_f(t['Linear Regression']['Stability score'], 4)} — a difference of {gap}.")
    return f"""**In short:** Ridge is Linear Regression plus a small penalty on large coefficients. With thousands of rows and scaled inputs the penalty barely changes anything, so the two give practically the same predictions.

**How it works**
- Ridge minimises `Σ(actual − predicted)² + α × Σ(coefficient²)` with α = 1.
- The data term sums over thousands of training rows, so a penalty of 1 is tiny by comparison.
- The penalty matters with few rows or strongly overlapping inputs — not the case here.

**In your app right now**{rl}
- Whichever is ranked first wins by noise, not by a real difference.

**What it means for you**
Choose either. Linear Regression is the simpler one to explain.""", _sections("p-models", "p-recommend")


def a_engineered(s):
    kept = s.get("kept") or []
    rows = s.get("selection") or []
    table = "\n".join(f"- **{r['Attribute']}** (`{r['Formula']}`): gain {r['Gain']:+.4f} → {'kept' if r['Kept'] else 'not kept'}"
                      for r in rows)
    return f"""**In short:** five extra attributes are calculated from your inputs — you never type them. Each is a hypothesis about battery wear and is added to the model **only if it measurably improves accuracy**.

**The attributes**
""" + "\n".join(f"- **{n}** = `{f}` — {d}" for n, (f, _, d) in C.ENGINEERED.items()) + f"""

**How selection works**
1. On the training part of shuffle 1 only (test rows never influence the choice), score the model with the original inputs using 5-fold cross-validation.
2. Add one calculated attribute and score again.
3. Keep it if `gain = CV R² with − CV R² without ≥ {C.FEATURE_GAIN_THRESHOLD}`.

**In your app right now** (baseline CV R² {_f(s.get('baseline'), 4)})
{table or '- not available'}
- Kept: **{', '.join(kept) if kept else 'none'}**

**What it means for you**
"None kept" is a legitimate result: models can often find these combinations themselves. The attributes are still used on the Predict page to describe your laptop's usage profile.""", _sections("engineered", "p-select")


def a_age(s):
    return """**In short:** Battery Age is treated as a **number**, not a category, because 6 years really is more than 5, and each extra year means the same thing.

**Why it matters**
- As a category, the model would lose the order and learn each age as an unrelated label, with fewer examples per age.
- It could not reason about an age it has never seen (e.g. 11 years).
- The input would become a dropdown instead of a free-text box.

**Background**
A generic rule treated any whole-number column with ≤ 10 distinct values as a category — meant for coded columns like "region 1–3". Battery Age (1–10) was caught by it, so this project sets it explicitly to numeric. Gaming User (0/1) behaves the same either way.""", _sections("p-types", "attr-age")


def a_prep(s):
    return """**In short:** before any model sees the data, two steps run inside the model pipeline, learned from training rows only.

**1. Median imputation** — blank values are filled with the training median of that column. The median is used because extreme values don't pull it.

**2. Standard scaling** — each column is rescaled:
`z = (value − mean) ÷ standard deviation`
This puts Design Capacity (tens of thousands) and Gaming User (0/1) on comparable scales, which matters for K-Nearest Neighbors, Support Vector Regression and Ridge. Tree models don't need it but aren't harmed.

**Why inside the pipeline**
The medians and scaling are re-learned on each training split, so test rows never influence them — this prevents a subtle form of data leakage.

**Other cleaning**
Rows without Battery Health and exact duplicates are removed; outliers are kept (they're genuine laptops) and flagged with range warnings at prediction time.""", _sections("p-prep", "p-clean")


def a_bands(s):
    lines = "\n".join(f"- **{label}**: {lo}% and above" if i == 0 else f"- **{label}**: {lo}% to below {BANDS[i - 1][0]}%"
                      for i, (lo, label, _) in enumerate(BANDS))
    return f"""**In short:** the predicted health is placed in one of three bands, set in `config.py` (`HEALTH_BANDS`).

{lines}

**How it works**
- Predictions are clipped to 0–100% (health outside that range is impossible) and then banded.
- The bands are **domain rules**, not something the model learns; they drive the gauge colour, the message and the scale bar.

**What it means for you**
Near a boundary (e.g. 70.5%), remember the typical error of ±{_m(s, 'MAE', 1)} points: the battery could be on either side of it.""", _sections("p-predict", "attr-health")


def a_error_range(s):
    return f"""**In short:** a regression model doesn't produce a "confidence %". Instead the app shows how wrong the active model typically is, measured on unseen test data.

**The calculations**
- **Likely range** = prediction ± MAE → currently ± **{_m(s, 'MAE')}** points.
- **90% within ±q** → q = 90th percentile of |actual − predicted| over all test rows of all 5 shuffles → currently **±{_m(s, '90% error within (±)')}** points.

**Example**
Prediction 76.9% → likely between {76.9 - float(s.get('metrics', {}).get('MAE', 0)):.1f}% and {76.9 + float(s.get('metrics', {}).get('MAE', 0)):.1f}%; 9 in 10 laptops land within ±{_m(s, '90% error within (±)', 1)} points.

**What it means for you**
These are averages. For inputs outside the ranges seen in the data (the app warns you), the real error can be larger.""", _sections("p-error")


def a_insights(s):
    return f"""**In short:** the Insights explain *why* the model gave this estimate, by testing what happens when each input is made typical.

**The core calculation (used everywhere)**
`Effect of X = prediction(your inputs) − prediction(your inputs with X set to its typical value)`
Typical = the median in the data (most common value for Yes/No). Negative → X lowers health versus a typical laptop.

**The four panels**
1. **Conclusion** — the inputs pulling health down most (effect ≤ −{C.INSIGHT_MIN_EFFECT}), the one working in its favour, and matching advice.
2. **What if** — for everyday habits only, the estimate if that habit were typical: `your prediction − effect`.
3. **From a typical laptop to yours** (waterfall) — starts from a laptop with every input typical, then adds each effect, biggest first. A small "combined effects" step makes the total exact, because inputs interact.
4. **How your laptop compares** — each input in words ("higher than 97% of laptops"), grouped as Concerns / In your favour / Normal by its effect.

**Caveats**
These are associations learned from the data, not proof of cause and effect; effects of correlated inputs (Cycle Count and Battery Age) are partly shared.{_pred_line(s) + ', ask me "why did my laptop get this prediction?" for the full breakdown.' if s.get('prediction') else ''}""", _sections("p-insights", "ins-waterfall", "ins-whatif", "ins-percentile")


def a_whatif(s):
    ch = ", ".join(C.CHANGEABLE)
    return f"""**In short:** "What if" shows the estimated health if one everyday habit were typical, with everything else unchanged.

**The calculation**
`What-if prediction = your prediction − effect of that input` (effect as defined in the Insights).
Shown for up to three habits that lower the estimate by at least {C.INSIGHT_MIN_EFFECT} points.

**Which inputs count as habits**
{ch}. Battery Age, Cycle Count and the capacities reflect history or hardware, so they are named but not offered as what-ifs.

**What it means for you**
It is the model's estimate, not a promise: it shows which habits are associated with lower health in this data, and by roughly how much.""", _sections("ins-whatif")


def a_percentile(s):
    return """**In short:** "Higher than 97% of laptops" means only 3% of laptops in the data have a higher value.

**The calculation**
`Percentile = (share of laptops with a lower value + ½ × share with an equal value) × 100`
The half-weight for equal values keeps whole-number inputs like Battery Age from being pushed to an extreme.

**How it's worded**
- Between the 35th and 65th percentile → "close to typical".
- Inputs are grouped by their effect on *your* estimate: **Concerns** (lower it by ≥ 0.5 points), **In your favour** (raise it by ≥ 0.5), **Normal**.""", _sections("ins-percentile")


def a_importance(s):
    imp = s.get("importance") or []
    top = "\n".join(f"- **{r['Attribute']}**: R² drops by {_f(r['Importance'], 3)} when shuffled" for r in imp[:5])
    return f"""**In short:** feature importance shows which inputs the model relies on most across *all* laptops.

**The calculation (permutation importance)**
`Importance of X = R²(test data) − R²(test data with column X randomly shuffled)`, averaged over 5 repeats.
Shuffling breaks the link between X and health; the bigger the drop, the more the model depends on X.

**In your app right now** ({s.get('model')})
{top or '- Open the Model performance page once to calculate it.'}

**Global vs local**
Feature importance is **global** (all laptops). The Insights on the Predict page are **local** (one laptop). Correlated inputs share credit, so each can look less important than their combined effect.""", _sections("p-importance")


def a_correlation(s):
    corr = s.get("corr", {})
    if corr:
        ranked = sorted(corr.items(), key=lambda kv: kv[1])
        neg = [kv for kv in ranked if kv[1] < -0.1][:3]
        pos = [kv for kv in reversed(ranked) if kv[1] > 0.1][:2]
        neg_t = "\n".join(f"- **{k}**: r = {v:+.2f} ({_strength(v)}) — higher values go with **lower** health" for k, v in neg)
        pos_t = "\n".join(f"- **{k}**: r = {v:+.2f} ({_strength(v)}) — higher values go with **higher** health" for k, v in pos)
    else:
        neg_t = pos_t = ""
    return f"""**In short:** the correlation chart shows how strongly each attribute moves together with battery health, from −1 to +1.

**How to read it**
- **Red bars (negative)**: when the attribute is higher, battery health tends to be **lower**.
- **Green bars (positive)**: when the attribute is higher, health tends to be **higher**.
- **Length** = strength: below 0.3 weak, 0.3–0.7 moderate, above 0.7 strong. Near 0 = no straight-line link.
- ƒ marks calculated attributes.

**In your data right now**
{neg_t}
{pos_t}

**Watch out for**
- Correlation is not causation, and it only captures straight-line relationships.
- Full Charge Capacity looks only moderately linked on its own, yet divided by Design Capacity it almost *is* health (see "why is Full Charge Capacity off").
- Inputs that move together (Cycle Count & Battery Age, CPU/GPU/Power/Temperature) share their link with health.""", _sections("page-eda", "p-insights")


def a_versions(s):
    return f"""**In short:** the Data page lets you replace the training CSV safely. Every version is kept and any can be restored.

**How to replace the data**
1. **Data** page → choose **Upload new CSV** → pick the file.
2. Read the **check report**: missing columns or fewer than {getattr(C, 'MIN_ROWS_REQUIRED', 30)} usable rows block the upload; non-numeric values, rows without Battery Health, duplicates and impossible values are warnings. A table compares medians and ranges with the current data.
3. Add an optional note and press **Replace** → saved as the next version, activated, all models retrain.

**Restore**
Version history → pick an earlier version → **Restore**. Nothing is ever deleted.

**In your app right now**
Active data: **version v{s.get('version')}**, {s.get('rows', 0):,} rows.

**Storage**
On your computer, versions live in `data/versions/`. When deployed with `GITHUB_TOKEN` and `GITHUB_REPO` in secrets, each Replace/Restore is also committed to GitHub, so versions survive restarts.""", _sections("p-versions", "replace-data", "page-data")


def a_batch(s):
    return f"""**In short:** Batch prediction scores many laptops at once from a CSV.

**Steps**
1. **Batch prediction** page → **Download template** (correct headers + one example row of typical values).
2. Fill one row per laptop with the {len(s.get('inputs', []))} inputs (Gaming User: 1/0 or Yes/No).
3. Upload → each row gets a **predicted health**, a **band** and **notes** on blank or out-of-range values → **Download predictions**.

**Good to know**
- Column names are matched ignoring capitals, spaces and punctuation; extra columns are kept but ignored.
- Blank cells are filled with the training median and flagged.
- If your file also contains Battery Health, the page reports the average error (MAE) against those true values — a quick accuracy check on new data.""", _sections("p-batch", "page-batch")


def a_limitations(s):
    return """**In short:** the tool is useful, but its conclusions are only as good as the data and the method. Main limits:

**Data**
- The data-collecting agency and methods are unknown, so results describe *this* dataset and may not transfer to other brands or climates.
- Capacity units are uncertain (mAh vs mWh), and definitions like "average temperature" aren't documented.
- Known drivers such as charging habits and climate are missing.
- One snapshot per laptop: the tool can't track how a battery wears over time.

**Model**
- Insights are associations, not proof of cause and effect.
- Inputs outside the data's range give less reliable predictions (you get a warning).
- Correlated inputs share their effects; hyper-parameters are defaults, not tuned.
- ± error ranges are averages, not guarantees.

**Operations**
- Anyone with the app link can replace the data (Restore undoes it).
- Free hosting sleeps when idle and retrains on wake (~30 s).
- Mr. GK can be wrong — check the cited sections.""", _sections("limitations", "lim-data", "lim-model")


def a_units(s):
    return """**In short:** the capacity values (about 40,000–80,000) are labelled **mAh**, but they look more like **mWh**.

- As mAh they'd be roughly ten times a typical laptop battery.
- As mWh (≈ 40–80 Wh) they match typical laptops.

The label doesn't affect any calculation — only what's displayed. Confirm the unit with the data source and change it in `config.py` (`FEATURES → Design Capacity / Full Charge Capacity → unit`).""", _sections("unit-note", "attr-design")


def a_mrgk(s):
    return f"""**In short:** I'm {C.MRGK_NAME}, the assistant built into Battery Clinic AI. I answer questions about the data, models, calculations and your predictions, using the user guide and the live app.

**How I answer**
- For common topics I use a built-in knowledge base with live numbers from your app — like this answer.
- In **AI mode** (when an `ANTHROPIC_API_KEY` is set) I also write conversational answers grounded in the guide and the app state, up to {C.MRGK_DAILY_CAP} per day.
- In **Guide mode** (no key, or daily limit reached) I combine the knowledge base with the most relevant guide sections.
- The 📖 buttons under an answer open the guide at the source section.
- I float over every page (the Mr. GK button at the top right opens and closes me), so you can ask while looking at a chart or prediction; common questions are under **FAQ**.

**Limits**
I can be wrong or oversimplify; check the cited sections. In AI mode, questions are sent to the AI provider, so don't type personal information.""", _sections("p-mrgk", "page-guide")


def a_variables(s):
    stats, corr = s.get("stats", {}), s.get("corr", {})
    lines = []
    for c, spec in C.FEATURES.items():
        st = stats.get(c)
        used = "used" if c in s.get("inputs", []) else "not used now"
        rng = f"{_f(st['min'], spec['decimals'])}–{_f(st['max'], spec['decimals'])} {spec['unit']}".strip() if st else ""
        lines.append(f"- **{c}** ({used}) — {spec['desc']} Range {rng}; correlation with health {corr.get(c, 0):+.2f}.")
    return f"""**In short:** the dataset has **{len(C.FEATURES) + 1} variables**: **{len(C.FEATURES)} inputs** and the target **{C.TARGET}** (%). The model currently uses **{len(s.get('inputs', []))}** inputs{'' if s.get('include') else f' ({C.OPTIONAL_FEATURE} is switched off)'}.

**The inputs**
""" + "\n".join(lines) + f"""

**The target**
- **{C.TARGET}** — current capacity as a % of design capacity; bands Good ≥ 70, Moderate 40–70, Poor < 40.

**Plus {len(C.ENGINEERED)} calculated attributes**
{', '.join(C.ENGINEERED)} — computed from the inputs; kept in the model only if they improve accuracy (kept now: {', '.join(s.get('kept') or []) or 'none'}).""", _sections("data", "attributes", "engineered")


def a_rows(s):
    r = s.get("report", {})
    return f"""**In short:** the active data (**version v{s.get('version')}**) has **{s.get('rows', 0):,} rows** after cleaning — one laptop per row.

**How that number is reached**
- Rows read from the file: {r.get('rows_raw', '–'):,}
- Removed without a {C.TARGET} value: {r.get('missing_target_removed', 0)}
- Removed as exact duplicates: {r.get('duplicates_removed', 0)}

**How they're used**
Each shuffle trains on {s.get('train_rows', 0):,} rows and tests on {s.get('test_rows', 0):,} ({s.get('split')} split); the final model is refitted on all {s.get('rows', 0):,} rows for predictions.""", _sections("p-load", "p-clean", "p-final")


def a_final(s):
    return f"""**In short:** after the comparison, the chosen model is trained once more on **all {s.get('rows', 0):,} rows**, and that model makes your predictions.

**Why**
The 5 shuffles exist to *measure* performance fairly. For real predictions it's better to learn from every available laptop. The reported scores are therefore slightly conservative: the final model has seen 15–30% more data than during testing.""", _sections("p-final")


def a_deploy(s):
    return """**In short:** the app runs anywhere Python runs; for sharing, a free Streamlit Community Cloud deployment works well.

**Steps**
1. Put the project folder on GitHub (`secrets.toml` is excluded by `.gitignore`).
2. share.streamlit.io → **New app** → choose the repository and `app.py`.
3. In the app's **Settings → Secrets**, paste `ANTHROPIC_API_KEY` (for Mr. GK's AI mode) and `GITHUB_TOKEN` / `GITHUB_REPO` (to store data versions in GitHub).

**Good to know**
Free hosting sleeps after inactivity; the next visit wakes it and retrains (~30 seconds). No other services are needed.""", _sections("deploy", "run-local")


def a_prediction(s):
    p = s.get("prediction")
    if not p:
        return ("**I don't see a prediction yet.** Go to **Predict**, enter your laptop's details and press **Predict battery "
                "health** — then ask me again and I'll walk you through every factor behind the estimate."), _sections("page-predict")
    drivers = p.get("drivers", [])
    base = p.get("base")
    steps = "\n".join(f"- **{d['Input']}**: {d['Effect']:+.1f} points" for d in drivers if abs(d["Effect"]) >= 0.05)
    neg = [d for d in drivers if d["Effect"] <= -C.INSIGHT_MIN_EFFECT]
    pos = [d for d in drivers if d["Effect"] >= C.INSIGHT_MIN_EFFECT]
    wi = p.get("whatif", [])
    stats = s.get("stats", {})
    row = p.get("row", {})
    compare = []
    for d in neg[:3]:
        c = d["Columns"][0] if len(d.get("Columns", [])) == 1 else None
        if c and c in stats and C.FEATURES[c]["kind"] != "binary":
            compare.append(f"- **{c}** is {_f(row.get(c), C.FEATURES[c]['decimals'])} {_unit(c)} vs a typical "
                           f"{_f(stats[c]['median'], C.FEATURES[c]['decimals'])} {_unit(c)}.")
    what = "\n".join(f"- If **{w['Input']}** were typical, the estimate would be **{w['New']:.1f}%** ({w['Gain']:+.1f})." for w in wi)
    return f"""**In short:** your laptop is estimated at **{p['pred']:.1f}% ({p['band']})** by {s.get('model')}. {('The biggest factors pulling it down are ' + ' and '.join(d['Input'] for d in neg[:2]) + '.') if neg else 'No single input stands out.'}

**Step by step, from a typical laptop to yours**
- Typical laptop (every input at its median): **{_f(base, 1)}%**
{steps}
- Result: **{p['pred']:.1f}%** (a small remainder comes from inputs interacting)

**Why these inputs matter**
{chr(10).join(compare) or '- Your inputs are close to typical values.'}
{('- Working in your favour: **' + pos[0]['Input'] + f"** ({pos[0]['Effect']:+.1f} points).") if pos else ''}

**What if**
{what or '- None of the everyday habits is lowering the estimate noticeably.'}

**How sure is this?**
The model's typical error is ±{_f(p.get('mae'), 1)} points, so the true value is most likely between {p['pred'] - p.get('mae', 0):.1f}% and {min(p['pred'] + p.get('mae', 0), 100):.1f}%. These are patterns in the data, not proof of cause and effect.""", _sections("p-insights", "ins-waterfall", "ins-whatif")


def a_precision(s):
    b = s.get("bands")
    rows = ""
    if b:
        for r in b["table"]:
            p = "–" if r["Precision"] != r["Precision"] else f"{r['Precision']:.2f}"
            rc = "–" if r["Recall"] != r["Recall"] else f"{r['Recall']:.2f}"
            rows += f"\n- **{r['Band']}**: precision {p}, recall {rc} ({r['Actual laptops']:,} test laptops)"
    return f"""**In short:** precision and recall are **classification** metrics, but this app does **regression** — it predicts a number (health %). So its main metrics are R², MAE and RMSE. Because every prediction is also placed in a health band, the app measures **band precision and recall** on the Model performance page.

**The calculations**
- `Precision (band) = laptops correctly predicted in the band ÷ all laptops predicted in the band`
- `Recall (band) = laptops correctly predicted in the band ÷ all laptops really in the band`
- `F1 = 2 × precision × recall ÷ (precision + recall)` · `Band accuracy = correct bands ÷ all predictions`

**In your app right now** ({s.get('model')}, pooled over the test sets of all 5 shuffles{f", {b['n']:,} predictions" if b else ''})
- Band accuracy: **{(b['accuracy'] * 100) if b else float('nan'):.1f}%**{rows}

**What it means for you**
- High precision for a band = when the app says "Moderate", it is usually right.
- High recall = the app rarely misses laptops that really are Moderate.
- Laptops just above or below a band boundary (70% or 40%) are the ones most often misplaced, because a ±{_m(s, 'MAE', 1)}-point error can cross the line.
- A band with no test laptops shows "–": it cannot be measured.""", _sections("p-bandacc", "p-metrics")


def a_skew(s):
    uni = (s.get("eda") or {}).get("uni", [])
    sk = sorted([u for u in uni if u.get("Skewness") == u.get("Skewness")], key=lambda u: -abs(u["Skewness"]))[:4]
    lines = "\n".join(f"- **{u['Variable']}**: skewness {u['Skewness']:+.2f} ({u['Shape']}), kurtosis {u['Kurtosis']:+.2f} ({u['Tails']})" for u in sk)
    return f"""**In short:** skewness describes how **lopsided** a variable's distribution is; kurtosis describes how **heavy its tails** are (how many extreme values).

**How to read them**
- **Skewness**: −0.5 to 0.5 roughly symmetric · 0.5–1 moderately skewed · above 1 highly skewed. Positive = a long tail of high values; negative = a long tail of low values.
- **Kurtosis** (excess, normal = 0): above 1 = heavy tails, more extremes · below −1 = flat, values spread evenly.

**The calculations**
`Skewness = mean((x − mean)³) ÷ std³` · `Kurtosis = mean((x − mean)⁴) ÷ std⁴ − 3` (with small-sample corrections)

**In your data right now** (most skewed)
{lines or '- Open the EDA page once to calculate them.'}

**What it means for you**
Skewed inputs are fine for tree models; for linear models, strong skew can let a few extreme laptops pull the fit. The app keeps these values and warns at prediction time when an input is outside the data's range. See **EDA & insights → Univariate**.""", _sections("p-edastats", "page-eda")


def a_vif(s):
    vif = (s.get("eda") or {}).get("vif", [])
    lines = "\n".join(f"- **{v['Variable']}**: VIF {v['VIF']:.1f} ({v['Multicollinearity']})" for v in vif[:5])
    return f"""**In short:** multicollinearity means some inputs carry largely the same information. **VIF** (variance inflation factor) measures it for each input.

**The calculation**
`VIF = 1 ÷ (1 − R²)`, where R² is how well the *other* inputs predict this one. Below 5 fine · 5–10 moderate · above 10 high.

**In your app right now** (inputs the model uses)
{lines or '- Open the EDA page once to calculate it.'}

**What it means for you**
- High overlap doesn't hurt prediction accuracy much, but it makes individual effects harder to separate — e.g. Cycle Count and Battery Age partly share their link with health.
- That's why the Insights say effects of correlated inputs are "partly shared".
- If Full Charge Capacity is switched on, it and Design Capacity overlap strongly (they are nearly proportional). See **EDA & insights → Multivariate**.""", _sections("p-edastats", "page-eda")


def a_outliers(s):
    out = (s.get("eda") or {}).get("out", [])
    worst = sorted(out, key=lambda o: -o["IQR outliers %"])[:4]
    lines = "\n".join(f"- **{o['Variable']}**: {o['IQR outliers']:,} by IQR ({o['IQR outliers %']:.1f}%), {o['|z| > 3']:,} with |z| > 3" for o in worst)
    return f"""**In short:** outliers are unusually low or high values. The app detects them two ways and **keeps** them.

**The rules**
- **IQR rule**: outside `Q1 − 1.5 × IQR` to `Q3 + 1.5 × IQR` (IQR = Q3 − Q1).
- **z-score rule**: more than 3 standard deviations from the mean, `|z| = |x − mean| ÷ std > 3`.

**In your data right now** (most outliers)
{lines or '- Open the EDA page once to calculate them.'}

**Why they are kept**
They are genuine laptops, not errors; tree models are robust to them; scaling limits their effect on linear models; and the Predict page warns when an input is outside the data's range. See **EDA & insights → Outliers**.""", _sections("p-edastats", "p-clean")


def a_selection(s):
    dec = (s.get("eda") or {}).get("dec", [])
    used = [d["Variable"] for d in dec if d["In model"] == "Yes"]
    notused = [d for d in dec if d["In model"] == "No" and d["Role"] != "Target"]
    lines = "\n".join(f"- **{d['Variable']}** — {d['Decision']}: {d['Justification']}" for d in notused)
    weak = [d for d in dec if d["In model"] == "Yes" and d["Strength"] in ("weak", "near zero")]
    wl = "\n".join(f"- **{d['Variable']}** (r = {d['r with health']:+.2f}): {d['Justification']}" for d in weak[:3])
    return f"""**In short:** of {len(dec)} variables, the model uses **{len(used)}** ({', '.join(used)}). The others are left out for evidence-based reasons, shown on **EDA & insights → Variable selection**.

**Not used, and why**
{lines or '- Every variable is used.'}

**Used, but with weak links on their own**
{wl or '- None.'}

**How the evidence is gathered**
- Correlation with health (Bivariate tab), overlap with other inputs — VIF (Multivariate tab), outliers (Outliers tab), and the active model's permutation importance.
- Calculated attributes must improve cross-validated R² by at least {C.FEATURE_GAIN_THRESHOLD} to enter the model.
- Full Charge Capacity is excluded because of target leakage, not weakness.""", _sections("p-selectionvars", "p-select", "p-fcc")


# ------------------------------------------------------------------ attribute profile
ALIASES = {"age": "Battery Age", "cycles": "Cycle Count", "cycle": "Cycle Count", "temperature": "Average Temperature",
           "temp": "Average Temperature", "heat": "Average Temperature", "gaming": "Gaming User", "gamer": "Gaming User",
           "cpu": "CPU Usage", "gpu": "GPU Usage", "power": "Power Consumption", "usage hours": "Daily Usage Hours",
           "daily usage": "Daily Usage Hours", "design capacity": "Design Capacity", "full charge": "Full Charge Capacity",
           "battery health": C.TARGET}


def find_attribute(question):
    q = question.lower()
    names = sorted(list(C.FEATURES) + list(C.ENGINEERED), key=len, reverse=True)
    for n in names:
        if n.lower() in q:
            return n
    for alias, n in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(alias)}\b", q) and n != C.TARGET:
            return n
    return None


def a_attribute(name, s):
    stats, corr = s.get("stats", {}), s.get("corr", {})
    st = stats.get(name, {})
    r = corr.get(name)
    if name in C.FEATURES:
        spec = C.FEATURES[name]
        what, why, unit, dec = spec["desc"], spec["why"], spec["unit"], spec["decimals"]
        status = ("used by the model" if name in s.get("inputs", []) else
                  f"not used right now ({C.OPTIONAL_FEATURE} is switched off)")
        section = {"Battery Age": "attr-age", "Daily Usage Hours": "attr-hours", "Gaming User": "attr-gaming",
                   "Design Capacity": "attr-design", "Cycle Count": "attr-cycles", "CPU Usage": "attr-cpu",
                   "GPU Usage": "attr-gpu", "Power Consumption": "attr-power", "Average Temperature": "attr-temp",
                   "Full Charge Capacity": "attr-fcc"}.get(name, "attributes")
        formula = ""
        used_in = [n for n, (_, src, _) in C.ENGINEERED.items() if name in src]
    else:
        f, src, what = C.ENGINEERED[name]
        why, unit, dec = "A calculated attribute: a hypothesis about how usage combines to wear the battery.", C.ENGINEERED_UNITS.get(name, ""), 1
        status = "kept in the model" if name in (s.get("kept") or []) else "used in the insights only (it did not improve accuracy enough to be kept)"
        section = {"Cycles per Year": "eng-cpy", "Heat Stress": "eng-heat", "Workload Intensity": "eng-wi",
                   "Power per Usage Hour": "eng-pph", "Gaming Load": "eng-gl"}.get(name, "engineered")
        formula = f"\n**Formula:** `{name} = {f}`\n"
        used_in = []
    imp = next((x for x in (s.get("importance") or []) if x["Attribute"] == name), None)
    eda = s.get("eda") or {}
    u = next((x for x in eda.get("uni", []) if x["Variable"] == name), None)
    o = next((x for x in eda.get("out", []) if x["Variable"] == name), None)
    dcs = next((x for x in eda.get("dec", []) if x["Variable"] == name), None)
    shape = ""
    if u and u.get("Skewness") == u.get("Skewness"):
        shape += f"\n- Shape: skewness {u['Skewness']:+.2f} ({u['Shape']}); kurtosis {u['Kurtosis']:+.2f} ({u['Tails']})."
    if o:
        shape += f"\n- Outliers: {o['IQR outliers']:,} by the IQR rule ({o['IQR outliers %']:.1f}%), {o['|z| > 3']:,} with |z| > 3."
    decision = f"\n\n**Used in the model?** {dcs['In model']} — {dcs['Justification']}" if dcs else ""
    p = s.get("prediction")
    mine = ""
    if p and name in p.get("row", {}):
        eff = next((d["Effect"] for d in p.get("drivers", []) if name in d.get("Columns", [])), None)
        v = p["row"][name]
        vtxt = ("Yes" if v >= 0.5 else "No") if C.FEATURES.get(name, {}).get("kind") == "binary" else f"{_f(v, dec)} {unit}"
        mine = (f"\n**In your latest prediction**\n- Your value: **{vtxt}** (typical {_f(st.get('median'), dec)} {unit})"
                + (f"\n- Effect on your estimate: **{eff:+.1f} points** versus a typical laptop." if eff is not None else ""))
    return f"""**{name}** — {what}
{formula}
**Why it may matter:** {why}

**In your data**
- Range {_f(st.get('min'), dec)} – {_f(st.get('max'), dec)} {unit}, median {_f(st.get('median'), dec)} {unit}.
- Correlation with {C.TARGET}: **{f'{r:+.2f}' if r is not None else '–'}** ({_strength(r) if r is not None else '–'}) — {('higher values go with lower health' if r is not None and r < -0.1 else 'higher values go with higher health' if r is not None and r > 0.1 else 'little straight-line link with health')}.
- Status: {status}.{shape}{f'''
- Feature importance ({s.get('model')}): R² drops by {_f(imp['Importance'], 3)} when it is shuffled.''' if imp else ''}{f'''
- Used to calculate: {', '.join(used_in)}.''' if used_in else ''}
{mine}{decision}

**Keep in mind:** correlation shows association, not cause; inputs that move together share their link with health.""", [section, "p-insights"]


# ------------------------------------------------------------------ router
TOPICS = [
    ("prediction", a_prediction, [("my laptop", 4), ("my prediction", 4), ("my result", 4), ("this prediction", 4),
                                  ("my battery", 4), ("why did i get", 4), ("explain my", 4), ("my estimate", 4)]),
    ("fcc", a_fcc, [("full charge capacity", 3), ("fcc", 3), ("off by default", 3), ("leak", 3), ("capacity ratio", 3)]),
    ("ridge_linear", a_ridge_linear, [("ridge", 3), ("linear regression", 2), ("tie", 2), ("same score", 2)]),
    ("recommend", a_recommend, [("recommended", 4), ("best model", 4), ("star", 2), ("chosen", 2), ("stability score", 4),
                                ("choose", 1), ("why is", 0.5), ("winner", 3), ("ranked", 3), ("ranking", 3)]),
    ("r2", a_r2, [("r²", 4), ("r2", 4), ("r-squared", 4), ("r squared", 4), ("coefficient of determination", 4)]),
    ("errors", a_errors, [("mae", 4), ("rmse", 4), ("mape", 4), ("error", 1.5), ("accuracy", 2), ("accurate", 2)]),
    ("overfit", a_overfit, [("overfit", 4), ("train r", 3), ("memoris", 3), ("memoriz", 3), ("gap", 2)]),
    ("split", a_split, [("split", 3), ("70/30", 3), ("80/20", 3), ("ratio", 2), ("test size", 3), ("training rows", 2)]),
    ("shuffles", a_shuffles, [("shuffle", 4), ("random state", 4), ("seed", 4), ("stability check", 3), ("spread", 2)]),
    ("models", a_models, [("which models", 4), ("models are", 3), ("candidate", 3), ("algorithm", 3),
                          ("random forest", 2), ("gradient boosting", 2), ("xgboost", 2), ("svr", 2), ("knn", 2),
                          ("decision tree", 2), ("how many models", 4)]),
    ("engineered", a_engineered, [("calculated attribute", 4), ("engineered", 4), ("feature engineering", 4),
                                  ("derived", 3), ("new attribute", 3), ("kept", 2)]),
    ("age", a_age, [("categorical", 3), ("category", 3), ("numeric", 2), ("battery age a number", 4)]),
    ("prep", a_prep, [("scaling", 4), ("impute", 4), ("imputation", 4), ("missing value", 3), ("preprocess", 4),
                      ("standardi", 3), ("normali", 3), ("clean", 2)]),
    ("bands", a_bands, [("band", 4), ("good", 1.5), ("moderate", 1.5), ("poor", 1.5), ("threshold", 3), ("gauge", 2)]),
    ("error_range", a_error_range, [("likely range", 4), ("confidence", 4), ("how sure", 4), ("uncertain", 3),
                                    ("90%", 3), ("plus or minus", 3), ("±", 3)]),
    ("insights", a_insights, [("insight", 3), ("driver", 3), ("waterfall", 4), ("typical laptop", 3), ("conclusion", 2),
                              ("effect", 2)]),
    ("whatif", a_whatif, [("what if", 4), ("what-if", 4), ("habit", 3), ("improve", 2)]),
    ("percentile", a_percentile, [("percentile", 4), ("higher than", 3), ("compares", 3), ("concern", 3)]),
    ("importance", a_importance, [("importance", 4), ("most important", 4), ("rely", 2), ("permutation", 4)]),
    ("correlation", a_correlation, [("correlation", 4), ("correlat", 3), ("heatmap", 3), ("red bar", 3), ("green bar", 3),
                                    ("eda", 2), ("relationship", 2)]),
    ("versions", a_versions, [("upload", 3), ("replace", 3), ("version", 3), ("restore", 4), ("new data", 3),
                              ("update the data", 4), ("csv", 1)]),
    ("batch", a_batch, [("batch", 4), ("many laptops", 4), ("template", 3), ("bulk", 3)]),
    ("limitations", a_limitations, [("limitation", 4), ("limits", 3), ("trust", 2), ("weakness", 3), ("caveat", 3),
                                    ("reliable", 2)]),
    ("units", a_units, [("mah", 4), ("mwh", 4), ("unit", 3)]),
    ("mrgk", a_mrgk, [("who are you", 4), ("mr. gk", 3), ("mr gk", 3), ("assistant", 2), ("guide mode", 4), ("ai mode", 4)]),
    ("precision", a_precision, [("precision", 5), ("recall", 5), ("f1", 4), ("confusion", 4), ("classification", 3),
                                ("band accuracy", 5)]),
    ("skew", a_skew, [("skew", 5), ("kurtosis", 5), ("distribution shape", 4), ("symmetric", 3), ("tail", 2)]),
    ("vif", a_vif, [("vif", 5), ("multicollinear", 5), ("collinear", 4), ("overlap", 2), ("variance inflation", 5)]),
    ("outliers", a_outliers, [("outlier", 5), ("iqr", 4), ("z-score", 4), ("extreme value", 3)]),
    ("selection", a_selection, [("not used", 5), ("why excluded", 5), ("excluded variable", 5), ("justif", 4),
                                ("variable selection", 5), ("unused", 4), ("left out", 4), ("dropped", 3)]),
    ("variables", a_variables, [("variable", 4), ("column", 3), ("feature", 1.5), ("input", 1.5), ("attributes", 2),
                                ("fields", 3)]),
    ("rows", a_rows, [("rows", 4), ("records", 3), ("how much data", 4), ("dataset size", 4), ("how many laptops", 4)]),
    ("final", a_final, [("final model", 4), ("refit", 4), ("all rows", 3), ("retrain", 2)]),
    ("deploy", a_deploy, [("deploy", 4), ("host", 3), ("github", 2), ("streamlit cloud", 4), ("online", 2), ("share", 1)]),
]
MIN_SCORE = 3


def answer(question, state):
    q = question.lower()
    best, best_score = None, 0.0
    for tid, fn, phrases in TOPICS:
        score = sum(w for p, w in phrases if p in q)
        if score > best_score:
            best, best_score = (tid, fn), score
    attr = find_attribute(question)
    # An attribute named in a question outranks weak topic matches ("does temperature affect health?")
    if attr and (best is None or best_score < 4 or best[0] in ("variables", "bands", "errors")):
        if not (best and best[0] in ("fcc",) and attr == C.OPTIONAL_FEATURE):
            text, sections = a_attribute(attr, state)
            return text, sections, f"attribute:{attr}"
    if best and best_score >= MIN_SCORE:
        text, sections = best[1](state)
        return text, sections, best[0]
    return None, [], None
