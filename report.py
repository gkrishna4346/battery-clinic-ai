"""Downloadable prediction records: one-row CSV and a one-page PDF report."""
import io
from datetime import datetime

import pandas as pd

import config as C

_REPLACE = {"−": "-", "≈": "~", "→": "->", "–": "-", "—": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
            "…": "...", "•": "-", "ƒ": "f", "⭐": "*", "✓": "v"}


def _t(text):
    """PDF core fonts support Latin-1 only; replace other characters."""
    s = str(text)
    for a, b in _REPLACE.items():
        s = s.replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")


def record(ctx):
    """ctx: dict with row, inputs, pred, band, mae, q90, model, split, include, version, explain, whatif."""
    rec = {"Date and time": ctx["time"], "Data version": f"v{ctx['version']}", "Model": ctx["model"],
           "Split": ctx["split"], f"{C.OPTIONAL_FEATURE} used": "Yes" if ctx["include"] else "No"}
    for c in ctx["inputs"]:
        v = ctx["row"][c]
        rec[c] = ("Yes" if v >= 0.5 else "No") if C.FEATURES[c]["kind"] == "binary" else v
    rec.update({f"Predicted {C.TARGET} (%)": round(ctx["pred"], 2), "Health band": ctx["band"],
                "Likely range low (%)": round(max(ctx["pred"] - ctx["mae"], 0), 2),
                "Likely range high (%)": round(min(ctx["pred"] + ctx["mae"], 100), 2),
                "Typical error (± points)": round(ctx["mae"], 2),
                "90% of predictions within (± points)": round(ctx["q90"], 2),
                "Typical laptop prediction (%)": round(ctx["explain"]["base"], 2)})
    for d in ctx["explain"]["drivers"]:
        rec[f"Effect: {d['Input']} (points)"] = round(d["Effect"], 2)
    return rec


def to_csv(ctx):
    return pd.DataFrame([record(ctx)]).to_csv(index=False).encode("utf-8")


def pdf_available():
    try:
        import fpdf  # noqa: F401
        return True
    except ImportError:
        return False


def to_pdf(ctx):
    from fpdf import FPDF   # imported here so the app still runs if fpdf2 is not installed
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=14)
    pdf.add_page()
    W = pdf.w - 2 * pdf.l_margin
    navy, blue, grey = (7, 18, 37), (47, 123, 255), (90, 100, 120)
    band_rgb = {"Good": (30, 160, 110), "Moderate": (215, 140, 20), "Poor": (210, 70, 70)}[ctx["band"]]

    pdf.set_fill_color(*navy)
    pdf.rect(0, 0, pdf.w, 30, "F")
    pdf.set_xy(pdf.l_margin, 8)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(W, 8, _t(f"{C.APP_TITLE} - Battery health report"))
    pdf.set_xy(pdf.l_margin, 18)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(W, 6, _t(f"{ctx['time']}   |   Data version v{ctx['version']}   |   Model: {ctx['model']}   |   "
                      f"Split {ctx['split']}   |   {C.OPTIONAL_FEATURE}: {'included' if ctx['include'] else 'excluded'}"))

    pdf.set_xy(pdf.l_margin, 38)
    pdf.set_text_color(*band_rgb)
    pdf.set_font("Helvetica", "B", 30)
    pdf.cell(55, 14, _t(f"{ctx['pred']:.1f}%"))
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(40, 14, _t(ctx["band"]))
    pdf.set_text_color(*grey)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(W - 95, 5, _t(f"Likely between {max(ctx['pred'] - ctx['mae'], 0):.1f}% and "
                                 f"{min(ctx['pred'] + ctx['mae'], 100):.1f}% (typical error ±{ctx['mae']:.2f} points). "
                                 f"90% of test predictions were within ±{ctx['q90']:.2f} points."))
    pdf.ln(4)

    def heading(text):
        pdf.ln(2)
        pdf.set_text_color(*blue)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(W, 7, _t(text), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 210, 225)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(20, 20, 20)
        pdf.set_font("Helvetica", "", 10)

    heading("Inputs")
    for c in ctx["inputs"]:
        spec = C.FEATURES[c]
        v = ctx["row"][c]
        val = ("Yes" if v >= 0.5 else "No") if spec["kind"] == "binary" else f"{v:,.{spec['decimals']}f} {spec['unit']}"
        pdf.cell(70, 6, _t(c))
        pdf.cell(W - 70, 6, _t(val), new_x="LMARGIN", new_y="NEXT")

    heading("From a typical laptop to this laptop")
    ex = ctx["explain"]
    pdf.cell(W, 6, _t(f"Typical laptop in the data: {ex['base']:.1f}%"), new_x="LMARGIN", new_y="NEXT")
    for s in ex["steps"]:
        pdf.cell(W, 6, _t(f"   {s['Input']}: {s['Effect']:+.1f} points"), new_x="LMARGIN", new_y="NEXT")
    if abs(ex["other"]) >= 0.05:
        pdf.cell(W, 6, _t(f"   Other inputs and combined effects: {ex['other']:+.1f} points"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(W, 6, _t(f"This laptop: {ex['final']:.1f}%"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    if ctx["whatif"]:
        heading("What if")
        for w in ctx["whatif"]:
            spec = C.FEATURES[w["Input"]]
            fmtv = (lambda v: "Yes" if v >= 0.5 else "No") if spec["kind"] == "binary" else \
                   (lambda v: f"{v:,.{spec['decimals']}f} {spec['unit']}".strip())
            pdf.multi_cell(W, 5.5, _t(f"If {w['Input']} were typical ({fmtv(w['Typical'])} instead of "
                                      f"{fmtv(w['Value'])}), estimated health would be {w['New']:.1f}% "
                                      f"({w['Gain']:+.1f} points)."), new_x="LMARGIN", new_y="NEXT")

    heading("Conclusion")
    for line in ctx["conclusion"]:
        pdf.multi_cell(W, 5.5, _t(line), new_x="LMARGIN", new_y="NEXT")
    if ctx["advice"]:
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(W, 6, "What may help", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for a in ctx["advice"]:
            pdf.multi_cell(W, 5.5, _t(f"- {a}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)
    pdf.set_text_color(*grey)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(W, 4.5, _t("Estimates come from a model trained on the project's dataset. Insights describe patterns in "
                              "that data, not proven causes. Inputs outside the training range are less reliable."))
    return bytes(pdf.output())


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")
