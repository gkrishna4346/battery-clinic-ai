"""
Battery Clinic AI - Streamlit app (v2.0).

    pip install -r requirements.txt
    python -m streamlit run app.py
"""
import base64
import io
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

import config as C
import engine as E
import mrgk as G
import report as Rp
import versions as V

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda rel: os.path.join(HERE, rel)  # noqa: E731

st.set_page_config(page_title=C.APP_TITLE, page_icon="🔋", layout="wide", initial_sidebar_state="expanded")


def _secrets():
    try:
        return dict(st.secrets)
    except Exception:
        return {}


SECRETS = _secrets()

# ============================================================ style & theme
AVATAR_B64 = __import__("base64").b64encode(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                              "assets", "mrgk.png"), "rb").read()).decode()
LIGHT = st.session_state.get("theme", "dark") == "light"
# Every dark colour used by the app (and the guide) with its light-theme equivalent
THEME_MAP = {
    "#071225": "#f4f7fc", "#061022": "#e9eff8", "#0e1f3b": "#ffffff", "#132a4f": "#f1f5fc", "#1f3a66": "#d3deef",
    "#0a1830": "#eef3fb", "#0f2a55": "#dde9fb", "#0d2447": "#eef4ff", "#0b2f57": "#e3effd", "#10284f": "#eef4ff",
    "#0c2142": "#eaf2ff", "#23508f": "#a9c3ea", "#16305a": "#e3eaf5", "#17315a": "#e3eaf5", "#18325c": "#e3eaf5",
    "#12306b": "#d9e6ff", "#0f2242": "#e6eefb", "#0d2140": "#e4edfb", "#0a1a33": "#eef3fb", "#2a4f86": "#9bb4dc",
    "#0f3b2e": "#dff7ee", "#1d6b50": "#86d4b3", "#0d2a24": "#e3f7ef", "#2a2210": "#fff5e0", "#6b5320": "#e8c77a",
    "#e6eefc": "#1b2a44", "#c9d6ee": "#3b4d6b", "#c3d2ec": "#3b4d6b", "#cfe0ff": "#1d3c78", "#8fa5c9": "#5d6f8f",
    "#93a8cb": "#5d6f8f", "#29c5f6": "#0b8fc4", "#34d399": "#0f9f6e", "#f5b041": "#c9820a", "#f0605d": "#d64541",
    "#b99bff": "#7b5cd6", "#2f7bff": "#2563eb", "#5b77a6": "#8aa2c8", "#4d6fa8": "#8aa2c8", "#3b5b8f": "#9fb3d4",
}


def T(s):
    """Apply the light theme to a string of HTML/CSS (no change in dark mode)."""
    if not LIGHT or not isinstance(s, str):
        return s
    for dark, light in THEME_MAP.items():
        s = s.replace(dark, light).replace(dark.upper(), light)
    return s


COL = {"bg": "#071225", "panel": "#0e1f3b", "panel2": "#132a4f", "line": "#1f3a66", "text": "#e6eefc",
       "muted": "#8fa5c9", "blue": "#2f7bff", "cyan": "#29c5f6", "green": "#34d399", "amber": "#f5b041",
       "red": "#f0605d"}
COL = {k: T(v) for k, v in COL.items()}

st.markdown(T(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, input, textarea, button, label {{
  font-family: 'Manrope', 'Segoe UI', system-ui, sans-serif; }}
h1, h2, h3, h4, .bs-h {{ font-family: 'Sora', 'Manrope', system-ui, sans-serif !important; letter-spacing: -0.01em; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stAppViewContainer"] {{
  background: radial-gradient(1200px 500px at 70% -10%, #0f2a55 0%, {COL['bg']} 55%); }}
[data-testid="stSidebar"] {{ background: #061022; border-right: 1px solid {COL['line']}; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: .35rem .6rem; border-radius: 10px; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: #12306b; }}
[class*="st-key-nav_"] {{ margin-bottom: -.55rem; }}
[class*="st-key-nav_"] button {{ justify-content: flex-start !important; border: 0 !important; min-height: 2.7rem;
  padding: .55rem .9rem !important; border-radius: 10px !important; }}
[class*="st-key-nav_"] button > div {{ justify-content: flex-start !important; width: 100%; }}
[class*="st-key-nav_"] button p {{ font-size: 1rem !important; text-align: left; }}
[class*="st-key-nav_"] button[kind="secondary"], [class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"] {{
  background: transparent !important; color: {COL['text']} !important; }}
[class*="st-key-nav_"] button[kind="secondary"]:hover, [class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"]:hover {{
  background: #0f2242 !important; }}
.bs-panel {{ background: {COL['panel']}; border: 1px solid {COL['line']}; border-radius: 16px; padding: 1.1rem 1.25rem; }}
.bs-title {{ font-family: 'Sora', sans-serif; font-weight: 600; font-size: 1.12rem; color: {COL['text']}; margin: 0 0 .6rem 0; }}
.bs-muted {{ color: {COL['muted']}; font-size: .88rem; }}
.bs-pill {{ display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .8rem; font-weight: 600; }}
.bs-card {{ background: {COL['panel2']}; border: 1px solid {COL['line']}; border-radius: 12px; padding: .8rem 1rem; height: 100%; }}
.bs-card .k {{ color: {COL['muted']}; font-size: .82rem; }}
.bs-card .v {{ font-family: 'Sora', sans-serif; font-size: 1.55rem; font-weight: 600; margin-top: .15rem; }}
.bs-card .s {{ color: {COL['muted']}; font-size: .76rem; margin-top: .1rem; }}
.stTextInput input {{ background: #0a1830 !important; border-radius: 10px !important; }}
[data-testid="stForm"] div[data-testid="stCaptionContainer"] {{ margin-top: -0.55rem; }}
.stButton button[kind="primary"], button[data-testid="stBaseButton-primaryFormSubmit"],
button[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(90deg, #2563eb, #1d9bf0) !important; border: 0 !important; font-weight: 700 !important; }}
.bs-item {{ padding: .5rem .1rem; border-bottom: 1px solid #16305a; font-size: .9rem; line-height: 1.45; }}
.bs-item:last-child {{ border-bottom: 0; }}
.bs-group {{ font-weight: 700; margin: .7rem 0 .15rem 0; font-size: .92rem; }}
.bs-avatar {{ width: 58px; height: 58px; border-radius: 50%; }}
[data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"],
[data-testid="stToolbarActionButton"] {{ display: none !important; }}
.st-key-topbar {{ position: fixed; top: .55rem; right: 1rem; z-index: 999991; width: auto !important; }}
.st-key-topbar [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: .55rem !important; width: auto !important; }}
.st-key-topbar [data-testid="stColumn"], .st-key-topbar [data-testid="column"] {{
  width: auto !important; flex: 0 0 auto !important; min-width: 0 !important; }}
.st-key-theme_btn button, .st-key-guide_btn button {{
  width: 42px !important; height: 42px !important; min-height: 42px !important; padding: 0 !important;
  border-radius: 50% !important; border: 1px solid {COL['line']} !important; background: {COL['panel']} !important;
  display: flex; align-items: center; justify-content: center; }}
.st-key-theme_btn button span, .st-key-guide_btn button span {{ font-size: 1.35rem !important; color: {COL['cyan']} !important; margin: 0 !important; }}
.st-key-theme_btn button:hover, .st-key-guide_btn button:hover, .st-key-gk_btn button:hover {{ border-color: {COL['cyan']} !important; }}
.st-key-gk_btn button {{
  height: 42px !important; min-height: 42px !important; border-radius: 999px !important; padding: 0 1rem 0 .25rem !important;
  border: 1px solid {COL['line']} !important; background: {COL['panel']} !important; display: flex; align-items: center; gap: .5rem; }}
.st-key-gk_btn button::before {{ content: ""; width: 34px; height: 34px; flex: 0 0 34px; border-radius: 50%;
  background: url(data:image/png;base64,{AVATAR_B64}) center / cover no-repeat; }}
.st-key-gk_btn button p {{ font-weight: 700 !important; white-space: nowrap; }}
.st-key-gkfloat {{ position: fixed; right: 1.1rem; top: 4.1rem; bottom: 1rem; width: min(440px, 94vw) !important;
  z-index: 999990; background: {COL['panel']}; border: 1px solid {COL['line']}; border-radius: 18px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, .45); padding: .9rem 1rem .6rem; overflow-y: auto; }}
.st-key-gk_close button {{ width: 34px !important; height: 34px !important; min-height: 34px !important; padding: 0 !important;
  border-radius: 50% !important; border: 1.5px solid {COL['muted']} !important; background: transparent !important;
  display: flex; align-items: center; justify-content: center; }}
.st-key-gk_close button span {{ font-size: 1.1rem !important; margin: 0 !important; color: {COL['text']} !important; }}
.st-key-gk_close {{ display: flex; justify-content: flex-end; }}
.st-key-gkfaq [data-testid="stExpander"] details {{ border-radius: 999px; }}
.st-key-gkfaq [data-testid="stExpander"] details[open] {{ border-radius: 14px; }}
[class*="st-key-starter_"] button {{ min-height: 2rem; padding: .2rem .6rem !important; border-radius: 999px !important; }}
[class*="st-key-starter_"] button p {{ font-size: .8rem !important; }}
[class*="st-key-cite_"] button {{ min-height: 2rem; padding: .15rem .6rem !important; }}
[class*="st-key-cite_"] button p {{ font-size: .82rem !important; }}
</style>
"""), unsafe_allow_html=True)


LIGHT_CSS = """<style>
.stApp { color: #1b2a44; }
.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #1b2a44; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #5d6f8f !important; }
.stApp input, .stApp textarea { background: #ffffff !important; color: #1b2a44 !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] { background: #ffffff !important; border-color: #cdd9ec !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"], [role="listbox"] { background: #ffffff !important; color: #1b2a44 !important; }
[role="option"] { color: #1b2a44 !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button,
button[kind="secondaryFormSubmit"], button[data-testid="stBaseButton-secondaryFormSubmit"] {
  background: #ffffff !important; color: #1b2a44 !important; border: 1px solid #cdd9ec !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #d3deef !important; }
[data-testid="stExpander"] details, [data-testid="stExpander"] summary { background: #ffffff; border-color: #d3deef; color: #1b2a44; }
[data-testid="stFileUploaderDropzone"] { background: #ffffff !important; color: #1b2a44 !important; }
[data-testid="stChatMessage"] { background: #f6f9fe; }
[data-testid="stChatInput"] textarea, [data-testid="stChatInput"] > div { background: #ffffff !important; }
.stTabs [data-baseweb="tab"] p { color: #3b4d6b; }
.stTabs [aria-selected="true"] p { color: #1b2a44; font-weight: 700; }
[data-testid="stSidebar"] * { color: #1b2a44; }
[data-testid="stAlert"] p { color: #1b2a44; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] [class*="st-key-nav_"] button[kind="secondary"] { background: transparent !important; border: 0 !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-secondary"]:hover { background: #dde8f8 !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stTextInputRootElement"] {
  border: 1px solid #c3d1e8 !important; background: #ffffff !important; }
[data-testid="stSelectbox"] [role="group"], [data-testid="stSelectbox"] [data-baseweb="select"] > div {
  background-color: #ffffff !important; border: 1px solid #c3d1e8 !important; }
[data-testid="stSelectbox"] [role="group"] *, [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #1b2a44 !important; }
[data-testid="stSelectbox"] [role="group"] svg { fill: #3b4d6b !important; }
[data-testid="stRadioOption"]:not(:has(input:checked)) > div > div:first-child { background: #8fa4c6 !important; }
[data-testid="stRadioOption"]:not(:has(input:checked)) > div > div:first-child > div { background: #ffffff !important; }
label[data-baseweb="radio"]:not(:has(input:checked)) > div:first-child { background: #ffffff !important; border: 1.5px solid #8fa4c6 !important; }
[data-testid="stCheckbox"] label:not(:has(input:checked)) > div:first-of-type { background: #b8c7df !important; }
[data-testid="stCheckbox"] label > div:first-of-type > div { background: #ffffff !important; box-shadow: 0 1px 2px #0003; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipHoverTarget"] svg, [data-testid="stTooltipHoverTarget"] button,
[data-testid="stTooltipIcon"] button { color: #5d6f8f !important; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipIcon"] svg * { stroke: #5d6f8f !important; fill: none !important; }
[data-testid="stSidebarCollapseButton"] svg *, [data-testid="stExpandSidebarButton"] svg *,
[data-testid="stSidebarCollapsedControl"] svg * { stroke: #3b4d6b; }
[data-testid="stChatInput"] { border: 1px solid #c3d1e8 !important; background: #ffffff !important; border-radius: 12px !important; }
[data-testid="stChatInput"] * { background-color: #ffffff !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #7f90ad !important; opacity: 1; }
[data-testid="stChatInput"] button { background: #eef3fb !important; }
[data-testid="stChatInput"] button svg { color: #2563eb !important; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapseButton"] svg,
[data-testid="stExpandSidebarButton"], [data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapseButton"] span, [data-testid="stExpandSidebarButton"] span { color: #3b4d6b !important; }
.st-key-gkfloat { box-shadow: 0 18px 50px rgba(27, 42, 68, .18) !important; }
.bs-html-table { overflow: auto; border: 1px solid #d3deef; border-radius: 10px; background: #ffffff; margin-bottom: .6rem; }
.bs-html-table table { border-collapse: collapse; width: 100%; font-size: .85rem; }
.bs-html-table th { position: sticky; top: 0; background: #eef3fb; text-align: left; padding: .45rem .6rem; color: #1b2a44; }
.bs-html-table td { padding: .4rem .6rem; border-top: 1px solid #e6edf7; color: #1b2a44; vertical-align: top; }
</style>"""
if LIGHT:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


def _wide(fn):
    """Full-width keyword that works on both older and newer Streamlit versions."""
    import inspect
    try:
        return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}
    except (TypeError, ValueError):
        return {"use_container_width": True}


WIDE_BTN, WIDE_DL, WIDE_FORM, WIDE_CHART = (_wide(st.button), _wide(st.download_button),
                                            _wide(st.form_submit_button), _wide(st.plotly_chart))


def html(s):
    """st.markdown with HTML; strips indentation/blank lines so Markdown doesn't treat it as code."""
    st.markdown(T("\n".join(line.strip() for line in s.splitlines() if line.strip())), unsafe_allow_html=True)


def fmt(v, decimals=1):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:,.{decimals}f}"


def style_fig(fig, height=330, title=None):
    fig.update_layout(template="plotly_white" if LIGHT else "plotly_dark", height=height,
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Manrope, sans-serif", color=T("#c9d6ee"), size=12),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    # Only set a title when there is one: an empty title object makes newer Plotly print "undefined"
    fig.update_layout(title=dict(text=title, font=dict(family="Sora, sans-serif", size=15)) if title else dict(text=""))
    fig.update_xaxes(gridcolor=T("#18325c"), zerolinecolor=T("#1f3a66"))
    fig.update_yaxes(gridcolor=T("#18325c"), zerolinecolor=T("#1f3a66"))
    return fig


def embed_html(content, height):
    """Embed trusted HTML (our own guide). st.iframe on new Streamlit, components.html on older versions."""
    if hasattr(st, "iframe"):
        st.iframe(content, height=height)
    else:
        components.html(content, height=height, scrolling=True)


def chart(fig):
    kw = {"theme": None} if LIGHT else {}
    if LIGHT:
        fig.update_layout(font_color="#3b4d6b", title_font_color="#1b2a44")
        fig.update_xaxes(tickfont_color="#3b4d6b", title_font_color="#3b4d6b")
        fig.update_yaxes(tickfont_color="#3b4d6b", title_font_color="#3b4d6b")
        if not any(getattr(an, "font", None) and an.font.color for an in fig.layout.annotations):
            fig.update_annotations(font_color="#3b4d6b")
    st.plotly_chart(fig, config={"displayModeBar": False}, **kw, **WIDE_CHART)


def show_table(df, max_rows=14, **kw):
    """Table sized to its rows, plus room for the horizontal scroll bar, so the last row is never hidden.
    In light mode tables are drawn as HTML, because Streamlit's table component keeps its own dark theme."""
    rows = min(len(df), max_rows)
    height = 38 + 35 * max(rows, 1) + 24
    if LIGHT:
        def cell(v):
            if v != v:
                return "–"
            a_ = abs(v)
            return f"{v:,.0f}" if a_ >= 1000 or float(v).is_integer() else f"{v:,.2f}" if a_ >= 1 else f"{v:.3f}"
        body = df.to_html(index=not kw.get("hide_index", False), border=0, na_rep="–", float_format=cell)
        cap = "" if len(df) <= max_rows else "max-height:620px;"
        st.markdown(f'<div class="bs-html-table" style="{cap}">{body}</div>', unsafe_allow_html=True)
    else:
        st.dataframe(df, height=height, **kw)


def avatar_b64():
    with open(P(C.MRGK_AVATAR), "rb") as f:
        return base64.b64encode(f.read()).decode()


# ============================================================ versions, data & training
LOG = V.ensure()
ACTIVE = V.active_version(LOG)


@st.cache_resource
def store():
    return {}


@st.cache_data
def load(path, data_id):  # data_id changes whenever the file changes, which refreshes the cache
    return E.load_data(path)


if not os.path.exists(P(C.DATA_FILE)):
    st.error(f"Data file not found: `{C.DATA_FILE}`. Upload one on the Data page or see the User guide, section 3.")
    st.stop()
DATA_ID = E.file_fingerprint(P(C.DATA_FILE))   # internal only; never shown to users
try:
    DF, REPORT = load(P(C.DATA_FILE), DATA_ID)
except ValueError as e:
    st.error(str(e))
    st.stop()
STATS = E.column_stats(E.add_engineered(DF))

PAGES = ["Data", "About dataset", "EDA & insights", "Model performance", "Predict", "Batch prediction"]
ICONS = {"Data": "📥", "About dataset": "🗂️", "EDA & insights": "📊", "Model performance": "🎯",
         "Predict": "🔮", "Batch prediction": "📄"}
SETUP_PAGES = {"Predict", "Batch prediction", "Model performance"}
ss = st.session_state
ss.setdefault("page", "Data")
ss.setdefault("view", "app")
ss.setdefault("include_fcc", C.OPTIONAL_FEATURE_DEFAULT and REPORT["optional_available"])
ss.setdefault("split", C.DEFAULT_SPLIT)
ss.setdefault("model_choice", {})
ss.setdefault("chat", [])
ss.setdefault("upload_n", 0)
if not REPORT["optional_available"]:
    ss.include_fcc = False


def to_app():
    ss.view = "app"


def go_page(p):
    ss.page = p
    ss.view = "app"


def open_guide(with_gk=False):
    ss.view = "guide"


def toggle_gk():
    ss.gk_open = not ss.get("gk_open", False)


def close_gk():
    ss.gk_open = False


ss.setdefault("gk_open", False)
GK_MODE = False   # Mr. GK floats over any page; the menu sidebar always stays available

if not GK_MODE:
    with st.sidebar:
        html(f"""
        <div style="display:flex;align-items:center;gap:.6rem;margin:.2rem 0 1rem 0">
          <div style="font-size:2rem">🔋</div>
          <div><div class="bs-h" style="font-weight:700;font-size:1.15rem">{C.APP_TITLE}</div>
          <div class="bs-muted" style="font-size:.78rem">{C.APP_SUBTITLE}</div></div>
        </div>""")
        for i, p in enumerate(PAGES):
            active = ss.view == "app" and ss.page == p
            st.button(f"{ICONS[p]}  {p}", key=f"nav_{i}", type="primary" if active else "secondary",
                      on_click=go_page, args=(p,), **WIDE_BTN)

INCLUDE = bool(ss.include_fcc)
SPLIT = ss.split
TEST_SIZE = C.SPLIT_OPTIONS[SPLIT]
CFG = (DATA_ID, INCLUDE, SPLIT)
S = store()
def _data_of(key):
    return key[0] if isinstance(key[0], str) else key[0][0]


for k in [k for k in S if isinstance(k, tuple) and _data_of(k) != DATA_ID]:  # forget results for old data
    S.pop(k, None)

if CFG not in S:
    _slot = st.empty()
    with _slot.container(), st.status("Training models on 5 shuffles of the data…", expanded=True) as status:
        bar = st.progress(0.0, text="Testing calculated attributes")
        base_inputs = E.raw_inputs(INCLUDE)
        kept, sel_table, sel_info = E.select_engineered(DF, base_inputs, TEST_SIZE)
        features = base_inputs + kept
        result = E.stability_run(DF, features, TEST_SIZE, progress=lambda f, t: bar.progress(f, text=t))
        S[CFG] = {"features": features, "inputs": base_inputs, "kept": kept, "selection": sel_table,
                  "selection_info": sel_info, "result": result, "trained_at": datetime.now()}
        status.update(label="Models trained", state="complete", expanded=False)
    _slot.empty()

R = S[CFG]
FEATURES, INPUTS, RESULT, TABLE = R["features"], R["inputs"], R["result"], R["result"]["table"]
MODEL_NAME = ss.model_choice.get(CFG, RESULT["recommended"])
if MODEL_NAME not in TABLE.index:
    MODEL_NAME = RESULT["recommended"]
MODEL_NAMES = list(TABLE.index)
if (CFG, MODEL_NAME) not in S:
    S[(CFG, MODEL_NAME)] = E.fit_final(DF, FEATURES, MODEL_NAME)
MODEL = S[(CFG, MODEL_NAME)]
M = TABLE.loc[MODEL_NAME]


# ============================================================ model setup with Apply
def model_label(n):
    star = "⭐ " if n == RESULT["recommended"] else ""
    return f"{star}{n}  ·  R² {TABLE.loc[n, 'Mean R²']:.3f} ± {TABLE.loc[n, 'R² spread (±)']:.3f}"


def ensure_pending():
    ss.setdefault("p_fcc", INCLUDE)
    ss.setdefault("p_split", SPLIT)
    for k in ("p_model_side", "p_model_page"):
        if ss.get(k) not in MODEL_NAMES:
            ss[k] = MODEL_NAME


def sync_model(src, dst):
    ss[dst] = ss[src]


def pending_changes(model_key):
    setup = []
    if ss.get("p_fcc", INCLUDE) != INCLUDE:
        setup.append(f"{C.OPTIONAL_FEATURE}: {'include' if ss.p_fcc else 'exclude'}")
    if ss.get("p_split", SPLIT) != SPLIT:
        setup.append(f"split {ss.p_split}")
    model = ss.get(model_key, MODEL_NAME)
    return setup, (model if model != MODEL_NAME else None)


def apply_changes(model_key, cfg):
    setup, model = pending_changes(model_key)
    if setup:
        ss.include_fcc, ss.split = ss.get("p_fcc", ss.include_fcc), ss.get("p_split", ss.split)
        for k in ("p_model_side", "p_model_page"):
            ss[k] = None
        ss.flash = ("Settings applied: " + ", ".join(setup) + ". All models were retrained and the recommended "
                    "model for the new setup is now active.")
    elif model:
        ss.model_choice[cfg] = model
        ss.flash = f"Active model changed to {model}. Predictions and reports now use this model."


SHOW_SETUP = ss.view == "app" and ss.page in SETUP_PAGES
with st.sidebar:
    if SHOW_SETUP:
        ensure_pending()
        st.divider()
        st.markdown("**Model setup**")
        st.toggle(f"Include {C.OPTIONAL_FEATURE}", key="p_fcc", disabled=not REPORT["optional_available"],
                  help="Off by default. Battery Health ≈ Full Charge Capacity ÷ Design Capacity × 100, so including it "
                       "makes the model mostly repeat that ratio.")
        st.radio("Train / test split", list(C.SPLIT_OPTIONS), key="p_split",
                 help="Every model is scored on 5 shuffles with this ratio; scores are averaged.")
        st.selectbox("Model", MODEL_NAMES, key="p_model_side", format_func=model_label,
                     on_change=sync_model, args=("p_model_side", "p_model_page"),
                     help="⭐ = recommended (highest stability score).")
        setup_p, model_p = pending_changes("p_model_side")
        st.button("✓ Apply", type="primary", disabled=not (setup_p or model_p), on_click=apply_changes,
                  args=("p_model_side", CFG), **WIDE_BTN)
        if setup_p:
            st.caption("Pending: " + ", ".join(setup_p) + ". Applying retrains all models (about 20–40 s) and selects "
                       "the recommended model.")
        elif model_p:
            st.caption(f"Pending: switch to {model_p}.")
if not GK_MODE:
  with st.sidebar:
    html(f"""
    <div class="bs-panel" style="padding:.8rem 1rem;margin-top:.6rem">
      <div class="bs-muted" style="font-size:.78rem">Active model</div>
      <div style="font-weight:700">{MODEL_NAME}</div>
      <div class="bs-muted" style="font-size:.78rem">R² {M['Mean R²']:.3f} ± {M['R² spread (±)']:.3f} · {SPLIT} split ·
      {C.OPTIONAL_FEATURE} {'on' if INCLUDE else 'off'}</div>
      <div class="bs-muted" style="font-size:.78rem;margin-top:.45rem;padding-top:.4rem;border-top:1px solid #1f3a66">
      Data version v{ACTIVE['version']} · {len(DF):,} rows</div>
    </div>""")


# ============================================================ header
target_unit = f" ({C.TARGET_UNIT})" if C.TARGET_UNIT else ""
subtitle = (f"Predicting {C.TARGET.lower()}{target_unit} from {len(INPUTS)} device inputs with "
            f"{MODEL_NAME} (R² {M['Mean R²']:.2f}, typical error ±{M['MAE']:.1f} points)")
def toggle_theme():
    ss.theme = "dark" if ss.get("theme", "dark") == "light" else "light"


html(f"""
<div style="display:flex;align-items:center;gap:.9rem">
  <div style="font-size:2.6rem;filter:drop-shadow(0 0 12px #34d39988)">🔋</div>
  <div><div class="bs-h" style="font-size:2rem;font-weight:700;line-height:1.15">
    Battery<span style="color:{COL['cyan']}">Clinic</span> AI</div>
    <div class="bs-muted" style="font-size:.95rem">{subtitle}</div></div>
</div>""")
with st.container(key="topbar"):
    tb = st.columns(3)
    tb[0].button("", key="theme_btn", icon=":material/dark_mode:" if LIGHT else ":material/light_mode:",
                 on_click=toggle_theme, help="Switch to dark theme" if LIGHT else "Switch to light theme")
    tb[1].button("", key="guide_btn", icon=":material/menu_book:", on_click=open_guide, help="Open the user guide")
    tb[2].button(C.MRGK_NAME, key="gk_btn", on_click=toggle_gk,
                 help=f"Close {C.MRGK_NAME}" if ss.get("gk_open") else f"Ask {C.MRGK_NAME}, the assistant")
st.write("")
if ss.get("flash"):
    st.success("✓ " + ss.flash)
    st.toast(ss.flash, icon="✅")
    ss.flash = None


# ============================================================ shared widgets
def hero():
    img = P(C.HERO_IMAGE)
    if os.path.exists(img):
        b64 = base64.b64encode(open(img, "rb").read()).decode()
        art = (f'<div style="flex:1.1;min-height:170px;border-radius:12px;background:url(data:image/jpeg;base64,{b64}) '
               f'center/cover"></div>')
    else:
        art = f"""<svg viewBox="0 0 260 150" style="flex:1;max-width:300px;min-width:180px" aria-hidden="true">
          <defs><linearGradient id="g1" x1="0" x2="1"><stop offset="0" stop-color="{COL['green']}"/>
          <stop offset="1" stop-color="{COL['cyan']}"/></linearGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="4"/></filter></defs>
          <rect x="40" y="12" width="180" height="108" rx="8" fill="#0a1a33" stroke="#2a4f86" stroke-width="2"/>
          <path d="M20 126 h220 l-14 12 h-192 z" fill="#132a4f" stroke="#2a4f86" stroke-width="2"/>
          <rect x="98" y="44" width="58" height="36" rx="6" fill="none" stroke="url(#g1)" stroke-width="5" filter="url(#glow)"/>
          <rect x="98" y="44" width="58" height="36" rx="6" fill="none" stroke="url(#g1)" stroke-width="3.5"/>
          <rect x="157" y="55" width="6" height="14" rx="2" fill="{COL['cyan']}"/>
          <path d="M130 50 l-10 14 h8 l-4 12 l12 -16 h-8 z" fill="{COL['green']}"/>
        </svg>"""
    html(f"""
    <div class="bs-panel" style="display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap;
         background:linear-gradient(110deg,#0e1f3b 0%,#0d2447 60%,#0b2f57 100%)">
      <div style="flex:1.4;min-width:260px">
        <div class="bs-h" style="font-size:1.7rem;font-weight:700;line-height:1.15">Know your battery.<br>
          <span style="color:{COL['cyan']}">Use it smarter.</span></div>
        <div class="bs-muted" style="margin-top:.5rem;font-size:.95rem;max-width:520px">
          Enter your laptop's usage details to estimate battery health, see what is driving the estimate,
          and how your laptop compares with {len(DF):,} others.</div>
      </div>
      {art}
    </div>""")


def gauge_svg(value, colour):
    r, sweep = 88, 270
    circ = 2 * np.pi * r
    arc = circ * sweep / 360
    filled = arc * max(0.0, min(value, 100)) / 100
    c1, c2 = (COL["green"], COL["cyan"]) if colour == COL["green"] or colour == C.HEALTH_BANDS[0][2] else (colour, colour)
    return f"""
    <svg viewBox="0 0 220 220" width="210" height="210" role="img" aria-label="Battery health {value:.1f} percent">
      <defs><linearGradient id="gg" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>
        <filter id="gl"><feGaussianBlur stdDeviation="5"/></filter></defs>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="#0a1830" stroke-width="16" stroke-linecap="round"
        stroke-dasharray="{arc:.1f} {circ:.1f}" transform="rotate(135 110 110)"/>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round"
        stroke-dasharray="{filled:.1f} {circ:.1f}" transform="rotate(135 110 110)" filter="url(#gl)" opacity=".55"/>
      <circle cx="110" cy="110" r="{r}" fill="none" stroke="url(#gg)" stroke-width="14" stroke-linecap="round"
        stroke-dasharray="{filled:.1f} {circ:.1f}" transform="rotate(135 110 110)"/>
      <text x="110" y="112" text-anchor="middle" fill="{COL['text']}" font-family="Sora, sans-serif"
        font-size="44" font-weight="700">{value:.1f}<tspan font-size="22">%</tspan></text>
      <text x="110" y="140" text-anchor="middle" fill="{COL['muted']}" font-size="14">Battery health</text>
    </svg>"""


def scale_bar(value):
    marker = max(0.0, min(value, 100))
    bands = sorted(C.HEALTH_BANDS, key=lambda b: b[0])
    edges = [b[0] for b in bands] + [100]
    stops = []
    for i, (lo, _, colour) in enumerate(bands):
        stops.append(f"{colour} {lo}%, {colour} {edges[i + 1]}%")
    legend = "".join(
        f'<div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{c};'
        f'margin-right:.4rem"></span>{label}<div class="bs-muted" style="font-size:.78rem;margin-left:1.1rem">'
        f'{lo}% – {edges[i + 1]}%</div></div>' for i, (lo, label, c) in enumerate(bands))
    return f"""
    <div class="bs-panel">
      <div class="bs-title">Battery health scale</div>
      <div style="position:relative;margin:1.4rem .3rem .4rem .3rem">
        <div style="height:14px;border-radius:99px;background:linear-gradient(90deg,{', '.join(stops)});opacity:.9"></div>
        <div style="position:absolute;left:calc({marker}% - 8px);top:-14px;width:0;height:0;border-left:8px solid transparent;
          border-right:8px solid transparent;border-top:11px solid {COL['text']}"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:.78rem" class="bs-muted">
        <span>0%</span><span>20%</span><span>40%</span><span>60%</span><span>80%</span><span>100%</span></div>
      <div style="display:flex;justify-content:space-between;margin-top:.8rem;font-size:.88rem">{legend}</div>
    </div>"""


def metric_cards(items):
    cols = st.columns(len(items))
    for col, (k, v, sub, colour) in zip(cols, items):
        with col:
            html(f'<div class="bs-card"><div class="k">{k}</div><div class="v" style="color:{colour}">{v}</div>'
                 f'<div class="s">{sub}</div></div>')


def default_text(c):
    s = STATS[c]
    if C.FEATURES[c]["kind"] == "binary":
        return "Yes" if s["mode"] >= 0.5 else "No"
    return fmt(s["median"], C.FEATURES[c]["decimals"]).replace(",", "")


def reset_inputs():
    for c in C.FEATURES:
        if c in STATS:
            st.session_state[f"in_{c}"] = default_text(c)
    st.session_state.pop("submitted_row", None)


def parse_inputs():
    row, errors, warnings = {}, [], []
    for c in INPUTS:
        spec = C.FEATURES[c]
        raw = st.session_state.get(f"in_{c}", default_text(c))
        if spec["kind"] == "binary":
            row[c] = 1.0 if raw == "Yes" else 0.0
            continue
        try:
            v = float(str(raw).replace(",", "").strip())
        except ValueError:
            errors.append(f"**{c}**: “{raw}” is not a number.")
            continue
        if v < 0 or (spec["unit"] == "%" and v > 100):
            errors.append(f"**{c}**: {v:g} is not possible (allowed {'0–100' if spec['unit'] == '%' else '0 or more'}).")
            continue
        s = STATS[c]
        if v < s["min"] or v > s["max"]:
            warnings.append(f"**{c}** = {v:g} is outside the range seen in the data "
                            f"({fmt(s['min'], spec['decimals'])} – {fmt(s['max'], spec['decimals'])}). "
                            "Predictions outside this range are less reliable.")
        row[c] = v
    return row, errors, warnings



# ============================================================ PAGE: Batch
def page_batch():
    html(f"""<div class="bs-panel"><div class="bs-title">📄 Batch prediction</div>
      <div class="bs-muted" style="margin:-.35rem 0 .5rem 0">Predict many laptops at once from a CSV file</div>
      <div style="line-height:1.6">Predict battery health for many laptops at once.
      <ol style="margin:.3rem 0 0 1.1rem;padding:0"><li>Download the template and fill one row per laptop
      (the {C.TARGET} column is not needed).</li><li>Upload the file below.</li>
      <li>Review the results and download them.</li></ol></div></div>""")
    st.write("")
    example = {c: (round(STATS[c]["median"], 2) if C.FEATURES[c]["kind"] != "binary" else int(STATS[c]["mode"]))
               for c in INPUTS}
    template = pd.DataFrame([example], columns=INPUTS)
    a, b = st.columns([1, 2])
    a.download_button("⬇️ Download template", template.to_csv(index=False).encode(), "battery_template.csv",
                      "text/csv", **WIDE_DL)
    b.caption(f"Required columns ({len(INPUTS)}): {', '.join(INPUTS)}. Gaming User: 1 = yes, 0 = no "
              "(Yes/No also accepted). The example row holds typical values.")
    up = st.file_uploader("Upload a CSV", type=["csv"])
    if up is None:
        return
    try:
        data = E.align_columns(E.read_csv_any(up))
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    missing = [c for c in INPUTS if c not in data.columns]
    if missing:
        st.error(f"The file is missing these columns: {missing}. Download the template to see the expected format.")
        return
    X = data.copy()
    for c in INPUTS:
        X[c] = E._to_number(X[c])
    preds = E.predict_frame(MODEL, X[INPUTS], FEATURES)
    out = data.copy()
    out[f"Predicted {C.TARGET}"] = np.round(preds, 2)
    out["Health band"] = [E.health_band(p)[0] for p in preds]
    flags = []
    for _, r in X[INPUTS].iterrows():
        bad = [c for c in INPUTS if pd.notna(r[c]) and (r[c] < STATS[c]["min"] or r[c] > STATS[c]["max"])]
        blank = [c for c in INPUTS if pd.isna(r[c])]
        flags.append("; ".join(([f"outside range: {', '.join(bad)}"] if bad else []) +
                               ([f"blank (filled with median): {', '.join(blank)}"] if blank else [])))
    out["Notes"] = flags
    st.success(f"Predicted {len(out):,} rows with {MODEL_NAME}.")
    counts = out["Health band"].value_counts()
    items = [(f"{label} ({lo}%+)", f"{counts.get(label, 0):,}", "laptops", colour) for lo, label, colour in C.HEALTH_BANDS]
    if C.TARGET in data.columns:
        actual = E._to_number(data[C.TARGET])
        ok = actual.notna()
        if ok.any():
            items.append(("MAE vs actual", f"{np.mean(np.abs(actual[ok] - preds[ok])):.2f}",
                          "file includes the true value", COL["cyan"]))
    metric_cards(items)
    st.write("")
    show_table(out, max_rows=15, hide_index=True)
    st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode(), "battery_predictions.csv", "text/csv",
                       type="primary")


# ============================================================ PAGE: EDA & insights (all variables, tabbed)
def eda_data():
    key = (DATA_ID, INCLUDE, SPLIT, "eda")
    if key not in S:
        eng, inputs, calc, tgt = E.eda_columns(DF)
        allv = inputs + calc + [tgt]
        imp_key = (CFG, MODEL_NAME, "importance")
        if imp_key not in S:
            S[imp_key] = E.importance_for(RESULT, MODEL_NAME, DF, FEATURES, TEST_SIZE)
        S[key] = {"eng": eng, "inputs": inputs, "calc": calc, "tgt": tgt, "all": allv,
                  "uni": E.univariate(eng, allv), "out": E.outlier_table(eng, inputs + calc + [tgt]),
                  "dec": E.variable_decisions(DF, INCLUDE, R["kept"], R["selection"], S[imp_key]),
                  "vif": E.vif_table(eng, [c for c in inputs if c in INPUTS])}
    return S[key]


def _role(c):
    return E.role_of(c, INCLUDE, R["kept"])


def _grid(n, cols=4):
    return int(np.ceil(n / cols)), cols


def page_eda():
    from plotly.subplots import make_subplots
    d = eda_data()
    eng, inputs, calc, tgt, allv, uni = d["eng"], d["inputs"], d["calc"], d["tgt"], d["all"], d["uni"]
    head_l, head_r = st.columns([3.2, 1])
    with head_l:
        html(f"""<div class="bs-panel"><div class="bs-title">📊 Exploratory data analysis</div>
          <div style="line-height:1.6">All <b>{len(allv)} variables</b>: {len(inputs)} inputs, {len(calc)} calculated attributes
          and the target, <b>{tgt}</b>. Every tab covers every variable, including those the model does not use; the
          <b>Variable selection</b> tab explains, with evidence, why each one is used or not.</div></div>""")
    with head_r:
        st.write("")
        st.download_button("⬇️ Download EDA report (HTML)", eda_report_html(), "BatteryClinic_EDA_report.html",
                           "text/html", **WIDE_DL)
    st.write("")
    tabs = st.tabs(["📋 Overview", "📈 Univariate", "🔗 Bivariate", "🕸️ Multivariate", "🚩 Outliers", "🎯 Target",
                    "✅ Variable selection"])

    # ---------------------------------------------------------------- Overview
    with tabs[0]:
        missing = int(eng[inputs + [tgt]].isna().sum().sum())
        metric_cards([("Rows used", f"{len(DF):,}", f"{REPORT['rows_raw']:,} read from the file", COL["cyan"]),
                      ("Variables", f"{len(allv)}", f"{len(inputs)} inputs · {len(calc)} calculated · 1 target", COL["green"]),
                      ("Missing values", f"{missing:,}", "filled with the training median", COL["amber"]),
                      ("Rows removed", f"{REPORT['duplicates_removed'] + REPORT['missing_target_removed']:,}",
                       f"{REPORT['duplicates_removed']} duplicates · {REPORT['missing_target_removed']} without target",
                       "#b99bff")])
        st.markdown("##### Variables")
        rows = []
        for c in allv:
            spec = C.FEATURES.get(c, {})
            rows.append({"Variable": c, "Role": _role(c),
                         "Type": ("Yes/No" if spec.get("kind") == "binary" else "Calculated number" if c in calc else "Number"),
                         "Unit": (spec.get("unit") or ("1 = yes, 0 = no" if spec else C.ENGINEERED_UNITS.get(c, C.TARGET_UNIT))),
                         "Non-missing": int(eng[c].notna().sum()), "Missing": int(eng[c].isna().sum()),
                         "Unique values": int(eng[c].nunique()),
                         "Meaning": spec.get("desc") or (C.ENGINEERED[c][2] + f" ({C.ENGINEERED[c][0]})" if c in calc
                                                         else "Current capacity as % of design capacity.")})
        show_table(pd.DataFrame(rows), max_rows=len(rows), hide_index=True)
        st.markdown("##### Summary statistics")
        show_table(uni[["Variable", "Mean", "Median", "Std", "Min", "Q1", "Q3", "Max"]].round(2), max_rows=len(uni),
                   hide_index=True)
        st.markdown("##### First rows of the data")
        show_table(DF.head(8), hide_index=True)

    # ---------------------------------------------------------------- Univariate
    with tabs[1]:
        st.markdown("##### Shape of every variable")
        show_table(uni[["Variable", "Mean", "Median", "Std", "Skewness", "Kurtosis", "Shape", "Tails"]].round(2),
                   max_rows=len(uni), hide_index=True)
        with st.expander("How to read skewness and kurtosis"):
            st.markdown("""
- **Skewness** measures asymmetry. Between −0.5 and 0.5: roughly symmetric. 0.5 to 1: moderately skewed. Above 1: highly skewed.
  Positive = a long tail of high values (most laptops low, a few very high); negative = a long tail of low values.
- **Kurtosis** (excess) measures how heavy the tails are compared with a normal distribution (0). Above 1: more extreme
  values than normal. Below −1: flat, values spread evenly.
- Formulas: skewness = mean((x − mean)³) ÷ std³; kurtosis = mean((x − mean)⁴) ÷ std⁴ − 3 (with small-sample corrections).
- Yes/No variables have no meaningful skewness or kurtosis.""")
        choice = st.selectbox("Look at one variable in detail", allv, index=allv.index("Cycle Count") if "Cycle Count" in allv else 0,
                              key="uni_var")
        r = uni.set_index("Variable").loc[choice]
        a, b = st.columns([1.3, 1], gap="medium")
        with a:
            fig = go.Figure(go.Histogram(x=eng[choice], nbinsx=35, marker_color=COL["cyan"]))
            fig.add_vline(x=r["Mean"], line_color=COL["amber"], line_dash="dash", annotation_text="mean",
                          annotation_font_color=COL["amber"])
            fig.add_vline(x=r["Median"], line_color=COL["green"], line_dash="dot", annotation_text="median",
                          annotation_position="bottom right", annotation_font_color=COL["green"])
            chart(style_fig(fig, height=320, title=f"Distribution of {choice}"))
        with b:
            fig = go.Figure(go.Box(x=eng[choice], orientation="h", marker_color=COL["blue"], boxpoints="outliers", name=""))
            chart(style_fig(fig, height=170, title="Box plot"))
            sk = "–" if pd.isna(r["Skewness"]) else f"{r['Skewness']:+.2f}"
            ku = "–" if pd.isna(r["Kurtosis"]) else f"{r['Kurtosis']:+.2f}"
            metric_cards([("Skewness", sk, r["Shape"], COL["cyan"]), ("Kurtosis", ku, r["Tails"], COL["green"])])
        st.markdown("##### All distributions")
        nr, nc = _grid(len(allv))
        fig = make_subplots(rows=nr, cols=nc, subplot_titles=allv, vertical_spacing=0.09, horizontal_spacing=0.05)
        for i, c in enumerate(allv):
            fig.add_trace(go.Histogram(x=eng[c], nbinsx=25, marker_color=COL["green"] if c == tgt else
                                       (COL["amber"] if c in calc else COL["cyan"]), showlegend=False), row=i // nc + 1, col=i % nc + 1)
        fig.update_annotations(font_size=11)
        chart(style_fig(fig, height=210 * nr))
        st.caption("Blue = inputs · amber = calculated attributes · green = target.")

    # ---------------------------------------------------------------- Bivariate
    with tabs[2]:
        corr = correlations()
        st.markdown(f"##### Which variables move with {tgt.lower()}?")
        a, b = st.columns([1.45, 1], gap="medium")
        with a:
            order = corr.reindex(corr.abs().sort_values().index)
            labels = [f"{c}  ƒ" if c in calc else c for c in order.index]
            fig = go.Figure(go.Bar(x=order.values, y=labels, orientation="h", cliponaxis=False,
                                   marker_color=[COL["red"] if v < 0 else COL["green"] for v in order.values],
                                   text=[f"{v:+.2f}" for v in order.values], textposition="outside",
                                   hovertemplate="%{y}: r = %{x:.2f}<extra></extra>"))
            for x0, x1, op in ((-0.7, -0.3, 0.035), (0.3, 0.7, 0.035), (-1.05, -0.7, 0.07), (0.7, 1.05, 0.07)):
                fig.add_vrect(x0=x0, x1=x1, fillcolor="#7fa8ff", opacity=op, line_width=0, layer="below")
            for x, t in ((0, "weak"), (-0.5, "moderate"), (0.5, "moderate"), (-0.87, "strong"), (0.87, "strong")):
                fig.add_annotation(x=x, y=1.03, yref="paper", text=t, showarrow=False, font=dict(size=11, color=COL["muted"]))
            fig.update_xaxes(range=[-1.05, 1.05], title=f"Correlation with {tgt} (−1 to +1)", zeroline=True,
                             zerolinecolor="#5b77a6", tickvals=[-1, -0.7, -0.3, 0, 0.3, 0.7, 1])
            chart(style_fig(fig, height=34 * len(order) + 110))
        with b:
            strong_neg = [(k, v) for k, v in corr.sort_values().items() if v <= -0.7]
            mod_neg = [(k, v) for k, v in corr.sort_values().items() if -0.7 < v <= -0.3]
            pos = [(k, v) for k, v in corr.sort_values(ascending=False).items() if v >= 0.3]
            weak = [k for k, v in corr.items() if abs(v) < 0.3]
            fmtl = lambda items: " and ".join(f"<b>{k}</b> ({v:+.2f})" for k, v in items)  # noqa: E731
            points = []
            if strong_neg:
                points.append(f"{fmtl(strong_neg)} {'are' if len(strong_neg) > 1 else 'is'} <b>strongly</b> linked with lower "
                              "health: the higher the value, the lower the battery health tends to be.")
            if mod_neg:
                points.append(f"{fmtl(mod_neg[:3])} {'show' if len(mod_neg) > 1 else 'shows'} a <b>moderate</b> link with lower health.")
            for k, v in pos[:2]:
                extra = (f" Expected: health is essentially {k} ÷ Design Capacity, which is why it is off by default."
                         if k == C.OPTIONAL_FEATURE else "")
                points.append(f"<b>{k}</b> ({v:+.2f}) goes with <b>higher</b> health.{extra}")
            if weak:
                points.append(f"{len(weak)} variables have only <b>weak</b> links on their own (|r| &lt; 0.3), e.g. "
                              f"{', '.join(weak[:3])}. They can still matter in combination.")
            items = "".join(f"<li style='margin:.35rem 0'>{p}</li>" for p in points)
            html(f"""<div class="bs-panel">
              <div class="bs-title">How to read this chart</div>
              <div style="line-height:1.55;font-size:.92rem">Each bar shows how strongly a variable moves together with
              {tgt.lower()}, from −1 to +1 (Pearson correlation).<br>
              <span style="color:{COL['red']}">■</span> <b>Red</b>: higher values go with <b>lower</b> health.
              <span style="color:{COL['green']}">■</span> <b>Green</b>: higher values go with <b>higher</b> health.<br>
              Longer bar = stronger link (shaded: weak below 0.3, moderate 0.3–0.7, strong above 0.7). ƒ = calculated.</div>
              <div class="bs-title" style="margin-top:.9rem">What this tells you</div>
              <ul style="margin:.2rem 0 0 1.1rem;padding:0;line-height:1.5;font-size:.92rem">{items}</ul>
              <div class="bs-muted" style="margin-top:.6rem;font-size:.8rem">Correlation shows association, not cause, and only
              captures straight-line relationships.</div></div>""")
        st.markdown("##### One variable against battery health")
        opts = inputs + calc
        choice = st.selectbox("Variable", opts, index=opts.index("Cycle Count") if "Cycle Count" in opts else 0, key="bi_var")
        pearson = eng[[choice, tgt]].corr().iloc[0, 1]
        spearman = eng[[choice, tgt]].corr(method="spearman").iloc[0, 1]
        a, b = st.columns(2, gap="medium")
        with a:
            if eng[choice].nunique() <= 2:
                fig = go.Figure()
                for v, name in ((0, "No"), (1, "Yes")):
                    fig.add_trace(go.Box(y=eng.loc[eng[choice] == v, tgt], name=name, marker_color=COL["cyan"] if v else COL["blue"]))
                fig.update_layout(showlegend=False)
                fig.update_yaxes(title=tgt)
                chart(style_fig(fig, title=f"{tgt} for {choice}: Yes vs No"))
            else:
                samp = eng.sample(min(2500, len(eng)), random_state=0)
                fig = go.Figure(go.Scattergl(x=samp[choice], y=samp[tgt], mode="markers",
                                             marker=dict(size=5, color=COL["blue"], opacity=.4), name="laptops"))
                ok = samp[[choice, tgt]].dropna()
                if len(ok) > 2 and ok[choice].nunique() > 1:
                    k_, c0 = np.polyfit(ok[choice], ok[tgt], 1)
                    xs = np.linspace(ok[choice].min(), ok[choice].max(), 50)
                    fig.add_trace(go.Scatter(x=xs, y=k_ * xs + c0, mode="lines", line=dict(color=COL["amber"], width=3),
                                             name="linear trend"))
                fig.update_xaxes(title=choice)
                fig.update_yaxes(title=tgt)
                chart(style_fig(fig, title=f"{choice} vs {tgt}"))
        with b:
            g = E.binned_target(eng, choice)
            fig = go.Figure(go.Bar(x=g["label"], y=g["Mean health"], marker_color=COL["green"],
                                   text=[f"{v:.1f}" for v in g["Mean health"]], textposition="outside", cliponaxis=False,
                                   customdata=g["Laptops"], hovertemplate="%{x}: %{y:.1f}% (%{customdata} laptops)<extra></extra>"))
            fig.update_yaxes(title=f"Average {tgt} (%)", range=[min(g["Mean health"]) - 8, max(g["Mean health"]) + 5])
            fig.update_xaxes(title=f"{choice} (groups of similar size)")
            chart(style_fig(fig, title=f"Average {tgt.lower()} across {choice}"))
        metric_cards([("Pearson r", f"{pearson:+.2f}", "straight-line link", COL["cyan"]),
                      ("Spearman ρ", f"{spearman:+.2f}", "rank link (also catches curves)", COL["green"]),
                      ("Strength", ("strong" if abs(pearson) >= .7 else "moderate" if abs(pearson) >= .3 else "weak"
                                    if abs(pearson) >= .1 else "near zero"), _role(choice), COL["amber"])])
        st.markdown(f"##### Average {tgt.lower()} across the range of every variable")
        nr, nc = _grid(len(opts))
        fig = make_subplots(rows=nr, cols=nc, subplot_titles=[f"{c} ƒ" if c in calc else c for c in opts],
                            vertical_spacing=0.1, horizontal_spacing=0.05)
        for i, c in enumerate(opts):
            g = E.binned_target(eng, c)
            fig.add_trace(go.Scatter(x=g["mid"], y=g["Mean health"], mode="lines+markers", showlegend=False,
                                     line=dict(color=COL["red"] if corr.get(c, 0) < 0 else COL["green"], width=2)),
                          row=i // nc + 1, col=i % nc + 1)
        fig.update_annotations(font_size=11)
        chart(style_fig(fig, height=200 * nr))
        st.caption("Each line shows the average battery health for groups of laptops ordered by that variable. A falling line "
                   "means higher values go with lower health; a flat line means little relationship.")

    # ---------------------------------------------------------------- Multivariate
    with tabs[3]:
        with_calc = st.checkbox("Include calculated attributes in the matrix", value=False, key="mv_calc")
        cols_m = inputs + (calc if with_calc else []) + [tgt]
        cm = eng[cols_m].corr()
        fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale="RdBu",
                                   text=np.round(cm.values, 2), texttemplate="%{text}", textfont=dict(size=9 if with_calc else 10)))
        chart(style_fig(fig, height=560 if with_calc else 500, title="Correlation matrix"))
        a, b = st.columns([1, 1], gap="medium")
        with a:
            st.markdown("##### Pairs of inputs that move together")
            pairs = E.high_pairs(eng, inputs + (calc if with_calc else []))
            if len(pairs):
                show_table(pairs.round(2), max_rows=12, hide_index=True)
            else:
                st.caption("No pairs with |r| ≥ 0.5.")
            st.caption("Pairs with |r| ≥ 0.5. Strongly related inputs share their information, so the model cannot fully "
                       "separate their individual effects.")
        with b:
            st.markdown("##### Multicollinearity (VIF)")
            vif = d["vif"]
            fig = go.Figure(go.Bar(x=vif["VIF"], y=vif["Variable"], orientation="h",
                                   marker_color=[COL["red"] if v >= 10 else COL["amber"] if v >= 5 else COL["green"] for v in vif["VIF"]],
                                   text=[f"{v:.1f}" for v in vif["VIF"]], textposition="outside", cliponaxis=False))
            fig.add_vline(x=5, line_dash="dot", line_color=COL["amber"], annotation_text="5", annotation_font_color=COL["amber"])
            fig.add_vline(x=10, line_dash="dot", line_color=COL["red"], annotation_text="10", annotation_font_color=COL["red"])
            fig.update_yaxes(autorange="reversed")
            chart(style_fig(fig, height=40 * len(vif) + 80))
            st.caption("VIF = 1 ÷ (1 − R²), where R² is how well the other model inputs predict this one. Below 5: fine; "
                       "5–10: moderate overlap; above 10: high. Calculated over the inputs the model currently uses.")
        st.markdown("##### Scatter matrix of the strongest variables")
        top = list(correlations().abs().sort_values(ascending=False).index[:4])
        samp = eng.sample(min(1500, len(eng)), random_state=1)
        bands = E.band_labels(samp[tgt])
        colour_map = {b[1]: b[2] for b in C.HEALTH_BANDS}
        fig = go.Figure(go.Splom(dimensions=[dict(label=c, values=samp[c]) for c in top + [tgt]],
                                 marker=dict(size=3, color=[colour_map[b] for b in bands], opacity=.55),
                                 diagonal_visible=False, showupperhalf=False, text=bands))
        chart(style_fig(fig, height=640))
        st.caption(f"The four variables most correlated with {tgt.lower()}, plotted against each other and the target. "
                   "Colours show the health band (green Good, amber Moderate, red Poor).")

    # ---------------------------------------------------------------- Outliers
    with tabs[4]:
        html(f"""<div class="bs-panel" style="line-height:1.6"><b>Two rules are used.</b> The <b>IQR rule</b> flags values below
          Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR (IQR = Q3 − Q1). The <b>z-score rule</b> flags values more than 3 standard
          deviations from the mean. Outliers are <b>kept</b>: they are genuine laptops, tree models are robust to them, scaling
          limits their effect on linear models, and the Predict page warns when an input is outside the data's range.</div>""")
        st.write("")
        out = d["out"]
        show_table(out.round(2), max_rows=len(out), hide_index=True)
        worst = out.sort_values("IQR outliers %", ascending=False).head(3)
        st.caption("Most outliers: " + ", ".join(f"{r['Variable']} ({r['IQR outliers %']:.1f}%)" for _, r in worst.iterrows()) + ".")
        numeric = [c for c in allv if eng[c].nunique() > 2]
        nr, nc = _grid(len(numeric))
        fig = make_subplots(rows=nr, cols=nc, subplot_titles=numeric, vertical_spacing=0.1, horizontal_spacing=0.05)
        for i, c in enumerate(numeric):
            fig.add_trace(go.Box(x=eng[c], orientation="h", boxpoints="outliers", showlegend=False, name="",
                                 marker=dict(color=COL["amber"] if c in calc else COL["cyan"], size=3)),
                          row=i // nc + 1, col=i % nc + 1)
        fig.update_yaxes(showticklabels=False)
        fig.update_annotations(font_size=11)
        chart(style_fig(fig, height=150 * nr))

    # ---------------------------------------------------------------- Target
    with tabs[5]:
        r = uni.set_index("Variable").loc[tgt]
        metric_cards([("Mean", f"{r['Mean']:.1f}%", f"median {r['Median']:.1f}%", COL["cyan"]),
                      ("Std", f"{r['Std']:.2f}", "points", COL["green"]),
                      ("Skewness", f"{r['Skewness']:+.2f}", r["Shape"], COL["amber"]),
                      ("Kurtosis", f"{r['Kurtosis']:+.2f}", r["Tails"], "#b99bff")])
        a, b = st.columns([1.4, 1], gap="medium")
        with a:
            fig = go.Figure(go.Histogram(x=eng[tgt], nbinsx=40, marker_color=COL["blue"]))
            for lo, label, colour in C.HEALTH_BANDS:
                fig.add_vline(x=lo, line_dash="dot", line_color=colour, annotation_text=label, annotation_font_color=colour)
            chart(style_fig(fig, title=f"Distribution of {tgt}"))
        with b:
            share = E.band_share(eng[tgt])
            fig = go.Figure(go.Bar(x=share["Band"], y=share["Share %"], text=[f"{v:.1f}%<br>({n:,})" for v, n in
                                                                              zip(share["Share %"], share["Laptops"])],
                                   textposition="outside", cliponaxis=False,
                                   marker_color=[{b[1]: b[2] for b in C.HEALTH_BANDS}[x] for x in share["Band"]]))
            fig.update_yaxes(title="% of laptops", range=[0, max(share["Share %"]) * 1.25 + 1])
            chart(style_fig(fig, title="Laptops per health band"))
        empty = share[share["Laptops"] == 0]["Band"].tolist()
        if empty:
            st.warning(f"No laptops in the {', '.join(empty)} band. The model has never seen such batteries, so predictions "
                       "in that range are extrapolations, and band precision/recall cannot be measured for it.")
        q = eng[tgt].quantile([.05, .25, .5, .75, .95])
        show_table(pd.DataFrame({"Percentile": ["5th", "25th", "50th (median)", "75th", "95th"],
                                 f"{tgt} (%)": q.values.round(2)}), hide_index=True)

    # ---------------------------------------------------------------- Variable selection
    with tabs[6]:
        dec = d["dec"]
        used = dec[dec["In model"] == "Yes"]["Variable"].tolist()
        notused = dec[(dec["In model"] == "No") & (dec["Variable"] != tgt)]["Variable"].tolist()
        html(f"""<div class="bs-panel" style="line-height:1.6"><div class="bs-title">Why each variable is used or not</div>
          The model uses <b>{len(used)}</b> variables: {', '.join(used)}.<br>
          <b>{len(notused)}</b> are not used: {', '.join(notused)}. The table gives the evidence for every decision:
          correlation with {tgt.lower()}, overlap with other inputs (VIF), outliers, the active model's importance, and the
          result of the selection test for calculated attributes.</div>""")
        st.write("")
        show_table(dec.round(3), max_rows=len(dec), hide_index=True)
        a, b = st.columns(2, gap="medium")
        with a:
            if REPORT["optional_available"]:
                ratio = DF[C.OPTIONAL_FEATURE] / DF["Design Capacity"] * 100
                rr = np.corrcoef(ratio, DF[tgt])[0, 1]
                idx = DF.sample(min(2500, len(DF)), random_state=1).index
                fig = go.Figure(go.Scattergl(x=ratio[idx], y=DF.loc[idx, tgt], mode="markers",
                                             marker=dict(size=4, color=COL["green"], opacity=.5)))
                fig.update_xaxes(title=f"{C.OPTIONAL_FEATURE} ÷ Design Capacity × 100")
                fig.update_yaxes(title=tgt)
                chart(style_fig(fig, title=f"Why {C.OPTIONAL_FEATURE} is excluded (r = {rr:.3f})"))
                st.caption("The capacity ratio almost equals battery health, so including it would make the model copy a ratio "
                           "(target leakage) instead of learning from usage.")
        with b:
            sel = R["selection"]
            if len(sel):
                fig = go.Figure(go.Bar(x=sel["Gain"], y=sel["Attribute"], orientation="h", cliponaxis=False,
                                       marker_color=[COL["green"] if k else COL["muted"] for k in sel["Kept"]],
                                       text=[f"{g:+.4f}" for g in sel["Gain"]], textposition="outside"))
                fig.add_vline(x=C.FEATURE_GAIN_THRESHOLD, line_dash="dot", line_color=COL["amber"],
                              annotation_text=f"keep threshold {C.FEATURE_GAIN_THRESHOLD}", annotation_font_color=COL["amber"])
                fig.update_xaxes(title="Change in cross-validated R² when added")
                chart(style_fig(fig, title="Calculated attributes: selection test"))
                st.caption("Each calculated attribute is added to the inputs and scored with 5-fold cross-validation on training "
                           "rows only. Bars below the threshold mean the attribute adds no real accuracy.")
        imp = S[(CFG, MODEL_NAME, "importance")].iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.Importance, y=imp.Attribute, orientation="h", marker_color=COL["blue"],
                               error_x=dict(array=imp.Std)))
        fig.update_xaxes(title="Drop in R² when the variable is shuffled")
        chart(style_fig(fig, height=40 * len(imp) + 90, title=f"How much the active model ({MODEL_NAME}) relies on each input"))


def eda_report_html():
    d = eda_data()
    eng, tgt = d["eng"], d["tgt"]
    corr = correlations()
    fig1 = go.Figure(go.Bar(x=corr.sort_values().values, y=corr.sort_values().index, orientation="h",
                            marker_color=["#d64541" if v < 0 else "#0f9f6e" for v in corr.sort_values().values]))
    fig1.update_layout(template="plotly_white", height=520, title=f"Correlation with {tgt}", margin=dict(l=10, r=10, t=50, b=10))
    cm = eng[d["inputs"] + [tgt]].corr()
    fig2 = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale="RdBu",
                                text=np.round(cm.values, 2), texttemplate="%{text}"))
    fig2.update_layout(template="plotly_white", height=560, title="Correlation matrix", margin=dict(l=10, r=10, t=50, b=10))
    fig3 = go.Figure(go.Histogram(x=eng[tgt], nbinsx=40, marker_color="#2563eb"))
    fig3.update_layout(template="plotly_white", height=360, title=f"Distribution of {tgt}", margin=dict(l=10, r=10, t=50, b=10))
    t = lambda df: df.to_html(index=False, border=0, classes="t", float_format=lambda v: f"{v:,.3f}")  # noqa: E731
    share = E.band_share(eng[tgt])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>{C.APP_TITLE} — EDA report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:system-ui,Segoe UI,sans-serif;color:#1b2a44;margin:0;background:#f4f7fc}}
main{{max-width:1150px;margin:auto;padding:2rem}} h1{{margin:0}} h2{{border-bottom:2px solid #d3deef;padding-bottom:.3rem;margin-top:2.2rem}}
.meta{{color:#5d6f8f}} .wrap{{overflow-x:auto;background:#fff;border:1px solid #d3deef;border-radius:10px;margin:.8rem 0}}
table.t{{border-collapse:collapse;width:100%;font-size:.85rem}} table.t th{{background:#eef3fb;text-align:left;padding:.45rem .6rem}}
table.t td{{padding:.4rem .6rem;border-top:1px solid #e6edf7;vertical-align:top}} .card{{background:#fff;border:1px solid #d3deef;border-radius:10px;padding:1rem;margin:.8rem 0}}</style>
</head><body><main>
<h1>{C.APP_TITLE} — Exploratory data analysis</h1>
<p class="meta">Data version v{ACTIVE['version']} · {len(DF):,} rows · {len(d['all'])} variables · generated {datetime.now():%d %b %Y %H:%M}</p>
<div class="card">Rows read {REPORT['rows_raw']:,}; removed {REPORT['duplicates_removed']} duplicates and {REPORT['missing_target_removed']}
rows without {tgt}. Model setup at export: {SPLIT} split, {C.OPTIONAL_FEATURE} {'included' if INCLUDE else 'excluded'}, active model {MODEL_NAME}.</div>
<h2>1. Variable selection — why each variable is used or not</h2><div class="wrap">{t(d['dec'])}</div>
<h2>2. Univariate — summary, skewness and kurtosis</h2><div class="wrap">{t(d['uni'])}</div>
<h2>3. Bivariate — correlation with {tgt}</h2>{fig1.to_html(full_html=False, include_plotlyjs='cdn')}
<h2>4. Multivariate — correlations and multicollinearity</h2>{fig2.to_html(full_html=False, include_plotlyjs=False)}
<div class="wrap">{t(E.high_pairs(eng, d['inputs']))}</div><div class="wrap">{t(d['vif'])}</div>
<h2>5. Outliers</h2><div class="wrap">{t(d['out'])}</div>
<h2>6. Target</h2>{fig3.to_html(full_html=False, include_plotlyjs=False)}<div class="wrap">{t(share)}</div>
<p class="meta">Correlation shows association, not cause. Outliers are kept as genuine laptops.</p>
</main></body></html>""".encode("utf-8")



# ============================================================ PAGE: Predict
def page_predict():
    hero()
    st.write("")
    for c in C.FEATURES:
        if c in STATS:
            ss.setdefault(f"in_{c}", default_text(c))
    right, left = st.columns([1, 1.05], gap="medium")   # inputs first (left), result second (right)

    with right:
        with st.container(border=True):
            html('<div class="bs-title">💻 Input device details</div>')
            with st.form("inputs", border=False):
                grid = st.columns(2)
                for i, c in enumerate(INPUTS):
                    spec, s = C.FEATURES[c], STATS[c]
                    unit = f" ({spec['unit']})" if spec["unit"] else ""
                    with grid[i % 2]:
                        if spec["kind"] == "binary":
                            st.radio(f"{spec['icon']} {c}", ["Yes", "No"], key=f"in_{c}", horizontal=True)
                            st.caption(f"{DF[c].mean() * 100:.0f}% of laptops in the data are 'Yes'")
                        else:
                            st.text_input(f"{spec['icon']} {c}{unit}", key=f"in_{c}")
                            st.caption(f"Range in data: {fmt(s['min'], spec['decimals'])} – "
                                       f"{fmt(s['max'], spec['decimals'])} (median {fmt(s['median'], spec['decimals'])})")
                b1, b2 = st.columns([3, 1])
                submitted = b1.form_submit_button("▶  Predict battery health", type="primary", **WIDE_FORM)
                b2.form_submit_button("↺ Reset", on_click=reset_inputs, **WIDE_FORM)
            if not INCLUDE:
                st.caption(f"{C.OPTIONAL_FEATURE} is excluded from the model. You can include it under Model setup "
                           "in the sidebar.")

    row, errors, warnings = parse_inputs()
    if submitted:
        ss.submitted_row = True
    is_example = not ss.get("submitted_row")

    with left:
        if errors:
            with st.container(border=True):
                html('<div class="bs-title">Prediction result</div>')
                st.error("Please correct these inputs:\n\n" + "\n".join(f"- {e}" for e in errors))
            return
        ex = E.explain(MODEL, row, FEATURES, INPUTS, STATS)
        pred = ex["final"]
        band, colour = E.health_band(pred)
        html(f"""
        <div class="bs-panel">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="bs-title" style="margin:0">📊 Prediction result</div>
            <span class="bs-pill" style="background:#0f3b2e;color:{COL['green']};border:1px solid #1d6b50">
              ● {MODEL_NAME}</span></div>
          {'<div class="bs-muted" style="margin-top:.3rem">Showing a typical laptop from the data. Enter your details and press Predict.</div>' if is_example else ''}
          <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin-top:.4rem">
            <div style="text-align:center">{gauge_svg(pred, colour)}
              <div style="margin-top:-2.2rem"><span class="bs-pill" style="background:{colour}22;color:{colour};
                border:1px solid {colour}66;font-size:.9rem">{band}</span></div></div>
            <div style="flex:1;min-width:200px" class="bs-card">
              <div style="font-weight:700;color:{colour};font-size:1.05rem">{C.BAND_MESSAGES[band]}</div>
              <div style="margin-top:.5rem;line-height:1.55">The estimated battery health is <b>{pred:.1f}%</b>.
                The model's typical error is about <b>±{M['MAE']:.1f}</b> points, so the true value is most likely
                between <b>{max(pred - M['MAE'], 0):.1f}%</b> and <b>{min(pred + M['MAE'], 100):.1f}%</b>.</div>
            </div>
          </div>
        </div>""")
        st.write("")
        cards = [("Typical error", f"±{M['MAE']:.2f}", "average miss, in health points", COL["cyan"]),
                 ("90% of predictions within", f"±{M['90% error within (±)']:.2f}", "health points, on test data",
                  COL["green"])]
        if INCLUDE:
            cards.append((f"{C.OPTIONAL_FEATURE} (input)", f"{row.get(C.OPTIONAL_FEATURE, 0):,.0f}",
                          C.FEATURES[C.OPTIONAL_FEATURE]["unit"], COL["text"]))
        metric_cards(cards)
        for w in warnings:
            st.warning(w)

        wi = E.what_if(ex["drivers"], pred, STATS)
        comps = E.compare_to_data(row, DF, INPUTS + list(C.ENGINEERED))
        lines, advice = E.conclusion(pred, band, ex["drivers"], comps)
        st.write("")
        if is_example:
            st.caption("⬇️ Downloads (CSV and PDF report) appear after you press Predict.")
        else:
            ctx = dict(time=Rp.now(), version=ACTIVE["version"], model=MODEL_NAME, split=SPLIT, include=INCLUDE,
                       inputs=INPUTS, row=row, pred=pred, band=band, mae=float(M["MAE"]),
                       q90=float(M["90% error within (±)"]), explain=ex, whatif=wi, conclusion=lines, advice=advice)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Prediction details (CSV)", Rp.to_csv(ctx), f"battery_prediction_{stamp}.csv",
                               "text/csv", **WIDE_DL)
            if Rp.pdf_available():
                d2.download_button("⬇️ Report (PDF)", Rp.to_pdf(ctx), f"battery_report_{stamp}.pdf", "application/pdf",
                                   **WIDE_DL)
            else:
                d2.caption("PDF report needs one extra library: run `pip install fpdf2` and restart the app.")
            ss.last_pred = {"pred": pred, "band": band, "mae": float(M["MAE"]), "row": row, "base": ex["base"],
                            "drivers": ex["drivers"], "whatif": wi, "conclusion": lines}

    st.write("")
    p1, p2 = st.columns([1.15, 1], gap="medium")
    with p1:
        with st.container(border=True):
            html(f'<div class="bs-title">🎯 Model performance '
                 f'<span class="bs-muted" style="font-weight:400">(average of 5 test shuffles, {SPLIT} split)</span></div>')
            metric_cards([("R² score", f"{M['Mean R²']:.3f}", f"spread ±{M['R² spread (±)']:.3f}", COL["cyan"]),
                          ("MAE", f"{M['MAE']:.3f}", "health points", COL["green"]),
                          ("RMSE", f"{M['RMSE']:.3f}", "health points", COL["amber"]),
                          ("MAPE", f"{M['MAPE %']:.2f}%", "relative error", "#b99bff")])
    with p2:
        html(scale_bar(pred))
    st.write("")
    insights_panel(row, pred, band, ex, wi, comps, lines, advice)


def _pct_phrase(p):
    if 35 <= p <= 65:
        return "close to typical"
    return f"higher than {p:.0f}% of laptops" if p > 50 else f"lower than {100 - p:.0f}% of laptops"


def _value_text(c, v):
    if c in C.FEATURES:
        spec = C.FEATURES[c]
        if spec["kind"] == "binary":
            return "Yes" if v >= 0.5 else "No"
        return f"{fmt(v, spec['decimals'])} {spec['unit']}".strip()
    return f"{fmt(v, 1)} {C.ENGINEERED_UNITS.get(c, '')}".strip()


def insights_panel(row, pred, band, ex, wi, comps, lines, advice):
    html('<div class="bs-h" style="font-size:1.35rem;font-weight:600;margin:.2rem 0 .6rem 0">💡 Insights</div>')
    c1, c2 = st.columns([1.15, 1], gap="medium")
    with c1:
        body = "".join(f"<p style='margin:.3rem 0;line-height:1.55'>{l}</p>" for l in lines)
        tips = "".join(f"<li style='margin:.2rem 0'>{a}</li>" for a in advice)
        html(f"""<div class="bs-panel" style="border-color:#23508f;background:linear-gradient(110deg,#0e1f3b,#10284f);height:100%">
          <div class="bs-title">Conclusion</div>{body}
          {'<div style="margin-top:.6rem;font-weight:600">What may help</div><ul style="margin:.3rem 0 0 1.1rem;padding:0">' + tips + '</ul>' if tips else ''}
          </div>""")
    with c2:
        if wi:
            items = "".join(
                f"""<div class="bs-item">If <b>{w['Input']}</b> were typical ({_value_text(w['Input'], w['Typical'])}
                instead of {_value_text(w['Input'], w['Value'])}), estimated health would be
                <b style="color:{COL['green']}">{w['New']:.1f}%</b> ({w['Gain']:+.1f} points).</div>""" for w in wi)
        else:
            items = ('<div class="bs-item">None of the everyday habits is lowering this estimate by more than '
                     f'{C.INSIGHT_MIN_EFFECT} points, so there is no clear quick win.</div>')
        fixed = [d["Input"] for d in ex["drivers"] if d["Effect"] <= -C.INSIGHT_MIN_EFFECT
                 and not all(c in C.CHANGEABLE for c in d["Columns"])]
        note = (f'<div class="bs-muted" style="margin-top:.5rem;font-size:.8rem">{", ".join(fixed)} also lower the estimate '
                'but reflect history or hardware, so they are not included as what-ifs.</div>') if fixed else ""
        html(f"""<div class="bs-panel" style="height:100%"><div class="bs-title">🔁 What if</div>
          <div class="bs-muted" style="margin-bottom:.3rem">Estimated health if one everyday habit were typical, all else unchanged.</div>
          {items}{note}</div>""")

    st.write("")
    c3, c4 = st.columns([1.15, 1], gap="medium")
    with c3:
        with st.container(border=True):
            html('<div class="bs-title">From a typical laptop to yours</div>')
            steps = ex["steps"]
            labels = ["Typical laptop"] + [s["Input"] for s in steps]
            values = [ex["base"]] + [s["Effect"] for s in steps]
            measure = ["absolute"] + ["relative"] * len(steps)
            if abs(ex["other"]) >= 0.05:
                labels.append("Other inputs & combined effects")
                values.append(ex["other"])
                measure.append("relative")
            labels.append("Your laptop")
            values.append(ex["final"])
            measure.append("total")
            texts = [f"{ex['base']:.1f}%"] + [f"{v:+.1f}" for v in values[1:-1]] + [f"{ex['final']:.1f}%"]
            path = np.cumsum([ex["base"]] + values[1:-1])
            lo, hi = min(path.min(), ex["final"]), max(path.max(), ex["base"])
            fig = go.Figure(go.Waterfall(orientation="h", y=labels, x=values, measure=measure, text=texts,
                                         textposition="outside", cliponaxis=False,
                                         increasing=dict(marker=dict(color=COL["green"])),
                                         decreasing=dict(marker=dict(color=COL["red"])),
                                         totals=dict(marker=dict(color=COL["blue"])),
                                         connector=dict(line=dict(color="#3b5b8f", width=1))))
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[max(lo - (hi - lo) * 0.35 - 2, 0), min(hi + (hi - lo) * 0.25 + 2, 105)],
                             title="Estimated battery health (%)")
            chart(style_fig(fig, height=60 + 42 * len(labels)))
            st.caption("Start from a typical laptop in the data, then add the effect of each of your inputs "
                       "(red lowers, green raises). Inputs interact slightly, so a small combined step makes the total exact.")
    with c4:
        effect = {}
        for d in ex["drivers"]:
            for c in d["Columns"]:
                effect[c] = (d["Effect"], d["Input"] if len(d["Columns"]) > 1 else None)
        groups = {"⚠️ Concerns": [], "✅ In your favour": [], "• Normal": []}
        for c in INPUTS:
            e, group_name = effect.get(c, (0.0, None))
            spec = C.FEATURES[c]
            if spec["kind"] == "binary":
                share = DF[c].mean() * 100
                where = f"{share:.0f}% of laptops in the data are 'Yes'"
            else:
                pc = next((x["Percentile"] for x in comps if x["Attribute"] == c), 50)
                where = f"{_pct_phrase(pc)} (typical {_value_text(c, E.typical_value(c, STATS))})"
            together = " (measured together with the other capacity input)" if group_name else ""
            if e <= -C.INSIGHT_MIN_EFFECT:
                key, eff = "⚠️ Concerns", f"Lowers the estimate by {abs(e):.1f} points{together}."
            elif e >= C.INSIGHT_MIN_EFFECT:
                key, eff = "✅ In your favour", f"Raises the estimate by {e:.1f} points{together}."
            else:
                key, eff = "• Normal", "Little effect on this estimate."
            groups[key].append((abs(e), f'<div class="bs-item"><b>{c}: {_value_text(c, row[c])}</b> — {where}. '
                                         f'<span class="bs-muted">{eff}</span></div>'))
        colours = {"⚠️ Concerns": COL["red"], "✅ In your favour": COL["green"], "• Normal": COL["muted"]}
        out = ""
        for g, items in groups.items():
            if items:
                items.sort(key=lambda t: -t[0])
                out += f'<div class="bs-group" style="color:{colours[g]}">{g}</div>' + "".join(i for _, i in items)
        prof = "".join(
            f'<div class="bs-item"><b>{x["Attribute"]}: {_value_text(x["Attribute"], x["Value"])}</b> — '
            f'{_pct_phrase(x["Percentile"])}. <span class="bs-muted">{C.ENGINEERED[x["Attribute"]][0]}</span></div>'
            for x in comps if x["Attribute"] in C.ENGINEERED)
        html(f"""<div class="bs-panel"><div class="bs-title">How your laptop compares</div>{out}
          <div class="bs-group" style="color:{COL['cyan']}">ƒ Usage profile (calculated)</div>{prof}</div>""")

    with st.expander("How to read these insights"):
        st.markdown(f"""
- **Conclusion** summarises the estimate and the inputs that matter most for *this* laptop.
- **What if** re-runs the model with one everyday habit set to the typical value in the data. It shows what the model
  would estimate, not a guarantee of what will happen to a real battery.
- **From a typical laptop to yours** starts with the estimate for a laptop whose inputs are all typical (the median),
  then adds the effect of each of your inputs, biggest first.
- **How your laptop compares** sorts your inputs by their effect on this estimate: *Concerns* lower it by at least
  {C.INSIGHT_MIN_EFFECT} points, *In your favour* raise it by at least {C.INSIGHT_MIN_EFFECT} points. "Higher than 90% of
  laptops" means only 10% of laptops in the data have a higher value.
- All insights describe patterns in the training data and the model's behaviour. They are associations, not proof of cause and effect.
""")


# ============================================================ PAGE: Model performance
def page_models():
    html(f"""<div class="bs-panel"><div class="bs-title">🎯 Model comparison</div>
      <div style="line-height:1.6">Each of {len(TABLE)} models was trained and tested on <b>5 different shuffles</b>
      of the data with a <b>{SPLIT}</b> split ({RESULT['train_rows']:,} training rows, {RESULT['test_rows']:,} test rows each time).
      Models are ranked by <b>stability score = mean R² − spread</b>, which rewards accuracy and consistency.
      Choose a model below and press <b>Apply</b>; the sidebar shows the same choice.</div></div>""")
    st.write("")
    show = TABLE.copy()
    show.insert(0, "Rank", range(1, len(show) + 1))
    show.index = [f"⭐ {n}" if n == RESULT["recommended"] else n for n in show.index]
    show_table(show.round(4), max_rows=len(show), column_config={
        "Mean R²": st.column_config.NumberColumn(help="Share of variation in health explained (1 = perfect)."),
        "R² spread (±)": st.column_config.NumberColumn(help="Standard deviation of R² across the 5 shuffles."),
        "MAE": st.column_config.NumberColumn(help="Mean absolute error, in health points."),
        "Overfit gap": st.column_config.NumberColumn(help="Train R² − test R². Above ~0.05 suggests memorising."),
    })

    ensure_pending()
    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        with st.container(border=True):
            html('<div class="bs-title">Choose the model used for predictions</div>')
            st.radio("Model", MODEL_NAMES, key="p_model_page", format_func=model_label, label_visibility="collapsed",
                     on_change=sync_model, args=("p_model_page", "p_model_side"))
            setup_p, model_p = pending_changes("p_model_page")
            st.button("✓ Apply", type="primary", disabled=not (setup_p or model_p), on_click=apply_changes,
                      args=("p_model_page", CFG), key="apply_page", **WIDE_BTN)
            st.caption(f"Active now: **{MODEL_NAME}**. ⭐ = recommended (highest stability score)."
                       + (" Setup changes are pending in the sidebar; applying them retrains and selects the recommended model."
                          if setup_p else ""))
    with b:
        sel = ss.get("p_model_page") or MODEL_NAME
        r = TABLE.loc[sel]
        with st.container(border=True):
            state = "active" if sel == MODEL_NAME else "selected, not yet applied"
            html(f'<div class="bs-title">{sel} <span class="bs-muted" style="font-weight:400">({state})</span></div>')
            metric_cards([("Mean R²", f"{r['Mean R²']:.3f}", f"spread ±{r['R² spread (±)']:.3f}", COL["cyan"]),
                          ("MAE", f"{r['MAE']:.2f}", "health points", COL["green"])])
            st.write("")
            metric_cards([("Overfit gap", f"{r['Overfit gap']:.3f}", "train − test R²", COL["amber"]),
                          ("Fit time", f"{r['Fit time (s)']:.2f}s", "per training run", "#b99bff")])
            if sel != MODEL_NAME:
                diff = r["Mean R²"] - M["Mean R²"]
                st.caption(f"Compared with the active {MODEL_NAME}: R² {diff:+.3f}, MAE {r['MAE'] - M['MAE']:+.2f} points.")

    ps = RESULT["per_seed"]
    fig = go.Figure()
    for n in TABLE.index[::-1]:
        v = ps.loc[ps.Model == n, "R2"]
        fig.add_trace(go.Box(x=v, name=n, boxpoints="all", jitter=.3, pointpos=0, orientation="h",
                             marker=dict(color=COL["green"] if n == MODEL_NAME else COL["blue"], size=6),
                             line=dict(color=COL["green"] if n == MODEL_NAME else "#4d6fa8"), showlegend=False))
    fig.update_xaxes(title="Test R² on each of the 5 shuffles (green = active model)")
    chart(style_fig(fig, height=40 * len(TABLE) + 90, title="Score on every shuffle"))

    st.markdown(f"#### Validation of the active model: {MODEL_NAME}")
    y_te, y_hat, _ = RESULT["first_split"][MODEL_NAME]
    a, b, c = st.columns(3)
    with a:
        fig = go.Figure(go.Scattergl(x=y_te, y=y_hat, mode="markers", marker=dict(size=5, color=COL["cyan"], opacity=.5)))
        lo, hi = float(min(y_te.min(), y_hat.min())), float(max(y_te.max(), y_hat.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=COL["amber"], dash="dash")))
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title="Actual")
        fig.update_yaxes(title="Predicted")
        chart(style_fig(fig, title="Actual vs predicted (shuffle 1)"))
    with b:
        fig = go.Figure(go.Histogram(x=y_te - y_hat, nbinsx=35, marker_color=COL["green"]))
        fig.update_xaxes(title="Actual − predicted (points)")
        chart(style_fig(fig, title="Error distribution"))
    with c:
        key = (CFG, MODEL_NAME, "importance")
        if key not in S:
            S[key] = E.importance_for(RESULT, MODEL_NAME, DF, FEATURES, TEST_SIZE)
        imp = S[key].iloc[::-1]
        fig = go.Figure(go.Bar(x=imp.Importance, y=imp.Attribute, orientation="h", marker_color=COL["blue"],
                               error_x=dict(array=imp.Std)))
        fig.update_xaxes(title="Drop in R² when shuffled")
        chart(style_fig(fig, title="Feature importance"))

    st.markdown("#### Band accuracy: precision and recall")
    if "all_splits" in RESULT:
        bm = E.band_metrics(RESULT["all_splits"][MODEL_NAME])
        st.caption(f"The model predicts a number (health %), so its main metrics are R², MAE and RMSE. Precision and recall "
                   f"measure something related: how often the predicted **health band** (Good / Moderate / Poor) matches the "
                   f"real one. Pooled over the test sets of all 5 shuffles ({bm['n']:,} predictions) for {MODEL_NAME}.")
        a1, a2 = st.columns([1, 1.25], gap="medium")
        with a1:
            cmx = bm["confusion"]
            fig = go.Figure(go.Heatmap(z=cmx.values, x=[f"Predicted {c}" for c in cmx.columns],
                                       y=[f"Actual {c}" for c in cmx.index], colorscale="Blues", showscale=False,
                                       text=cmx.values, texttemplate="%{text}", textfont=dict(size=14)))
            fig.update_yaxes(autorange="reversed")
            chart(style_fig(fig, height=300, title="Confusion matrix (laptops)"))
        with a2:
            tb = bm["table"].copy()
            metric_cards([("Band accuracy", f"{bm['accuracy'] * 100:.1f}%", "predictions in the right band", COL["cyan"])])
            st.write("")
            show_table(tb.round(3), hide_index=True)
            empty = tb[tb["Actual laptops"] == 0]["Band"].tolist()
            st.caption("**Precision**: of laptops predicted in a band, the share really in it. **Recall**: of laptops really "
                       "in a band, the share the model put there. **F1**: their balance."
                       + (f" No test laptops were in the {', '.join(empty)} band, so it cannot be measured (shown as –)."
                          if empty else ""))
    else:
        st.caption("Band accuracy appears after the models retrain.")

    st.markdown("#### Calculated attributes: kept or not")
    info = R["selection_info"]
    st.caption(f"Each calculated attribute is added one at a time and scored with 5-fold cross-validation on the training "
               f"part of shuffle 1 (reference model: Hist Gradient Boosting). It is kept only if R² improves by at least "
               f"{C.FEATURE_GAIN_THRESHOLD}. Baseline R² without any: {info['baseline']:.4f}.")
    if len(R["selection"]):
        show_table(R["selection"].round(4), hide_index=True)
    st.caption(f"Kept: {', '.join(R['kept']) if R['kept'] else 'none — the original inputs already capture the signal'}. "
               "Calculated attributes are still used in the insights on the Predict page.")


# ============================================================ PAGE: Data
def do_replace(file_bytes):
    entry, gh = V.add_version(file_bytes, ss.get("data_note", ""), SECRETS)
    ss.flash = (f"Data replaced: now using version v{entry['version']} ({entry['rows']:,} rows). Models retrain automatically."
                + (f" {gh[1]}." if gh else ""))
    ss.data_mode = "Use existing CSV"
    ss.upload_n += 1
    ss.pop("last_pred", None)


def do_restore():
    v = ss.restore_choice
    entry, gh = V.restore(v, SECRETS)
    ss.flash = (f"Restored data version v{v} ({entry['rows']:,} rows). Models retrain automatically."
                + (f" {gh[1]}." if gh else ""))
    ss.pop("last_pred", None)


def page_data():
    log = V.ensure()
    act = V.active_version(log)
    html(f"""<div class="bs-panel"><div class="bs-title">📥 Data</div>
      <div style="line-height:1.6">The app trains on one CSV file. It is currently using <b>version v{act['version']}</b>
      ({act['rows']:,} rows, added {act['created']}{' — ' + act['note'] if act.get('note') else ''}).
      Uploading a new file keeps every earlier version, so you can restore any of them.</div></div>""")
    st.write("")
    st.radio("Data source", ["Use existing CSV", "Upload new CSV"], key="data_mode", horizontal=True)
    if ss.data_mode == "Use existing CSV":
        st.caption(f"Using version v{act['version']}. First rows of the data the models are trained on:")
        show_table(DF.head(10), hide_index=True)
    else:
        up = st.file_uploader("Choose a CSV file", type=["csv"], key=f"upload_{ss.upload_n}")
        if up is not None:
            raw = up.getvalue()
            try:
                raw_df = E.read_csv_any(io.BytesIO(raw))
            except Exception as e:
                st.error(f"Could not read this file as a CSV: {e}")
                return
            rep = E.validate_upload(raw_df, DF)
            st.markdown("#### Check report")
            for e in rep["errors"]:
                st.error(e)
            for w in rep["warnings"]:
                st.warning(w)
            for i in rep["info"]:
                st.info(i)
            if not rep["errors"]:
                if rep.get("compare") is not None and len(rep["compare"]):
                    st.caption("Current data compared with the new file:")
                    show_table(rep["compare"].round(2), max_rows=20, hide_index=True)
                st.caption("First rows of the new file after cleaning:")
                show_table(rep["clean"].head(8), hide_index=True)
                st.text_input("Version note (optional)", key="data_note", placeholder="e.g. Added March survey data")
            st.button("🔁 Replace", type="primary", disabled=bool(rep["errors"]), on_click=do_replace, args=(raw,))
            st.caption("Replace saves this file as a new version, makes it active and retrains all models. "
                       "The previous versions stay available below.")

    st.markdown("#### Version history")
    rows = [{"Version": f"v{v['version']}" + ("  ✓ active" if v["version"] == log["active"] else ""),
             "Added": v["created"], "Rows": v["rows"], "Note": v.get("note", ""),
             "How": {"original": "Initial file", "upload": "Uploaded in app", "manual": "Replaced outside app"}
             .get(v.get("source"), v.get("source", ""))} for v in sorted(log["versions"], key=lambda v: -v["version"])]
    show_table(pd.DataFrame(rows), max_rows=12, hide_index=True)
    others = [v["version"] for v in log["versions"] if v["version"] != log["active"]]
    if others:
        c1, c2 = st.columns([2, 1])
        c1.selectbox("Restore an earlier version", sorted(others, reverse=True), key="restore_choice",
                     format_func=lambda v: f"v{v}")
        c2.write("")
        c2.button("↩️ Restore", on_click=do_restore, **WIDE_BTN)
    st.caption(f"Versions are stored in the {V.storage_label(SECRETS)}.")


# ============================================================ PAGE: About dataset
def page_about():
    html(f"""<div class="bs-panel"><div class="bs-title">🗂️ About the dataset</div>
      <div style="line-height:1.7">Data version <b>v{ACTIVE['version']}</b>, added {ACTIVE['created']}.<br>
      Rows read: {REPORT['rows_raw']:,} → used: {REPORT['rows_final']:,}
      (removed {REPORT['missing_target_removed']} without a {C.TARGET} value and {REPORT['duplicates_removed']} duplicates).
      {'<br>Ignored extra columns: ' + ', '.join(map(str, REPORT['ignored_columns'])) if REPORT['ignored_columns'] else ''}</div></div>""")
    st.write("")
    src = os.path.join(P(C.VERSIONS_DIR), ACTIVE["file"])
    if not os.path.exists(src):          # fallback: the active data file itself
        src = P(C.DATA_FILE)
    how = {"original": "the original data file", "upload": f"uploaded in the app on {ACTIVE['created']}",
           "manual": f"replaced outside the app (recorded {ACTIVE['created']})"}.get(ACTIVE.get("source"), "the data file")
    d1, d2 = st.columns([1.4, 2.6])
    with open(src, "rb") as f:
        d1.download_button(f"⬇️ Download dataset (v{ACTIVE['version']})", f.read(), file_name=ACTIVE["file"],
                           mime="text/csv", type="primary", key="dl_dataset", **WIDE_DL)
    d2.caption(f"The exact file the current models are trained on: version v{ACTIVE['version']}, {how}, "
               f"{ACTIVE['rows']:,} rows, unchanged. Cleaning (removing duplicates and rows without {C.TARGET}) "
               "happens inside the app, so the file still contains those rows.")
    st.write("")
    rows = []
    for c, spec in C.FEATURES.items():
        if c not in DF.columns:
            continue
        s = STATS[c]
        rows.append({"Attribute": c, "Meaning": spec["desc"], "Unit": spec["unit"] or "1 = yes, 0 = no",
                     "Type": "Yes/No" if spec["kind"] == "binary" else "Number",
                     "Role": "Optional input (off by default)" if c == C.OPTIONAL_FEATURE else "Input",
                     "Min": s["min"], "Median": s["median"], "Max": s["max"], "Missing": s["missing"],
                     "Why it may matter": spec["why"]})
    s = STATS[C.TARGET]
    rows.append({"Attribute": C.TARGET, "Meaning": "Current capacity as a percentage of the original design capacity.",
                 "Unit": C.TARGET_UNIT, "Type": "Number", "Role": "Target (predicted)", "Min": s["min"],
                 "Median": s["median"], "Max": s["max"], "Missing": 0, "Why it may matter": "This is what the model predicts."})
    st.markdown("#### Attribute dictionary")
    show_table(pd.DataFrame(rows).round(2), max_rows=len(rows), hide_index=True)
    st.markdown("#### Calculated attributes")
    show_table(pd.DataFrame([{"Attribute": n, "Formula": f, "Meaning": d, "Median": STATS[n]["median"],
                              "Used by model": "Yes" if n in R["kept"] else "No (insights only)"}
                             for n, (f, _, d) in C.ENGINEERED.items()]).round(2), hide_index=True)
    st.info("To update the data, use the Data page: choose Upload new CSV and press Replace.")


# ============================================================ User guide + Mr. GK
def correlations():
    key = (DATA_ID, "corr")
    if key not in S:
        eng = E.add_engineered(DF)
        cols = [c for c in list(C.FEATURES) + list(C.ENGINEERED) if c in eng.columns]
        S[key] = eng[cols + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET)
    return S[key]


@st.cache_resource
def guide_index(mtime):
    return G.build_index(open(P(C.GUIDE_FILE), encoding="utf-8").read())


def live_stats_html():
    rows = "".join(
        f"<tr><td>{c}</td><td>{fmt(STATS[c]['min'], 2)}</td><td>{fmt(STATS[c]['median'], 2)}</td>"
        f"<td>{fmt(STATS[c]['max'], 2)}</td><td>{STATS[c]['missing']}</td></tr>"
        for c in list(C.FEATURES) + [C.TARGET] if c in STATS)
    lb = "".join(f"<tr><td>{n}</td><td>{r['Mean R²']:.4f}</td><td>±{r['R² spread (±)']:.4f}</td><td>{r['MAE']:.3f}</td></tr>"
                 for n, r in TABLE.head(5).iterrows())
    return f"""<div class="live"><h4>Live figures from the running app</h4>
      <p>Data version <b>v{ACTIVE['version']}</b>, {len(DF):,} rows. Setup: {SPLIT} split,
      {C.OPTIONAL_FEATURE} {'included' if INCLUDE else 'excluded'}, active model <b>{MODEL_NAME}</b>.</p>
      <div class="table-wrap"><table><thead><tr><th>Attribute</th><th>Min</th><th>Median</th><th>Max</th><th>Missing</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
      <p>Top of the current leaderboard:</p>
      <div class="table-wrap"><table><thead><tr><th>Model</th><th>Mean R²</th><th>Spread</th><th>MAE</th></tr></thead>
      <tbody>{lb}</tbody></table></div></div>"""


def guide_html(compact):
    g = open(P(C.GUIDE_FILE), encoding="utf-8").read().replace("<!--LIVE_DATA-->", live_stats_html())
    if compact:
        g = g.replace("<body>", '<body class="compact">', 1)
    g = T(g)
    jump = ss.get("guide_jump")
    if jump:
        target, nonce = jump
        g = g.replace("</body>", f"""<script>/* jump {nonce} */ setTimeout(function(){{
          var el=document.getElementById("{target}"); if(el){{ el.scrollIntoView({{block:"start"}});
          el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash"); }} }}, 350);</script></body>""", 1)
    return g


def set_jump(target):
    ss.view = "guide"
    ss.guide_jump = (target, ss.get("jump_n", 0) + 1)
    ss.jump_n = ss.get("jump_n", 0) + 1


def ask(q):
    ss.gk_q = q


def app_state():
    return {"version": ACTIVE["version"], "rows": len(DF), "split": SPLIT, "include": INCLUDE, "inputs": INPUTS,
            "kept": R["kept"], "model": MODEL_NAME, "recommended": RESULT["recommended"],
            "leaderboard": [f"{n} {r['Mean R²']:.3f} ± {r['R² spread (±)']:.3f}, MAE {r['MAE']:.2f}"
                            for n, r in TABLE.iterrows()],
            "prediction": ss.get("last_pred"),
            "table": {n: {k: float(v) for k, v in r.items()} for n, r in TABLE.iterrows()},
            "metrics": {k: float(v) for k, v in M.items()},
            "stats": STATS, "corr": {k: float(v) for k, v in correlations().items()},
            "train_rows": RESULT["train_rows"], "test_rows": RESULT["test_rows"],
            "selection": R["selection"].to_dict("records") if len(R["selection"]) else [],
            "baseline": R["selection_info"]["baseline"],
            "importance": S[(CFG, MODEL_NAME, "importance")].to_dict("records") if (CFG, MODEL_NAME, "importance") in S else [],
            "report": REPORT,
            "bands": (lambda bm: {"accuracy": bm["accuracy"], "n": bm["n"], "table": bm["table"].to_dict("records")})(
                E.band_metrics(RESULT["all_splits"][MODEL_NAME])) if "all_splits" in RESULT else None,
            "eda": (lambda d: {"uni": d["uni"].to_dict("records"), "out": d["out"].to_dict("records"),
                               "vif": d["vif"].to_dict("records"), "dec": d["dec"].to_dict("records")})(eda_data())}


def mrgk_panel(idx):
    key = G.api_key(SECRETS)
    used = G.usage_today()
    ai_on = bool(key) and used < C.MRGK_DAILY_CAP
    mode = ("AI mode" if ai_on else "Guide mode · daily AI limit reached" if key else "Guide mode")
    mode_col = COL["green"] if ai_on else COL["amber"]
    hl, hr = st.columns([5, 1])
    with hl:
        html(f"""<div style="display:flex;align-items:center;gap:.7rem">
          <img class="bs-avatar" style="width:48px;height:48px" src="data:image/png;base64,{AVATAR_B64}" alt="{C.MRGK_NAME}">
          <div><div class="bs-h" style="font-weight:700;font-size:1.15rem">{C.MRGK_NAME}</div>
          <div class="bs-muted" style="font-size:.76rem">AI assistant · can make mistakes</div>
          <span class="bs-pill" style="margin-top:.2rem;background:{mode_col}22;color:{mode_col};border:1px solid {mode_col}66;
            font-size:.7rem">● {mode}</span></div></div>""")
    with hr:
        st.button("", key="gk_close", icon=":material/close:", on_click=close_gk, help=f"Close {C.MRGK_NAME}")
    box = st.container(height=430, border=True)
    with box:
        if not ss.chat:
            with st.chat_message("assistant", avatar=P(C.MRGK_AVATAR)):
                st.markdown("Hi, I'm Mr. GK. Ask me about the data, the models, any calculation, or your own prediction.")
        for i, m in enumerate(ss.chat):
            with st.chat_message(m["role"], avatar=P(C.MRGK_AVATAR) if m["role"] == "assistant" else "🧑"):
                st.markdown(m["display"])
                for j, c in enumerate(m.get("cites", [])):
                    st.button(f"📖 {c['number']}  {c['title']}", key=f"cite_{i}_{j}", on_click=set_jump, args=(c["id"],))
                if m.get("note"):
                    st.caption(m["note"])
    q = st.chat_input(f"Ask {C.MRGK_NAME} about the project…", key="gk_input")
    q = q or ss.pop("gk_q", None)
    if q:
        import mrgk_kb as KB
        state = app_state()
        kb_text, kb_ids, _ = KB.answer(q, state)
        chunks = G.relevant(idx, q, C.MRGK_CONTEXT_SECTIONS)
        if kb_ids:   # the knowledge-base sections go first in the AI context
            by_id = {c["id"]: c for c in idx["chunks"]}
            chunks = [by_id[i] for i in kb_ids if i in by_id] + [c for c in chunks if c["id"] not in kb_ids]
            chunks = chunks[:C.MRGK_CONTEXT_SECTIONS]
        text, note = None, ""
        if ai_on:
            try:
                history = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
                text = G.ask_ai(key, q, history, chunks, G.state_to_text(state), reference=kb_text)
                G.bump_usage()
            except Exception:
                note = "The AI service is unavailable right now, so this answer comes from the built-in knowledge base."
        if text:
            cites = G.citations(text, idx, chunks) if re.search(r"\[§", text) else G.section_citations(idx, kb_ids)
        elif kb_text:
            text, cites = kb_text, G.section_citations(idx, kb_ids)
        else:
            text = G.guide_answer(q, chunks, state)
            cites = G.citations(text, idx, chunks)
        display = re.sub(r"\s*\[§\s*[\d.]+[^\]]*\]", "", text)   # section buttons below the answer replace inline refs
        ss.chat += [{"role": "user", "content": q, "display": q},
                    {"role": "assistant", "content": text, "display": display, "cites": cites, "note": note}]
        st.rerun()
    with st.container(key="gkfaq"):
        with st.expander("FAQ"):
            faq = [("Why is FCC off?", "Why is Full Charge Capacity off by default?"),
                   ("Best model choice", "How is the recommended model chosen?"),
                   ("What is R²?", "What does R² mean?"),
                   ("Explain my prediction", "Explain my prediction"),
                   ("How many variables?", "How many variables are available?"),
                   ("Limitations", "What are the limitations?")]
            fc = st.columns(2)
            for k, (label, q) in enumerate(faq):
                fc[k % 2].button(label, key=f"starter_{k}", on_click=ask, args=(q,), help=q, **WIDE_BTN)
    if ss.chat:
        st.button("Clear conversation", key="gk_clear", on_click=lambda: ss.update(chat=[]))


def page_guide():
    st.markdown("<style>.block-container{max-width:none;padding-left:2rem;padding-right:2rem}</style>",
                unsafe_allow_html=True)
    t1, t3 = st.columns([4, 1.4])
    t1.caption("Search the guide from its contents panel; press / to jump to the search box. "
               "Choose any page in the sidebar to return to the app.")
    raw = open(P(C.GUIDE_FILE), encoding="utf-8").read()
    t3.download_button("⬇️ Download guide (HTML)", raw.encode("utf-8"), "BatteryClinic_User_Guide.html", "text/html",
                       **WIDE_DL)
    embed_html(guide_html(compact=False), 880)


if ss.view == "guide":
    page_guide()
else:
    {"Data": page_data, "About dataset": page_about, "EDA & insights": page_eda, "Model performance": page_models,
     "Predict": page_predict, "Batch prediction": page_batch}[ss.page]()

if ss.get("gk_open"):
    with st.container(key="gkfloat"):
        mrgk_panel(guide_index(os.path.getmtime(P(C.GUIDE_FILE))))
