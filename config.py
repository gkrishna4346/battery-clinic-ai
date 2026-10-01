"""
Battery Clinic AI - project configuration.

Everything that describes the *domain* lives here: column names, labels, units,
health thresholds, engineered features and insight rules. The engine and the app
read from this file, so adapting the tool to a changed dataset usually means
editing only this file.
"""

APP_TITLE = "Battery Clinic AI"
APP_SUBTITLE = "Laptop battery health checkup"
DATA_FILE = "data/battery_health.csv"
GUIDE_FILE = "docs/user_guide.html"
HERO_IMAGE = "assets/hero.jpg"          # optional; a built-in illustration is used if missing

TARGET = "Battery Health"
TARGET_UNIT = "%"

# Full Charge Capacity almost determines Battery Health on its own
# (health ≈ full charge capacity ÷ design capacity × 100), so it is OFF by default.
OPTIONAL_FEATURE = "Full Charge Capacity"
OPTIONAL_FEATURE_DEFAULT = False

# ------------------------------------------------------------------ input columns
# kind: "numeric" (free-text number box) or "binary" (Yes/No toggle stored as 1/0)
FEATURES = {
    "Battery Age": {
        "unit": "years", "kind": "numeric", "icon": "📅", "decimals": 0,
        "desc": "How long the battery has been in use.",
        "why": "Lithium-ion cells lose capacity with time through calendar ageing, even when idle.",
    },
    "Daily Usage Hours": {
        "unit": "hours/day", "kind": "numeric", "icon": "⏱️", "decimals": 1,
        "desc": "Average hours per day the laptop is used.",
        "why": "More hours of use generally mean more charge cycles and more time at elevated temperature.",
    },
    "Gaming User": {
        "unit": "", "kind": "binary", "icon": "🎮", "decimals": 0,
        "desc": "Whether the laptop is regularly used for gaming (1 = yes, 0 = no).",
        "why": "Gaming produces sustained high power draw and heat.",
    },
    "Design Capacity": {
        "unit": "mAh", "kind": "numeric", "icon": "🔋", "decimals": 0,
        "desc": "Capacity the battery was designed to hold when new (manufacturer specification).",
        "why": "A hardware specification; mainly provides scale for the other capacity figures.",
    },
    "Cycle Count": {
        "unit": "cycles", "kind": "numeric", "icon": "🔄", "decimals": 0,
        "desc": "Number of full charge–discharge cycles completed so far.",
        "why": "Each cycle causes a small amount of permanent wear; the most direct usage-based wear signal.",
    },
    "CPU Usage": {
        "unit": "%", "kind": "numeric", "icon": "🧠", "decimals": 1,
        "desc": "Average processor utilisation.",
        "why": "Higher load means higher power draw and more heat.",
    },
    "GPU Usage": {
        "unit": "%", "kind": "numeric", "icon": "🖥️", "decimals": 1,
        "desc": "Average graphics processor utilisation.",
        "why": "GPUs are among the most power-hungry components in a laptop.",
    },
    "Power Consumption": {
        "unit": "W", "kind": "numeric", "icon": "⚡", "decimals": 1,
        "desc": "Average power drawn by the laptop.",
        "why": "Higher discharge rates put more stress on the cells.",
    },
    "Average Temperature": {
        "unit": "°C", "kind": "numeric", "icon": "🌡️", "decimals": 1,
        "desc": "Average operating temperature of the battery / device.",
        "why": "Heat accelerates the chemical reactions that degrade battery cells.",
    },
    "Full Charge Capacity": {
        "unit": "mAh", "kind": "numeric", "icon": "🔌", "decimals": 0,
        "desc": "Charge the battery can actually hold today when fully charged.",
        "why": "Directly reflects accumulated wear; Battery Health is essentially this ÷ Design Capacity.",
    },
}

# ------------------------------------------------------------------ health bands
# (lower bound inclusive, label, colour). Checked from the top down.
HEALTH_BANDS = [
    (70, "Good", "#34d399"),
    (40, "Moderate", "#f5b041"),
    (0, "Poor", "#f0605d"),
]
BAND_MESSAGES = {
    "Good": "Your battery is in good condition.",
    "Moderate": "Your battery shows noticeable wear.",
    "Poor": "Your battery is heavily worn.",
}

# ------------------------------------------------------------------ engineered features
# name -> (formula text, list of source columns, description)
ENGINEERED = {
    "Cycles per Year": ("Cycle Count ÷ Battery Age", ["Cycle Count", "Battery Age"],
                        "How intensively the battery has been cycled, independent of its age."),
    "Heat Stress": ("Average Temperature × Daily Usage Hours", ["Average Temperature", "Daily Usage Hours"],
                    "Combined exposure to heat over a typical day (°C·hours)."),
    "Workload Intensity": ("(CPU Usage + GPU Usage) ÷ 2", ["CPU Usage", "GPU Usage"],
                           "One combined measure of how hard the laptop works."),
    "Power per Usage Hour": ("Power Consumption ÷ Daily Usage Hours", ["Power Consumption", "Daily Usage Hours"],
                             "Separates high-power use from simply long use."),
    "Gaming Load": ("Gaming User × GPU Usage", ["Gaming User", "GPU Usage"],
                    "Sustained GPU load specific to gamers (0 for non-gamers)."),
}
ENGINEERED_UNITS = {"Cycles per Year": "cycles/yr", "Heat Stress": "°C·h/day",
                    "Workload Intensity": "%", "Power per Usage Hour": "W per h", "Gaming Load": "%"}
# An engineered feature is kept only if it raises the mean cross-validated R² by at least this much
FEATURE_GAIN_THRESHOLD = 0.002

# ------------------------------------------------------------------ modelling
SPLIT_OPTIONS = {"70 / 30": 0.30, "75 / 25": 0.25, "80 / 20": 0.20, "85 / 15": 0.15}
DEFAULT_SPLIT = "80 / 20"
STABILITY_SEEDS = [11, 23, 42, 57, 89]      # 5 internal shuffles
CV_FOLDS_FOR_SELECTION = 5

# ------------------------------------------------------------------ insight rules
# Advice shown only when this input is pulling the prediction DOWN by more than
# INSIGHT_MIN_EFFECT points compared with a typical laptop.
INSIGHT_MIN_EFFECT = 0.5
ADVICE = {
    "Battery Age": "Age-related wear cannot be reversed; keep an eye on health and plan a replacement as it falls toward the Poor band.",
    "Cycle Count": "Reducing full drain-to-empty cycles (topping up partially instead) is associated with slower wear.",
    "Average Temperature": "Keeping the laptop cool — good ventilation, hard surfaces, avoiding direct sun — may slow further wear.",
    "Daily Usage Hours": "Long daily use on battery adds cycles; running on mains power for long sessions may help.",
    "CPU Usage": "Sustained high processor load adds heat; closing background apps can reduce it.",
    "GPU Usage": "Heavy graphics load adds heat and power draw; lower graphics settings on battery may help.",
    "Power Consumption": "High power draw stresses the cells; power-saving modes reduce the discharge rate.",
    "Gaming User": "Gaming sessions are demanding; plugging in while gaming reduces battery stress.",
    "Design Capacity": "Design capacity is a hardware specification — no action needed.",
    "Full Charge Capacity": "Full charge capacity directly reflects the wear already accumulated.",
    "Capacity (Design + Full Charge)": "Full charge capacity relative to design capacity directly reflects wear already accumulated — it is a measurement of health rather than a cause.",
}

# ------------------------------------------------------------------ insights v2
# Inputs a user can realistically change through habits (used for what-if suggestions).
# Age, cycle count and capacities reflect history or hardware and cannot be changed.
CHANGEABLE = ["Daily Usage Hours", "Gaming User", "CPU Usage", "GPU Usage", "Power Consumption", "Average Temperature"]
WATERFALL_STEPS = 6            # largest driver steps shown before "other inputs & combined effects"
MIN_CORR_FOR_DIRECTION = 0.05  # below this, an attribute is treated as having little link with health

# ------------------------------------------------------------------ data versions
VERSIONS_DIR = "data/versions"
VERSION_LOG = "data/versions/versions.json"
MIN_ROWS_WARNING = 200         # fewer rows than this -> warning in the upload check report
MIN_ROWS_ERROR = 30            # fewer rows than this -> upload blocked

# ------------------------------------------------------------------ Mr. GK assistant
MRGK_NAME = "Mr. GK"
MRGK_AVATAR = "assets/mrgk.png"
MRGK_MODEL = "claude-haiku-4-5-20251001"   # any Anthropic model id; Haiku is fast and low-cost
MRGK_MAX_TOKENS = 1200
MRGK_DAILY_CAP = 60           # AI answers per day across all users; guide mode after that
MRGK_CONTEXT_SECTIONS = 6     # guide sections sent with each question
