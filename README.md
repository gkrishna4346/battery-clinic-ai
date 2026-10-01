# Battery Clinic AI  (v3.1)

Laptop battery health prediction with in-app data management, model training, comparison, explanation,
and a built-in assistant, Mr. GK.

## Run locally

```
conda create -n batterysense python=3.11 -y
conda activate batterysense
pip install -r requirements.txt
python -m streamlit run app.py
```

The first start trains all models (about 20–40 seconds); results are then cached.

## Your data

Open the **Data** page → **Upload new CSV** → check the report → **Replace**. Every version is kept and can be restored.
**The bundled file is a synthetic stand-in** with the correct 11 columns, for testing only.

## Optional settings

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
- `ANTHROPIC_API_KEY` turns on Mr. GK's AI mode (otherwise he works in free Guide mode).
- `GITHUB_TOKEN` + `GITHUB_REPO` store data versions in your GitHub repository when deployed.

## Deploy

Push this folder to GitHub (secrets.toml is ignored), create an app at share.streamlit.io pointing to `app.py`,
and paste your secrets into the app's Settings → Secrets.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `engine.py` | Loading, checks, calculated attributes, training, stability check, prediction, insights |
| `versions.py` | Data versions, restore, optional GitHub storage |
| `report.py` | CSV and PDF prediction records |
| `mrgk.py` | Mr. GK assistant |
| `config.py` | Columns, units, health bands, thresholds, advice text, assistant settings |
| `docs/user_guide.html` | Full searchable user guide |
| `data/battery_health.csv` | Active training data (`data/versions/` holds all versions) |

See the user guide for every attribute, pipeline stage and decision.
