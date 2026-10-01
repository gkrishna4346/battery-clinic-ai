"""
Data versions for Battery Clinic AI.

Every data file ever used is kept in data/versions/ with a log (versions.json).
The active version is copied to config.DATA_FILE, which the app trains on.

Storage
  - Local (default): files stay in the project folder.
  - GitHub (when deployed): if GITHUB_TOKEN and GITHUB_REPO are set in Streamlit secrets,
    each replace/restore is also committed to the repository, so it survives restarts
    and GitHub keeps the full history.
"""
import base64
import hashlib
import json
import os
import shutil
from datetime import datetime

import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda rel: os.path.join(HERE, rel)  # noqa: E731


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _read_log():
    path = P(C.VERSION_LOG)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_log(log):
    os.makedirs(P(C.VERSIONS_DIR), exist_ok=True)
    with open(P(C.VERSION_LOG), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def _count_rows(path):
    with open(path, "rb") as f:
        return max(sum(1 for line in f if line.strip()) - 1, 0)


def _store_file(src_path, source, note, log):
    number = max([v["version"] for v in log["versions"]], default=0) + 1
    stem = os.path.splitext(os.path.basename(C.DATA_FILE))[0]
    name = f"{stem}_v{number}_{datetime.now().strftime('%Y-%m-%d')}.csv"
    os.makedirs(P(C.VERSIONS_DIR), exist_ok=True)
    shutil.copyfile(src_path, os.path.join(P(C.VERSIONS_DIR), name))
    entry = {"version": number, "file": name, "created": _now(), "rows": _count_rows(src_path),
             "note": note, "source": source, "md5": _md5(src_path)}
    log["versions"].append(entry)
    return entry


def ensure():
    """Create the log on first run, and register a hand-replaced data file as a new version."""
    data = P(C.DATA_FILE)
    log = _read_log()
    if log is None:
        log = {"active": 1, "versions": [], "history": []}
        _store_file(data, "original", "Initial dataset", log)
        log["history"].append({"time": _now(), "action": "created", "version": 1})
        _write_log(log)
        return log
    active = next((v for v in log["versions"] if v["version"] == log["active"]), None)
    if os.path.exists(data) and (active is None or active.get("md5") != _md5(data)):
        entry = _store_file(data, "manual", "File replaced outside the app", log)
        log["active"] = entry["version"]
        log["history"].append({"time": _now(), "action": "detected", "version": entry["version"]})
        _write_log(log)
    return log


def active_version(log):
    return next(v for v in log["versions"] if v["version"] == log["active"])


def add_version(file_bytes, note, secrets=None):
    """Save uploaded bytes as a new version and make it active."""
    log = ensure()
    tmp = P(C.VERSIONS_DIR + "/_upload.tmp")
    with open(tmp, "wb") as f:
        f.write(file_bytes)
    entry = _store_file(tmp, "upload", note.strip() or "Uploaded in the app", log)
    os.remove(tmp)
    shutil.copyfile(os.path.join(P(C.VERSIONS_DIR), entry["file"]), P(C.DATA_FILE))
    log["active"] = entry["version"]
    log["history"].append({"time": _now(), "action": "replace", "version": entry["version"]})
    _write_log(log)
    gh = github_sync([entry["file"]], f"Battery Clinic AI: data v{entry['version']} — {entry['note']}", secrets)
    return entry, gh


def restore(version, secrets=None):
    log = ensure()
    entry = next(v for v in log["versions"] if v["version"] == version)
    shutil.copyfile(os.path.join(P(C.VERSIONS_DIR), entry["file"]), P(C.DATA_FILE))
    log["active"] = version
    log["history"].append({"time": _now(), "action": "restore", "version": version})
    _write_log(log)
    gh = github_sync([], f"Battery Clinic AI: restore data v{version}", secrets)
    return entry, gh


# ------------------------------------------------------------------ GitHub (optional)
def github_settings(secrets):
    try:
        token, repo = secrets.get("GITHUB_TOKEN"), secrets.get("GITHUB_REPO")
    except Exception:
        return None
    if not token or not repo:
        return None
    return {"token": token, "repo": repo, "branch": secrets.get("GITHUB_BRANCH", "main"),
            "prefix": secrets.get("GITHUB_PATH_PREFIX", "").strip("/")}


def storage_label(secrets):
    s = github_settings(secrets)
    return f"GitHub repository {s['repo']}" if s else "project folder on this computer"


def github_sync(version_files, message, secrets):
    """Commit the active data file, the log and any new version files. Returns (ok, detail)."""
    s = github_settings(secrets) if secrets is not None else None
    if not s:
        return None
    import requests
    api = f"https://api.github.com/repos/{s['repo']}/contents"
    headers = {"Authorization": f"Bearer {s['token']}", "Accept": "application/vnd.github+json"}
    paths = [C.DATA_FILE, C.VERSION_LOG] + [f"{C.VERSIONS_DIR}/{f}" for f in version_files]
    try:
        for rel in paths:
            remote = f"{s['prefix']}/{rel}" if s["prefix"] else rel
            with open(P(rel), "rb") as f:
                content = base64.b64encode(f.read()).decode()
            r = requests.get(f"{api}/{remote}", headers=headers, params={"ref": s["branch"]}, timeout=20)
            body = {"message": message, "content": content, "branch": s["branch"]}
            if r.status_code == 200:
                body["sha"] = r.json()["sha"]
            r = requests.put(f"{api}/{remote}", headers=headers, json=body, timeout=30)
            if r.status_code not in (200, 201):
                return False, f"GitHub rejected {remote}: {r.status_code} {r.text[:150]}"
        return True, f"Committed to {s['repo']} ({s['branch']})"
    except Exception as e:  # network problems must never break the app
        return False, f"GitHub sync failed: {e}"
