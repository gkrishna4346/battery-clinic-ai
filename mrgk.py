"""
Mr. GK - the Battery Clinic AI assistant.

Two modes
  AI mode    : the question, the most relevant guide sections and the live app state are sent to the
               Anthropic API; answers must come from that material and cite guide sections as [§9.4].
  Guide mode : no API key, daily cap reached, or the API fails -> the best-matching guide sections
               are returned with short extracts. Free and always available.
"""
import html as _html
import math
import os
import re
from collections import Counter

import config as C

STOP = set("""a an and are as at be but by can do does for from how i if in into is it its me my of on or our so
that the their them then there these this to was what when where which who why will with would you your
about should could than too very also any all not no yes please tell explain mean means""".split())


SYNONYMS = [(r"\br[\s-]?squared\b", "r²"), (r"\br2\b", "r²"), (r"\bfcc\b", "full charge capacity"),
            (r"\bml\b", "model"), (r"\bcv\b", "cross-validation"), (r"\btemp\b", "temperature")]


def _tokens(text):
    t = text.lower()
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    return [w for w in re.findall(r"[a-z0-9²%]+", t) if w not in STOP and len(w) > 1]


def _clean(fragment):
    fragment = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", fragment)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    text = re.sub(r"\s+", " ", _html.unescape(fragment)).strip()
    return re.sub(r"\s+([,.;:!?)])", r"\1", text)


def build_index(guide_html):
    """Split the guide into chunks at every anchored section, sub-heading and attribute card."""
    body = guide_html.split("<main", 1)[-1].split("<script", 1)[0]
    anchor_re = re.compile(r'<(section|h3|div class="attr"|h4)[^>]*\bid="([^"]+)"[^>]*>')
    anchors = list(anchor_re.finditer(body))
    chunks, section_num, h3_num = [], "", ""
    for i, m in enumerate(anchors):
        tag, cid = m.group(1), m.group(2)
        segment = body[m.start(): anchors[i + 1].start() if i + 1 < len(anchors) else len(body)]
        if tag == "section":
            n = re.search(r'<h2><span class="n">([\d.]+)</span>', segment)
            section_num = n.group(1) if n else section_num
        head = re.search(r"<h[234][^>]*>([\s\S]*?)</h[234]>", segment)
        title_html = head.group(1) if head else cid
        num_m = re.search(r'<span class="n">([\d.]+)</span>', title_html)
        if tag == "section":
            h3_num = section_num
        number = num_m.group(1) if num_m else (h3_num if tag == "h4" else section_num)
        if tag == "h3" and num_m:
            h3_num = number
        title = _clean(re.sub(r'<span class="n">[\d.]+</span>', "", title_html))
        text = _clean(segment)
        if len(text) < 40:
            continue
        body_only = re.sub(r"<h[234][^>]*>[\s\S]*?</h[234]>", " ", segment, count=1)
        prose = _clean(re.sub(r"<table[\s\S]*?</table>", " ", body_only))
        chunks.append({"id": cid, "number": number, "title": title, "text": text, "prose": prose,
                       "tokens": Counter(_tokens(text)),
                       "title_tokens": set(_tokens(title))})
    df = Counter()
    for c in chunks:
        df.update(set(c["tokens"]))
    n = max(len(chunks), 1)
    idf = {w: math.log(1 + n / (1 + d)) for w, d in df.items()}
    num_to_id = {}
    for c in chunks:
        num_to_id.setdefault(c["number"], c["id"])
    return {"chunks": chunks, "idf": idf, "num_to_id": num_to_id}


def search(index, question, k=4):
    q = _tokens(question)
    scored = []
    for c in index["chunks"]:
        s = sum(index["idf"].get(w, 0) * (1 + math.log(1 + c["tokens"][w])) for w in q if c["tokens"][w])
        s += sum(3 * index["idf"].get(w, 0) for w in q if w in c["title_tokens"])
        if s > 0:
            scored.append((s, c))
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:k]]


def _best_sentences(text, question, n=2):
    q = set(_tokens(question))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    ranked = sorted(sentences, key=lambda s: -len(q & set(_tokens(s))))
    picked = [s if len(s) <= 240 else s[:237].rsplit(" ", 1)[0] + "…" for s in ranked[:n] if q & set(_tokens(s))]
    out = " ".join(picked) or text[:220].rsplit(" ", 1)[0] + "…"
    return out if len(out) <= 320 else out[:317].rsplit(" ", 1)[0] + "…"


# ------------------------------------------------------------------ daily cap (shared by all users)
def _usage_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "mrgk_usage.json")


def usage_today():
    import json
    from datetime import date
    try:
        with open(_usage_path(), encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get(str(date.today()), 0))
    except (OSError, ValueError):
        return 0


def bump_usage():
    import json
    from datetime import date
    path = _usage_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = str(date.today())
    json.dump({today: usage_today() + 1}, open(path, "w", encoding="utf-8"))


# ------------------------------------------------------------------ API
def api_key(secrets):
    try:
        key = secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


SYSTEM = """You are Mr. GK, the friendly expert assistant built into Battery Clinic AI, a tool that predicts laptop battery health.
Answer questions about the project, the app, its data, its models and the user's latest prediction.

Rules:
- Use ONLY the reference answer, the guide extracts and the live app state provided below. If the answer is not there,
  say so plainly and suggest where in the guide or app to look. Never invent numbers, features or settings.
- Give thorough, well-structured answers, like a patient senior data scientist explaining to a curious colleague:
  start with **In short:** (one or two sentences), then the relevant parts of: **How it works**, **The calculation**
  (formula in backticks), **In your app right now** (live numbers from the app state), **What it means for you**.
  Use short bullet points. Skip parts that don't apply. Aim for 150-350 words; less for simple questions.
- When a REFERENCE ANSWER is provided, treat it as accurate and build on it; adapt it to the exact question asked.
- Cite the guide sections you used in square brackets with the section number exactly as given, e.g. [§9.4].
  These markers are turned into buttons and removed from the visible answer, so do not write section numbers anywhere else.
- Insights are associations in the data, not proven causes; say so when giving advice.
- Be warm and direct. You may refer to yourself as Mr. GK."""


def _context(chunks, state_text, reference=None):
    parts = [f"[§{c['number']}] {c['title']}\n{c['text'][:2200]}" for c in chunks]
    ref = f"REFERENCE ANSWER (accurate, from the built-in knowledge base)\n{reference}\n\n" if reference else ""
    return ref + "LIVE APP STATE\n" + state_text + "\n\nGUIDE EXTRACTS\n" + "\n\n".join(parts)


def ask_ai(key, question, history, chunks, state_text, reference=None):
    import requests
    messages = []
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": f"{_context(chunks, state_text, reference)}\n\nQUESTION: {question}"})
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": C.MRGK_MODEL, "max_tokens": C.MRGK_MAX_TOKENS, "system": SYSTEM,
                            "messages": messages}, timeout=45)
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def quick_facts(question, state):
    """Direct answers from the live app state for common factual questions."""
    q = " " + question.lower() + " "
    counting = any(w in q for w in ("how many", "which", "what are", "list", "number of", "available", "what all"))
    if counting and any(w in q for w in ("variable", "column", "feature", "attribute", "input", "field")):
        allin = list(C.FEATURES)
        used = state["inputs"]
        text = (f"The dataset has **{len(allin) + 1} variables**: **{len(allin)} inputs** and the target, **{C.TARGET}** (%).\n\n"
                f"Inputs: {', '.join(allin)}.\n\n"
                f"The model currently uses **{len(used)}** of them"
                + (f" ({C.OPTIONAL_FEATURE} is switched off)" if not state["include"] else "")
                + (f", plus {len(state['kept'])} calculated attribute(s): {', '.join(state['kept'])}" if state["kept"] else "")
                + f". {len(C.ENGINEERED)} calculated attributes ({', '.join(C.ENGINEERED)}) are also used in the insights.")
        return text
    if counting and "model" in q:
        top = "\n".join(f"- {line}" for line in state["leaderboard"][:5])
        return (f"**{len(state['leaderboard'])} models** are compared. The recommended one is **{state['recommended']}** "
                f"and the active one is **{state['model']}**. Top of the leaderboard (mean R² ± spread, MAE):\n{top}")
    if any(w in q for w in (" rows", "records", "how much data", "dataset size", "size of the data", "how many laptops")):
        return (f"The active data (version v{state['version']}) has **{state['rows']:,} rows** after cleaning, "
                f"one laptop per row.")
    return None


FACT_SECTIONS = {"variables": ["data", "engineered"], "models": ["p-models", "p-recommend"], "rows": ["p-load", "p-clean"]}


def fact_topic(question):
    q = " " + question.lower() + " "
    counting = any(w in q for w in ("how many", "which", "what are", "list", "number of", "available", "what all"))
    if counting and any(w in q for w in ("variable", "column", "feature", "attribute", "input", "field")):
        return "variables"
    if counting and "model" in q:
        return "models"
    if any(w in q for w in (" rows", "records", "how much data", "dataset size", "size of the data", "how many laptops")):
        return "rows"
    return None


def relevant(index, question, k=6):
    """Guide sections for a question: the matching reference sections first for factual questions, then search results."""
    found = search(index, question, k)
    topic = fact_topic(question)
    if not topic:
        return found
    by_id = {c["id"]: c for c in index["chunks"]}
    first = [by_id[i] for i in FACT_SECTIONS[topic] if i in by_id]
    return first + [c for c in found if c["id"] not in FACT_SECTIONS[topic]][: max(k - len(first), 0)]


def guide_answer(question, chunks, state):
    """Guide mode: an in-depth knowledge-base answer when the topic is known, otherwise fuller guide extracts."""
    import mrgk_kb as KB
    text, _, _ = KB.answer(question, state)
    if text:
        return text
    if not chunks:
        return ("I couldn't find that in the guide. Try different words — for example a page name, an attribute "
                "or a metric — or browse the guide's contents.")
    lines = ["**Here's what the guide says about this:**"]
    for c in chunks[:3]:
        lines.append(f"**{c['title']}**\n\n{_extract(c.get('prose') or c['text'], question)}")
    return "\n\n".join(lines)


def _extract(text, question, limit=650):
    """A readable passage: the best-matching sentence plus its neighbours, up to ~limit characters."""
    q = set(_tokens(question))
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text[:limit]
    best = max(range(len(sentences)), key=lambda i: len(q & set(_tokens(sentences[i]))))
    start = max(best - 1, 0)
    out = ""
    for s in sentences[start:]:
        if len(out) + len(s) > limit and out:
            break
        out += (" " if out else "") + s
    return out if len(out) <= limit + 80 else out[:limit].rsplit(" ", 1)[0] + "…"


def section_citations(index, ids):
    by_id = {c["id"]: c for c in index["chunks"]}
    out, seen = [], set()
    for i in ids:
        c = by_id.get(i)
        if c and c["id"] not in seen:
            seen.add(c["id"])
            out.append({"id": c["id"], "number": c["number"], "title": c["title"]})
    return out[:3]


def citations(text, index, fallback_chunks):
    nums = re.findall(r"\[§\s*([\d.]+)[^\]]*\]", text)
    seen, out = set(), []
    for n in nums:
        n = n.rstrip(".")
        if n in index["num_to_id"] and n not in seen:
            seen.add(n)
            out.append((n, index["num_to_id"][n]))
    if not out:
        for c in fallback_chunks[:3]:
            if c["number"] not in seen:
                seen.add(c["number"])
                out.append((c["number"], c["id"]))
    titles = {c["id"]: c["title"] for c in index["chunks"]}
    return [{"number": n, "id": i, "title": titles.get(i, "")} for n, i in out[:4]]


def _extra_state(state):
    out = []
    m = state.get("metrics")
    if m:
        out.append(f"Active model metrics (5 shuffles): mean R² {m['Mean R²']:.3f} ± {m['R² spread (±)']:.3f}, "
                   f"MAE {m['MAE']:.2f}, RMSE {m['RMSE']:.2f}, MAPE {m['MAPE %']:.2f}%, 90% within ±{m['90% error within (±)']:.2f}, "
                   f"overfit gap {m['Overfit gap']:.3f}.")
    if state.get("corr"):
        out.append("Correlations with health: " + ", ".join(f"{k} {v:+.2f}" for k, v in
                                                           sorted(state["corr"].items(), key=lambda kv: kv[1])) + ".")
    if state.get("train_rows"):
        out.append(f"Each shuffle: {state['train_rows']} training rows, {state['test_rows']} test rows.")
    return "\n".join(out)


def state_to_text(state):
    lines = [f"Data version: v{state['version']} ({state['rows']:,} rows). Target: {C.TARGET} (%).",
             f"Settings: split {state['split']}; {C.OPTIONAL_FEATURE} {'included' if state['include'] else 'excluded'}.",
             f"Dataset variables: {len(C.FEATURES) + 1} ({len(C.FEATURES)} inputs: {', '.join(C.FEATURES)}; target {C.TARGET}).",
             f"Inputs used by the model now: {', '.join(state['inputs'])}.",
             f"Calculated attributes kept by the model: {', '.join(state['kept']) or 'none'}.",
             f"Active model: {state['model']} (recommended: {state['recommended']}).",
             "Leaderboard (mean R² ± spread, MAE): " + "; ".join(state["leaderboard"])]
    p = state.get("prediction")
    if p:
        lines.append(f"User's latest prediction: {p['pred']:.1f}% ({p['band']}), likely range ±{p['mae']:.2f}.")
        lines.append("Inputs: " + ", ".join(f"{k}={v:g}" for k, v in p["row"].items()))
        lines.append(f"Typical laptop prediction {p['base']:.1f}%. Driver effects (points): "
                     + ", ".join(f"{d['Input']} {d['Effect']:+.2f}" for d in p["drivers"][:6]))
        if p["whatif"]:
            lines.append("What-if: " + "; ".join(f"{w['Input']} typical -> {w['New']:.1f}%" for w in p["whatif"]))
    else:
        lines.append("The user has not made a prediction yet.")
    extra = _extra_state(state)
    if extra:
        lines.append(extra)
    return "\n".join(lines)
