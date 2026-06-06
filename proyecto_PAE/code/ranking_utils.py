import re


_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "using", "with", "within", "will", "new", "project",
    "projects", "proposal", "proposals", "research", "study", "studies", "support",
    "supports", "include", "includes", "including", "development", "develop", "developing",
    "system", "systems", "method", "methods", "tool", "tools",
}

_TECH_PHRASES = [
    ("air traffic management", 4.5),
    ("atm", 4.0),
    ("egnos", 4.5),
    ("gnss", 4.5),
    ("gps", 4.0),
    ("sbas", 4.0),
    ("satellite navigation", 4.5),
    ("navigation systems", 3.5),
    ("navigation", 2.0),
    ("flight management", 3.0),
    ("trajectory management", 3.5),
    ("ifr", 3.5),
    ("instrument flight rules", 3.5),
    ("approach aids", 4.0),
    ("approach and landing", 4.5),
    ("gnss approaches", 4.5),
    ("gps approaches", 4.5),
    ("lpv", 3.5),
    ("rnav", 3.0),
    ("rnp", 3.0),
    ("autoland", 2.5),
    ("avionics", 2.0),
    ("augmentation", 2.0),
    ("tilt rotor", 3.0),
    ("rotorcraft", 2.5),
    ("helicopter", 2.0),
    ("guidance", 2.0),
    ("low noise", 3.0),
    ("noise", 2.0),
    ("helicopter sar", 2.5),
    ("search and rescue", 2.5),
    ("general aviation", 2.0),
    ("regional aviation", 2.0),
    ("corporate aviation", 2.0),
    ("school and training aviation", 2.0),
]


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def tokenize(text: str) -> set[str]:
    tokens = _TOKEN_RE.findall(normalize_text(text))
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
    }


def extract_project_signals(text: str, max_signals: int = 8) -> list[str]:
    normalized = normalize_text(text)
    signals = []
    seen = set()

    for phrase, _ in _TECH_PHRASES:
        if phrase in normalized and phrase not in seen:
            signals.append(phrase)
            seen.add(phrase)
        if len(signals) >= max_signals:
            break

    return signals


def build_project_query(text: str) -> str:
    text = str(text).strip()
    signals = extract_project_signals(text)

    parts = [text]
    if signals:
        parts.append("Detected technical cues: " + ", ".join(signals))

    return "\n\n".join(part for part in parts if part).strip()


def _row_text(row) -> str:
    return f"{row.get('category_code', '')} {row.get('category_label', '')} {row.get('text', '')}"


def score_taxonomy_row(project_text: str, row) -> float:
    query = normalize_text(project_text)
    query_tokens = tokenize(project_text)

    row_code = str(row.get("category_code", "")).strip().upper()
    row_label = str(row.get("category_label", ""))
    row_text = _row_text(row)
    row_norm = normalize_text(row_text)
    label_tokens = tokenize(row_label)
    row_tokens = tokenize(row_text)

    label_overlap = len(query_tokens & label_tokens)
    text_overlap = len(query_tokens & row_tokens)

    score = (3.0 * label_overlap) + (1.0 * text_overlap)

    for phrase, weight in _TECH_PHRASES:
        if phrase in query and phrase in row_norm:
            score += weight

    # Penalize broad umbrella codes slightly so more specific categories can surface.
    if row_code.endswith("0"):
        score -= 2.0

    propulsion_terms = ["propulsion", "engine", "engines", "turbine", "turbofan", "turbojet", "combustion", "nozzle", "air intake"]
    if any(term in row_norm for term in propulsion_terms) and not any(term in query for term in propulsion_terms):
        score -= 3.0

    ecs_terms = ["environmental control", "air conditioning", "cabin", "thermal management", "pressurization", "ventilation"]
    if any(term in row_norm for term in ecs_terms) and not any(term in query for term in ecs_terms):
        score -= 2.5

    compliance_terms = ["certification", "compliance", "regulation", "regulatory", "authority", "authorities"]
    if any(term in row_norm for term in compliance_terms) and not any(term in query for term in compliance_terms):
        score -= 3.0

    return score


def build_taxonomy_context_lines(project_text: str, taxonomy_df, max_candidates: int = 30) -> list[dict]:
    rows = []
    project_signals = extract_project_signals(project_text)

    for _, row in taxonomy_df.iterrows():
        row_score = score_taxonomy_row(project_text, row)
        row_text = str(row.get("text", "")).replace("\n", " ").strip()
        snippet = row_text[:220] + ("..." if len(row_text) > 220 else "")
        joined_row = normalize_text(f"{row.get('category_label', '')} {row.get('text', '')}")

        rows.append(
            {
                "category_code": str(row["category_code"]).strip().upper(),
                "category_label": str(row["category_label"]).strip(),
                "score": row_score,
                "snippet": snippet,
                "signals": [signal for signal in project_signals if signal in joined_row],
            }
        )

    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:max_candidates]


def _lookup_taxonomy_row(taxonomy_df, category_code: str):
    code = str(category_code).strip().upper()
    matches = taxonomy_df.loc[taxonomy_df["category_code"].astype(str).str.upper() == code]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_domain_guidance(taxonomy_df) -> str:
    priority_codes = ["1G5", "1D3", "1F4", "1G0", "1F18", "1F22"]
    lines = []

    for code in priority_codes:
        row = _lookup_taxonomy_row(taxonomy_df, code)
        if row is not None:
            lines.append(f"- {code}: {row['category_label']}")

    if not lines:
        return ""

    return (
        "High-priority navigation and operations families for EGNOS/GNSS-like projects:\n"
        + "\n".join(lines)
    )


def build_few_shot_navigation_example(taxonomy_df) -> str:
    nav = _lookup_taxonomy_row(taxonomy_df, "1G5")
    nav_fm = _lookup_taxonomy_row(taxonomy_df, "1D3")
    tests = _lookup_taxonomy_row(taxonomy_df, "1F4")
    atm = _lookup_taxonomy_row(taxonomy_df, "1G0")

    if nav is None or nav_fm is None or tests is None or atm is None:
        return ""

    return f"""
Example for an EGNOS/GNSS adoption project:
Project mentions EGNOS, GNSS, GPS approaches, LPV, validation, and operational adoption.
Expected tendency:
1. 1G5 - {nav['category_label']}
2. 1D3 - {nav_fm['category_label']}
3. 1F4 - {tests['category_label']}
4. 1G0 - {atm['category_label']}
5. 1F18 - Decision Support Systems
Avoid generic categories like aircraft health monitoring or propulsion unless they are explicitly central.
""".strip()


def _score_ranked_prediction(project_text: str, pred: dict, row, position: int, total: int) -> float:
    query = normalize_text(project_text)
    label = str(pred.get("category_label", "")).strip()
    reason = str(pred.get("reason", "")).strip()

    lexical = score_taxonomy_row(project_text, row) if row is not None else 0.0

    domain_bonus = 0.0
    row_text = normalize_text(f"{label} {reason} {row.get('text', '') if row is not None else ''}")

    nav_query = any(term in query for term in ["egnos", "gnss", "gps", "lpv", "rnav", "rnp", "approach", "landing"])
    if nav_query and any(term in row_text for term in ["navigation", "satellite navigation", "flight management", "autoland", "approach", "gnss", "gps"]):
        domain_bonus += 4.0

    ops_query = any(term in query for term in ["adoption", "validation", "operational", "trial", "pilot"])
    if ops_query and any(term in row_text for term in ["flight ground tests", "validation", "engineering", "operations", "air traffic management"]):
        domain_bonus += 1.5

    model_rank_bonus = (total - position) * 0.05
    return lexical + domain_bonus + model_rank_bonus


def rerank_predictions_with_domain_bias(project_text: str, ranked_predictions: list[dict], taxonomy_df) -> list[dict]:
    if not ranked_predictions:
        return ranked_predictions

    rows = {str(row["category_code"]).strip().upper(): row for _, row in taxonomy_df.iterrows()}

    scored = []
    total = len(ranked_predictions)

    for idx, pred in enumerate(ranked_predictions):
        code = str(pred.get("category_code", "")).strip().upper()
        row = rows.get(code)
        total_score = _score_ranked_prediction(project_text, pred, row, idx, total)
        scored.append((total_score, idx, pred))

    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [item[2] for item in scored]
