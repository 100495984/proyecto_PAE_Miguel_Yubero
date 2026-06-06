import json
import re

from ollama_client import RANKED_PREDICTIONS_SCHEMA, ollama_generate


MAX_PROJECT_CHARS = 3200
PROFILE_PROJECT_CHARS = 2600
FAMILY_SUMMARY_CHARS = 260
CATEGORY_SNIPPET_CHARS = 420
PROFILE_GEN_TOKENS = 220
EVAL_GEN_TOKENS = 160
REASON_GEN_TOKENS = 120
FINAL_GEN_TOKENS = 340
SHORTLIST_SIZE_MIN = 12
SHORTLIST_SIZE_MAX = 16
SHORTLIST_SNIPPET_CHARS = 180

WEAK_FIT_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "will", "have", "has", "had", "are", "was", "were",
    "been", "being", "their", "there", "about", "within", "without", "under", "over", "between", "during", "such",
    "than", "then", "them", "they", "these", "those", "into", "onto", "across", "through", "using", "used", "use",
    "new", "major", "output", "work", "works", "study", "studies", "project", "projects", "proposal", "field", "area",
    "european", "union", "research", "development", "transport", "sector", "sectors", "first", "second", "third",
    "particular", "different", "issues", "include", "includes", "including", "response", "responding", "topic", "topics",
    "call", "fp7", "fp6", "fp5", "none", "aviation", "technology", "which", "should", "necessary", "results",
    "generated", "identified", "developed", "another", "applicable", "conditions", "recognised", "each", "measure",
}

PROJECT_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_focus": {"type": "string"},
        "secondary_focus": {"type": "string"},
        "target_object": {"type": "string"},
        "hazard_context": {"type": "string"},
        "materials_or_methods": {
            "type": "array",
            "items": {"type": "string"},
        },
        "evidence_phrases": {
            "type": "array",
            "items": {"type": "string"},
        },
        "excluded_themes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "primary_focus",
        "secondary_focus",
        "target_object",
        "hazard_context",
        "materials_or_methods",
        "evidence_phrases",
        "excluded_themes",
    ],
    "additionalProperties": False,
}

FAMILY_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "family_code": {"type": "string"},
        "relevance": {"type": "integer"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["family_code", "relevance", "confidence", "reason"],
    "additionalProperties": False,
}

CATEGORY_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category_code": {"type": "string"},
        "relevance": {"type": "integer"},
        "confidence": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["category_code", "relevance", "confidence", "reason"],
    "additionalProperties": False,
}

CATEGORY_REASON_SCHEMA = {
    "type": "object",
    "properties": {
        "category_code": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["category_code", "reason"],
    "additionalProperties": False,
}


def extract_last_json_object(text: str):
    end = text.rfind("}")
    if end == -1:
        return ""

    depth = 0
    for index in range(end, -1, -1):
        if text[index] == "}":
            depth += 1
        elif text[index] == "{":
            depth -= 1
            if depth == 0:
                return text[index:end + 1]

    return ""


def _prepare_project_text(text: str, max_chars: int) -> str:
    cleaned = str(text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "\n\n[Truncated project description]"


def _family_budget(top_k: int) -> int:
    if top_k <= 1:
        return 1
    if top_k <= 3:
        return 2
    return 3


def _taxonomy_roots(taxonomy_df):
    roots = []
    tax = taxonomy_df.fillna("")

    root_rows = tax.loc[tax["parent_code"].astype(str).str.strip() == ""].sort_values("category_code")
    for _, root in root_rows.iterrows():
        root_code = str(root["category_code"]).strip().upper()
        children = tax.loc[
            tax["parent_code"].astype(str).str.strip().str.upper() == root_code
        ].sort_values("category_code")

        child_examples = [
            f"{str(row['category_code']).strip().upper()} {str(row['category_label']).strip()}"
            for _, row in children.head(10).iterrows()
        ]
        summary_text = str(root.get("text", "")).replace("\n", " ").strip()

        roots.append(
            {
                "family_code": root_code,
                "family_label": str(root["category_label"]).strip(),
                "summary": summary_text[:FAMILY_SUMMARY_CHARS] + ("..." if len(summary_text) > FAMILY_SUMMARY_CHARS else ""),
                "child_examples": child_examples,
            }
        )

    return roots


def _candidate_rows_from_families(taxonomy_df, selected_family_codes):
    tax = taxonomy_df.fillna("")
    family_lookup = {
        str(row["category_code"]).strip().upper(): row
        for _, row in tax.loc[tax["parent_code"].astype(str).str.strip() == ""].iterrows()
    }

    candidates = []
    seen = set()

    for family_code in selected_family_codes:
        family_code = str(family_code).strip().upper()
        family_row = family_lookup.get(family_code)
        if family_row is None:
            continue

        family_label = str(family_row["category_label"]).strip()
        child_rows = tax.loc[
            tax["parent_code"].astype(str).str.strip().str.upper() == family_code
        ].sort_values("category_code")

        for _, row in child_rows.iterrows():
            code = str(row["category_code"]).strip().upper()
            if code in seen:
                continue

            row_text = str(row.get("text", "")).replace("\n", " ").strip()
            candidates.append(
                {
                    "category_code": code,
                    "category_label": str(row["category_label"]).strip(),
                    "family_code": family_code,
                    "family_label": family_label,
                    "snippet": row_text[:CATEGORY_SNIPPET_CHARS] + ("..." if len(row_text) > CATEGORY_SNIPPET_CHARS else ""),
                }
            )
            seen.add(code)

    return candidates


def _build_project_profile_prompt(project_text: str) -> str:
    return f"""
You are preparing a project profile for aerospace taxonomy classification.

Task:
Summarize the real technical focus of the project.

Rules:
- Focus on what is being designed, manufactured, analysed, validated, or integrated.
- Distinguish the source of a hazard from the technical target of the work.
- If the project protects aircraft parts from engine burst, shrapnel, impact, or heat, do not call propulsion the primary focus unless the engine itself is the design target.
- Keep each field short and concrete.
- `evidence_phrases` must contain short phrases copied or closely paraphrased from the project text.
- `excluded_themes` should name domains that appear only as context or that should not dominate classification.
- Return ONLY JSON.

JSON format:
{{
  "primary_focus": "Main technical focus",
  "secondary_focus": "Secondary technical focus or 'none'",
  "target_object": "What is being designed/protected/tested",
  "hazard_context": "Threat or context if present, otherwise 'none'",
  "materials_or_methods": ["item1", "item2"],
  "evidence_phrases": ["phrase1", "phrase2", "phrase3"],
  "excluded_themes": ["theme1", "theme2"]
}}

Project description:
{project_text}
"""


def _profile_block(project_profile) -> str:
    return "\n".join(
        [
            f"Primary focus: {project_profile['primary_focus']}",
            f"Secondary focus: {project_profile['secondary_focus']}",
            f"Target object: {project_profile['target_object']}",
            f"Hazard/context: {project_profile['hazard_context']}",
            f"Materials or methods: {', '.join(project_profile['materials_or_methods']) or 'none'}",
            f"Evidence phrases: {', '.join(project_profile['evidence_phrases']) or 'none'}",
            f"Excluded themes: {', '.join(project_profile['excluded_themes']) or 'none'}",
        ]
    )


def _family_details_block(family_row) -> str:
    examples = "; ".join(family_row["child_examples"]) if family_row["child_examples"] else "No child categories listed."
    return (
        f"Family code: {family_row['family_code']}\n"
        f"Family label: {family_row['family_label']}\n"
        f"Family summary: {family_row['summary']}\n"
        f"Sample child categories: {examples}"
    )


def _category_details_block(category_row) -> str:
    return (
        f"Category code: {category_row['category_code']}\n"
        f"Category label: {category_row['category_label']}\n"
        f"Parent family: {category_row['family_code']} - {category_row['family_label']}\n"
        f"Category summary: {category_row['snippet']}"
    )


def _build_family_eval_prompt(project_profile, family_row) -> str:
    return f"""
You are evaluating one taxonomy family for an aerospace research project.

Task:
Score how relevant this family is to the project's real technical focus.

Scale:
- 3 = core family
- 2 = clearly relevant secondary family
- 1 = weak/contextual mention
- 0 = not relevant

Rules:
- Judge ONLY this family.
- Use the project focus, not just surface keywords.
- If a domain appears only as the source of a hazard or operating context, score it 0 or 1.
- Innovation alone is not enough. Do not score umbrella families such as future concepts, breakthrough technologies, or generic new materials above 1 when a more specific technical family explains the project better.
- Potential future applications do not count as evidence.
- The reason must be concrete and project-specific.
- Return ONLY JSON.

JSON format:
{{
  "family_code": "{family_row['family_code']}",
  "relevance": 0,
  "confidence": 0,
  "reason": "Short concrete explanation"
}}

Project profile:
{_profile_block(project_profile)}

Family under review:
{_family_details_block(family_row)}
"""


def _build_category_eval_prompt(project_profile, family_evaluation, category_row) -> str:
    return f"""
You are evaluating one taxonomy category for an aerospace research project.

Task:
Score how well this category matches the project's technical work.

Scale:
- 3 = direct core match
- 2 = strong supporting match
- 1 = plausible but secondary
- 0 = mismatch

Rules:
- Judge ONLY this category.
- Prefer the technical target of the work over the source of the hazard.
- If the category mainly concerns propulsion, acoustics, human factors, or another side domain that is not being designed or studied directly, score it 0 or 1.
- If the project is about shields, fibres, composites, structural integration, impact resistance, or manufacturing of aircraft parts, structural/materials/manufacturing categories usually fit better than propulsion categories.
- Potential future enhancements, speculative benefits, or hypothetical applications do not count. If the category fits only because the project could maybe evolve in that direction, score it 0.
- Do not use umbrella innovation/new-materials categories when a more specific structures/materials/manufacturing category already captures the same evidence.
- Smart materials categories require explicit sensing, actuation, adaptability, or embedded intelligence in the project text.
- Noise or acoustics categories require explicit acoustic, sound, noise, vibration, or measurement objectives.
- Testing or validation categories require explicit testing, measurements, characterization, experiments, or validation work.
- Security categories require explicit aircraft/passenger/crew security measures, not just protection from physical damage.
- The reason must cite concrete project evidence.
- Return ONLY JSON.

JSON format:
{{
  "category_code": "{category_row['category_code']}",
  "relevance": 0,
  "confidence": 0,
  "reason": "Short concrete explanation"
}}

Project profile:
{_profile_block(project_profile)}

Selected family context:
Family {family_evaluation['family_code']} scored {family_evaluation['relevance']}/3 because: {family_evaluation['reason']}

Category under review:
{_category_details_block(category_row)}
"""


def _build_reason_prompt(project_profile, category_row, category_evaluation) -> str:
    fit_level = "core" if category_evaluation["relevance"] == 3 else "secondary"
    return f"""
You are writing the final justification for one selected taxonomy category.

Task:
Explain in one precise sentence why this category fits the project.

Rules:
- Mention concrete project details.
- Avoid generic phrases such as "falls under" or "belongs to".
- Do not invent testing, sensing, acoustics, or other activities that are not explicit in the project text.
- If the fit is secondary rather than core, make that explicit in the sentence.
- Do not say that the project performs testing, validation, or advanced manufacturing unless the project text says so directly. If the category is only a secondary methodological fit, say it supports or aligns with the work instead.
- Return ONLY JSON.

JSON format:
{{
  "category_code": "{category_row['category_code']}",
  "reason": "Precise one-sentence justification"
}}

Project profile:
{_profile_block(project_profile)}

Category evaluation summary:
Fit level: {fit_level}
Relevance: {category_evaluation['relevance']}/3
Confidence: {category_evaluation['confidence']}/100
Draft reason: {category_evaluation['reason']}

Selected category:
{_category_details_block(category_row)}
"""


def _run_llm_json(prompt: str, model_name: str, max_new_tokens: int, response_schema):
    try:
        decoded = ollama_generate(
            prompt,
            model_name,
            num_predict=max_new_tokens,
            response_format=response_schema,
        )
    except RuntimeError as err:
        if "Ollama" in str(err):
            return None, ""
        raise

    clean_json = extract_last_json_object(decoded) or str(decoded).strip()
    if not clean_json:
        return None, clean_json

    try:
        return json.loads(clean_json), clean_json
    except Exception:
        return None, clean_json


def _clean_list_strings(values, max_items: int):
    cleaned = []
    seen = set()
    if not isinstance(values, list):
        return cleaned

    for item in values:
        text = str(item).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        cleaned.append(text)
        seen.add(key)
        if len(cleaned) == max_items:
            break

    return cleaned


def _coerce_int(value, minimum: int, maximum: int, default: int):
    try:
        number = int(value)
    except Exception:
        return default
    return max(minimum, min(maximum, number))


def _parse_project_profile(data):
    if not isinstance(data, dict):
        return None

    primary_focus = str(data.get("primary_focus", "")).strip()
    if not primary_focus:
        return None

    return {
        "primary_focus": primary_focus,
        "secondary_focus": str(data.get("secondary_focus", "")).strip() or "none",
        "target_object": str(data.get("target_object", "")).strip() or "none",
        "hazard_context": str(data.get("hazard_context", "")).strip() or "none",
        "materials_or_methods": _clean_list_strings(data.get("materials_or_methods"), 6),
        "evidence_phrases": _clean_list_strings(data.get("evidence_phrases"), 6),
        "excluded_themes": _clean_list_strings(data.get("excluded_themes"), 4),
    }


def _fallback_project_profile(project_text: str):
    short_text = project_text.replace("\n", " ").strip()
    return {
        "primary_focus": short_text[:180] + ("..." if len(short_text) > 180 else ""),
        "secondary_focus": "none",
        "target_object": "unspecified aircraft-related system or component",
        "hazard_context": "none",
        "materials_or_methods": [],
        "evidence_phrases": [],
        "excluded_themes": [],
    }


def _normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _contains_any(text: str, patterns) -> bool:
    return any(pattern in text for pattern in patterns)


def _contains_any_phrase(text: str, phrases) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) for phrase in phrases)


def _contains_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _build_positive_project_text(project_text: str, project_profile) -> str:
    return " ".join(
        [
            _normalize_text(project_text),
            _normalize_text(project_profile.get("primary_focus", "")),
            _normalize_text(project_profile.get("secondary_focus", "")),
            _normalize_text(project_profile.get("target_object", "")),
            _normalize_text(project_profile.get("hazard_context", "")),
            _normalize_text(" ".join(project_profile.get("materials_or_methods", []))),
            _normalize_text(" ".join(project_profile.get("evidence_phrases", []))),
        ]
    ).strip()


def _build_project_signals(project_text: str, project_profile):
    positive_text = _build_positive_project_text(project_text, project_profile)

    wind_tunnel = _contains_any(
        positive_text,
        [
            "wind tunnel",
            "wind-tunnel",
            "pressurized low speed wind tunnel",
            "pressurised low speed wind tunnel",
            "wt test campaign",
        ],
    )
    aerodynamic_terms = _contains_any(
        positive_text,
        [
            "aerodynamic",
            "aerodynamics",
            "airflow",
            "laminar wing",
            "lift",
            "drag",
            "boundary layer",
            "slat",
            "flap",
            "empennage",
            "fairing",
            "horizontal tail plane",
            "htp",
            "handling qualities",
            "landing performance",
        ],
    )
    acoustic_terms = _contains_any(
        positive_text,
        [
            " acoustic",
            " acoustics",
            " aeroacoustic",
            " aeroacoustics",
            " sound",
            " noise",
        ],
    )
    physical_protection_context = _contains_any(
        positive_text,
        [
            "shrapnel",
            "ballistic",
            "engine burst",
            "debris",
            "impact resistance",
            "energy absorption",
            "energy-absorbing",
            "damage tolerance",
            "protect critical components",
            "protect aircraft components",
        ],
    )
    noise_shielding_context = _contains_any(
        positive_text,
        [
            "shielding engine noise",
            "shield engine noise",
            "noise shielding",
            "shielding noise",
        ],
    )
    internal_noise_context = _contains_any(
        positive_text,
        [
            "internal noise",
            "cabin noise",
            "cockpit noise",
            "interior noise",
            "passenger comfort",
            "cabin acoustic",
        ],
    )
    external_noise_context = acoustic_terms and (
        _contains_any(
            positive_text,
            [
                "engine noise",
                "external noise",
                "acoustic data",
                "acoustic testing",
                "aerodynamic and acoustic",
            ],
        )
        or wind_tunnel
        or aerodynamic_terms
    )
    instrumentation = _contains_any(
        positive_text,
        [
            "probe",
            "probes",
            "pressure instrumentation",
            "instrumentation",
            "kulite",
            "kulites",
            "gauge",
            "gauges",
            "sensor",
            "sensors",
        ],
    )
    fem_structural = _contains_any(
        positive_text,
        [
            "finite element",
            "structural verification",
            "representative loads",
            "fuselage modification",
            "fuselage modifications",
        ],
    ) or _contains_any_phrase(
        positive_text,
        ["fem", "loads", "strain", "stress", "fatigue"],
    )
    aeroelastic_explicit = _contains_any(
        positive_text,
        [
            "aeroelastic",
            "flutter",
            "gust response",
            "structural deformation in flow",
            "flexible structures in flowing fluid",
        ],
    )
    vibration_explicit = _contains_any(
        positive_text,
        [
            "vibration",
            "vibrations",
            "buckling",
            "modal",
            "dynamic response",
        ],
    )
    cfd_explicit = _contains_any(
        positive_text,
        [
            "computational fluid dynamics",
            "cfd",
            "flow solver",
            "fluid dynamics simulation",
            "aerodynamic simulation",
            "aero solver",
        ],
    )
    unsteady_explicit = _contains_any(
        positive_text,
        [
            "unsteady",
            "transient",
            "dynamic response",
            "dynamic stall",
            "steady and unsteady",
            "handling qualities",
        ],
    )
    high_lift = _contains_any(
        positive_text,
        [
            "high lift",
            "leading edge slat",
            "trailing edge flap",
            "slat",
            "slats",
            "flap",
            "flaps",
            "airbrake",
            "airbrakes",
        ],
    )
    wing_design_terms = _contains_any(
        positive_text,
        [
            "laminar wing",
            "empennage",
            "fairing",
            "horizontal tail plane",
            "htp",
            "nacelle",
            "pylon",
            "fuselage",
            "wing-body fairing",
            "body-wing fairing",
            "leading edge",
            "trailing edge",
        ],
    )
    egnos_gnss = _contains_any(
        positive_text,
        [
            "egnos",
            "gnss",
            "gps",
            "sbas",
            "lpv",
            "rnav",
            "rnp",
            "apv approach",
            "navigation concept",
        ],
    )
    flight_procedures = _contains_any(
        positive_text,
        [
            "procedure",
            "procedures",
            "approach procedures",
            "curved procedures",
            "flight profile",
            "flight profiles",
            "ifr",
            "approach",
            "routes",
            "route",
            "operations",
            "operational deployment",
        ],
    )
    certification_work = _contains_any(
        positive_text,
        [
            "certification",
            "standards",
            "safety work",
            "regulatory",
            "regulator",
            "operational practices",
        ],
    )
    explicit_training = _contains_any(
        positive_text,
        [
            "training",
            "selection",
            "school and training aviation",
            "training aviation",
        ],
    )
    aerospace_domain = _contains_any(
        positive_text,
        [
            "aircraft",
            "aviation",
            "aeronautic",
            "air transport",
            "rotorcraft",
            "helicopter",
            "tilt rotor",
            "airport",
            "atm",
            "egnos",
            "gnss",
            "flight",
            "fuselage",
            "wing",
            "nacelle",
            "empennage",
        ],
    )
    out_of_domain_industry = _contains_any(
        positive_text,
        [
            "offshore wind",
            "wind farm",
            "wind farms",
            "wind turbine",
            "wind turbines",
            "renewable energy",
            "non-destructive evaluation",
        ],
    )
    bench_ground_test = _contains_any(
        positive_text,
        [
            "ground test",
            "ground tests",
            "flight test",
            "flight tests",
            "test bench",
            "test benches",
            "bench testing",
            "bench test",
            "test readiness review",
            "test rig",
            "test rigs",
            "icing test",
            "icing tests",
            "wt campaign",
            "operational testing",
            "campaigning",
        ],
    )
    systems_simulation_core = _contains_any(
        positive_text,
        [
            "software tool",
            "simulation environment",
            "shared simulation environment",
            "numerical model",
            "numerical models",
            "modelica",
            "fast time simulation",
            "energy management system",
            "electrical power",
            "thermal energy",
            "electro-thermal",
            "electro thermal",
            "data acquisition",
            "control command",
            "acquisition and control",
            "power network",
            "power networks",
            "knowledge-based engineering",
            "avionics architecture",
            "on-board systems",
            "on board systems",
            "electrical systems for aeronautics",
        ],
    )
    study_assessment_core = _contains_any(
        positive_text,
        [
            "requirements study",
            "trade-off",
            "trade off",
            "market survey",
            "benchmark",
            "assessment study",
            "requirements associated",
            "define the most optimum",
            "analysis, assessment",
            "analyse these requirements",
            "analyze these requirements",
            "swot analysis",
        ],
    )
    process_tooling_manufacturing = _contains_any(
        positive_text,
        [
            "tooling",
            "assembly device",
            "assembly devices",
            "metrology",
            "tolerance",
            "tolerances",
            "dimensional stability",
            "moulding",
            "molding",
            "metal injection moulding",
            "metal injection molding",
            "mim process",
            "milling",
            "turning",
            "industrialisation",
            "industrialization",
            "quality controls",
            "prototype",
            "prototypes",
            "manufacturing cost",
            "assembly of components",
            "assembly hardware",
        ],
    )
    demonstrator_platform = _contains_any(
        positive_text,
        [
            "demonstrator",
            "demonstrators",
            "feature demonstrators",
            "validation platform",
            "shared simulation environment",
            "test platform",
            "test platforms",
        ],
    )
    electrical_systems_core = _contains_any(
        positive_text,
        [
            "electrical system",
            "electrical systems",
            "electrical power",
            "power distribution",
            "power networks",
            "energy management system",
            "electro-thermal",
            "electro thermal",
            "on-board systems",
            "on board systems",
            "avionics architecture",
            "data acquisition",
            "control command",
        ],
    )
    fuel_systems_core = _contains_any(
        positive_text,
        [
            "fuel system",
            "fuel systems",
            "fuel tank",
            "fuel line",
            "fuel pump",
            "fuel circuit",
            "fuel flow",
        ],
    )
    metallic_process = _contains_any(
        positive_text,
        [
            "metal injection moulding",
            "metal injection molding",
            "mim process",
            "milling and turning",
            "metallic materials",
            "metal parts",
            "aluminium",
            "titanium",
            "alloy",
        ],
    ) and (process_tooling_manufacturing or _contains_any(positive_text, ["manufacturing", "prototype", "prototypes", "testing"]))

    signals = {
        "design_work": _contains_any_phrase(positive_text, ["design", "designing", "designed", "conception", "integrate", "integrated", "integration"]),
        "manufacture_explicit": _contains_any_phrase(positive_text, ["manufacture", "manufacturing", "fabrication", "fabricate", "fabricated", "produced", "production"]),
        "manufacturing_process": _contains_any(
            positive_text,
            [
                "manufacturing process",
                "production engineering",
                "factory",
                "robotics",
                "machining",
                "welding",
                "tooling",
                "fabrication simulation",
                "process route",
                "process routes",
                "tooling",
                "model assembly",
                "surface refinement",
                "moulding",
                "molding",
                "metal injection moulding",
                "metal injection molding",
                "metrology",
                "milling",
                "turning",
                "dimensional stability",
                "tolerances",
                "quality controls",
                "industrialisation",
                "industrialization",
            ],
        ) or process_tooling_manufacturing,
        "testing_validation": _contains_any(
            positive_text,
            [
                " test",
                "testing",
                "experimental",
                "experiment",
                "validation",
                "validate",
                "verification",
                "characterization",
                "characterisation",
                "measurement",
                "measurements",
                "measured",
                "tested",
                "test campaign",
            ],
        ) or wind_tunnel or instrumentation,
        "acoustics": acoustic_terms,
        "smart": _contains_any(
            positive_text,
            ["smart material", "smart structure", "sensor", "sensors", "adaptive", "self-sensing", "embedded intelligence"],
        ),
        "security": _contains_any(positive_text, [" security", " passenger", " crew", " cockpit", "bomb-proof", "bomb proof", "barrier device"]),
        "propulsion_core": _contains_any(
            positive_text,
            [" propulsion", " turbomach", " compressor", " combustor", " turbine", " nozzle", " thrust", " engine performance", " combustion"],
        ) and not _contains_any(positive_text, ["engine burst", "burst protection", "shrapnel from engine", "wind turbine", "wind turbines", "offshore wind", "wind farm"]),
        "flight_dynamics": _contains_any(
            positive_text,
            [
                "flight mechanics",
                " aircraft performance",
                " performance analysis",
                "stability and control",
                "dynamic stability",
                " controllable",
                " controllability",
                " control system",
                " trajectory",
                " go-around",
                " go around",
                " flight envelope",
                " range and endurance",
            ],
        ),
        "failure_analysis": _contains_any(
            positive_text,
            [
                "failure analysis",
                "damage analysis",
                "system failure",
                "hazard analysis",
                "environmental hazard",
                "accident",
                "incident",
                "bird strike",
            ],
        ),
        "digital_process": _contains_any(
            positive_text,
            [
                "it tool",
                "collaborative",
                "virtual reality",
                "lifecycle",
                "life-cycle",
                "industry 4.0",
                "digital twin",
                "digital thread",
                "simulator environment",
            ],
        ) or systems_simulation_core,
        "innovation_concepts": _contains_any(
            positive_text,
            [
                "unconventional",
                "new aircraft concept",
                "hybrid electric",
                "electric propulsion",
                "scenario analysis",
                "scenarios analysis",
            ],
        ),
        "aerodynamics_physics": aerodynamic_terms or wind_tunnel,
        "avionics_systems": _contains_any(
            positive_text,
            ["avionics", "electrical system", "communications bus", "sensor suite", "power distribution", "cockpit display", "on-board equipment"],
        ) or electrical_systems_core,
        "atm": _contains_any(positive_text, ["air traffic", "navigation systems", "trajectory management", "airspace management"])
        or _contains_any_phrase(positive_text, ["atm", "gnss", "egnos", "rnav", "rnp", "lpv"]),
        "airport": _contains_any(positive_text, ["airport", "runway", "terminal", "ground handling"]),
        "human": _contains_any(
            positive_text,
            [
                "human factors",
                "pilot workload",
                "controller workload",
                "crew workload",
                "training",
                "selection",
                "man-machine",
                "human-machine",
                "survivability",
            ],
        ),
        "uas": _contains_any(positive_text, ["unmanned", "uas", "drone", "autonomous flight"]),
        "metallic_materials": _contains_any(positive_text, ["metallic", "metal parts", "alloy", "aluminium", "titanium"]),
        "nonmetal_materials": _contains_any(
            positive_text,
            ["fibre", "fiber", "fibres", "fibers", "membrane", "non-metal", "non metallic", "ceramic", "organic"],
        ),
        "composite_materials": _contains_any(positive_text, ["composite", "composites", "fibre matrix", "fiber matrix"]),
        "shield_protection": physical_protection_context
        or (
            _contains_any_phrase(positive_text, ["shield", "protective shield", "protection"])
            and not noise_shielding_context
            and _contains_any(positive_text, ["impact", "burst", "debris", "damage", "ballistic", "shrapnel", "load"])
        ),
        "structural_terms": _contains_any(
            positive_text,
            ["aircraft integration", "integration constraints", "fuselage", "verification", "fem", "pylon", "empennage"]
        ) or _contains_any_phrase(
            positive_text,
            ["structural", "structure", "stress", "fatigue", "loads", "load case", "load cases"],
        ) or _contains_regex(positive_text, r"(?<!work)\bload\b"),
        "meta_methodology": _contains_any(
            positive_text,
            [
                "methodology for impact assessment",
                "impact assessment",
                "evaluation of the impact",
                "framework programmes",
                "framework programmes'",
                "mefisto methodology",
                "leverage effect",
                "driving effect",
                "structuring effect",
                "process methodology",
            ],
        ),
        "wind_tunnel": wind_tunnel,
        "cfd_explicit": cfd_explicit,
        "unsteady_explicit": unsteady_explicit,
        "high_lift": high_lift,
        "wing_design_terms": wing_design_terms,
        "external_noise_context": external_noise_context,
        "internal_noise_context": internal_noise_context,
        "instrumentation": instrumentation,
        "fem_structural": fem_structural,
        "aeroelastic_explicit": aeroelastic_explicit,
        "vibration_explicit": vibration_explicit,
        "egnos_gnss": egnos_gnss,
        "flight_procedures": flight_procedures,
        "certification_work": certification_work,
        "explicit_training": explicit_training,
        "aerospace_domain": aerospace_domain,
        "out_of_domain_industry": out_of_domain_industry,
        "bench_ground_test": bench_ground_test,
        "systems_simulation_core": systems_simulation_core,
        "study_assessment_core": study_assessment_core,
        "process_tooling_manufacturing": process_tooling_manufacturing,
        "demonstrator_platform": demonstrator_platform,
        "electrical_systems_core": electrical_systems_core,
        "fuel_systems_core": fuel_systems_core,
        "metallic_process": metallic_process,
    }
    signals["physical_aero_project"] = any(
        [
            wind_tunnel,
            cfd_explicit,
            unsteady_explicit,
            high_lift,
            wing_design_terms,
            instrumentation,
            fem_structural,
            aeroelastic_explicit,
            vibration_explicit,
        ]
    )
    signals["atm_navigation_core"] = signals["atm"] and (egnos_gnss or flight_procedures)
    signals["structures_materials_core"] = (
        signals["shield_protection"] or signals["nonmetal_materials"] or signals["composite_materials"] or signals["structural_terms"] or signals["metallic_process"]
    ) and (signals["design_work"] or signals["manufacture_explicit"])
    signals["strong_technical_topic"] = any(
        [
            signals["structures_materials_core"],
            signals["acoustics"],
            signals["propulsion_core"],
            signals["flight_dynamics"],
            signals["failure_analysis"],
            signals["digital_process"],
            signals["innovation_concepts"],
            signals["aerodynamics_physics"],
            signals["avionics_systems"],
            signals["atm"],
            signals["airport"],
            signals["human"],
            signals["uas"],
            signals["wind_tunnel"],
            signals["fem_structural"],
            signals["bench_ground_test"],
            signals["systems_simulation_core"],
            signals["process_tooling_manufacturing"],
            signals["electrical_systems_core"],
        ]
    )
    signals["out_of_domain_project"] = out_of_domain_industry
    signals["meta_nontechnical_project"] = (signals["meta_methodology"] and not signals["strong_technical_topic"]) or signals["out_of_domain_project"]
    return signals


def _force_relevance(evaluation, relevance: int, reason: str, confidence: int = 100):
    updated = dict(evaluation)
    updated["relevance"] = relevance
    updated["confidence"] = max(updated.get("confidence", 0), confidence)
    updated["reason"] = reason
    return updated


def _category_tags(category_row):
    label_text = _normalize_text(category_row["category_label"])
    summary_text = _normalize_text(category_row["snippet"])
    text = f"{label_text} {summary_text}"
    tags = set()

    if _contains_any(label_text, ["noise", "acoustic", "acoustics", "accoustic", "accoustics", "sound"]):
        tags.add("acoustics")
    if _contains_any(label_text, ["smart materials", "smart structures"]) or _contains_any(summary_text, ["sensor", "actuator", "adaptive"]):
        tags.add("smart")
    if _contains_any(label_text, ["security"]) or _contains_any(summary_text, ["cockpit", "bomb-proof", "bomb proof"]):
        tags.add("security")
    if _contains_any(label_text, ["test", "testing", "measurement", "validation", "experimental"]):
        tags.add("testing")
    if _contains_any(label_text, ["test bench", "bench", "flight/ ground tests", "flight/ground tests"]):
        tags.add("bench_testing")
    if _contains_any(label_text, ["manufacturing", "fabrication"]) or _contains_any(summary_text, ["machining", "welding", "factory", "production engineering", "fabrication simulation"]):
        tags.add("manufacturing")
    if _contains_any(label_text, ["methodology", "operational research", "methods", "tools"]):
        tags.add("generic_methods")
    if _contains_any(label_text, ["performance analysis", "stability", "control", "trajectory"]) or _contains_any(summary_text, ["flight mechanics", "flight envelope"]):
        tags.add("flight_dynamics")
    if _contains_any(label_text, ["system failure", "damage analysis"]) or _contains_any(summary_text, ["accident", "incident", "bird strike"]):
        tags.add("failure_analysis")
    if "environmental hazard" in label_text:
        tags.add("hazard_analysis")
    if _contains_any(label_text, ["collaborative product", "it tools", "life-cycle", "lifecycle", "industry 4.0", "digital"]):
        tags.add("digital_process")
    if _contains_any(label_text, ["numerical models", "simulation", "virtual reality", "reference data"]):
        tags.add("simulation_models")
    if _contains_any(label_text, ["breakthrough technologies", "new materials", "innovative concepts", "scenarios analysis", "unconventional configurations"]):
        tags.add("innovation_umbrella")
    if _contains_any(label_text, ["propulsion", "turbomach", "compressor", "combustor", "turbine", "air-breathing"]):
        tags.add("propulsion")
    if _contains_any(label_text, ["human", "crew", "training", "survivability"]):
        tags.add("human")
    if _contains_any(label_text, ["electrical", "electronics", "avionics", "on-board", "on board"]):
        tags.add("systems_electrical")
    if _contains_any(label_text, ["fuel systems"]):
        tags.add("fuel_systems")
    if "airport" in label_text:
        tags.add("airport")
    if _contains_any(label_text, ["air traffic", "atm", "navigation"]):
        tags.add("atm")
    if _contains_any(label_text, ["unmanned", "uas", "drone", "autonomous"]):
        tags.add("uas")
    if ("non-metallic" not in label_text and _contains_any(label_text, ["metalic materials", "metallic materials"])) or _contains_any(summary_text, ["metal parts"]):
        tags.add("metallic")
    if _contains_any(label_text, ["non-metallic"]) or _contains_any(summary_text, ["organic", "ceramic", "fibre", "fiber", "membrane"]):
        tags.add("nonmetal")
    if "composite" in label_text:
        tags.add("composite")
    if _contains_any(label_text, ["structural analysis and design"]) or _contains_any(summary_text, ["stress analysis", "fatigue", "structural part", "static loads"]):
        tags.add("structural_design")
    if _contains_any(label_text, ["large scale validation experiments", "large scale validation platforms"]):
        tags.add("large_validation")
    if _contains_any(label_text, ["general purpose equipment"]):
        tags.add("general_equipment")

    return tags


def _category_label_text(category_row) -> str:
    return _normalize_text(category_row["category_label"])


def _category_search_text(category_row) -> str:
    return " ".join(
        [
            _category_label_text(category_row),
            _normalize_text(category_row.get("snippet", "")),
            _normalize_text(category_row.get("family_label", "")),
        ]
    ).strip()


def _category_label_has(category_row, patterns) -> bool:
    return _contains_any(_category_label_text(category_row), patterns)


def _category_text_has(category_row, patterns) -> bool:
    return _contains_any(_category_search_text(category_row), patterns)


def _apply_family_domain_rules(family_evaluation, project_signals):
    code = family_evaluation["family_code"]
    adjusted = dict(family_evaluation)

    if project_signals["meta_nontechnical_project"]:
        return _force_relevance(
            adjusted,
            0,
            "Domain rule: the project is a programme-impact/methodology assessment without a direct technical aerospace topic, so no taxonomy family is a strong match.",
        )

    if code == "1B0" and project_signals["structures_materials_core"]:
        return _force_relevance(
            adjusted,
            max(adjusted["relevance"], 2),
            "Domain rule: the project explicitly combines shield/protection, fibres or composites, and design/manufacture work, which strongly matches Aerostructures and materials.",
            confidence=95,
        )

    if code == "1C0" and not project_signals["propulsion_core"]:
        return _force_relevance(adjusted, 0, "Domain rule: propulsion appears only as hazard context, not as the technical target of the project.")
    if code == "1E0" and not project_signals["flight_dynamics"] and not project_signals["failure_analysis"]:
        return _force_relevance(adjusted, 0, "Domain rule: there is no explicit flight-dynamics, performance, controllability, or failure-analysis work in the project text.")
    if code == "1F0" and not project_signals["digital_process"]:
        return _force_relevance(adjusted, 0, "Domain rule: the project text does not mention collaborative IT tools, lifecycle integration, or digital engineering methods.")
    if code == "1J0" and not project_signals["innovation_concepts"]:
        return _force_relevance(adjusted, 0, "Domain rule: generic innovation is not enough; there is no explicit concepts/scenarios evidence beyond the core technical work.")
    if code == "1L0" and not project_signals["digital_process"]:
        return _force_relevance(adjusted, 0, "Domain rule: Industry 4.0 or digital-industry themes are not explicit in the project text.")
    if code == "1A0" and not project_signals["aerodynamics_physics"]:
        return _force_relevance(adjusted, 0, "Domain rule: there is no explicit aerodynamics or flight-physics work in the project text.")
    if code == "1D0" and not project_signals["avionics_systems"]:
        return _force_relevance(adjusted, 0, "Domain rule: avionics or equipment-system development is not explicit in the project text.")
    if code == "1G0" and not project_signals["atm"]:
        return _force_relevance(adjusted, 0, "Domain rule: air-traffic-management themes are not explicit in the project text.")
    if code == "1H0" and not project_signals["airport"]:
        return _force_relevance(adjusted, 0, "Domain rule: airport-related themes are not explicit in the project text.")
    if code == "1I0" and not project_signals["human"]:
        return _force_relevance(adjusted, 0, "Domain rule: human-factors or crew/passenger themes are not explicit in the project text.")
    if code == "1K0" and not project_signals["uas"]:
        return _force_relevance(adjusted, 0, "Domain rule: unmanned-aerial-system themes are not explicit in the project text.")

    return adjusted


def _rule_out_category(category_row, reason: str):
    return {
        "category_code": category_row["category_code"],
        "relevance": 0,
        "confidence": 100,
        "reason": reason,
    }


def _precheck_category_domain_rules(category_row, project_signals):
    tags = _category_tags(category_row)
    code = category_row["category_code"]

    if category_row["family_code"] == "1L0" and not project_signals["digital_process"]:
        return _rule_out_category(category_row, "Domain rule: Industry 4.0 or digital-industry categories require explicit digital-process, lifecycle, or collaborative-engineering evidence.")
    if code == "1A8" and not project_signals["wind_tunnel"]:
        return _rule_out_category(category_row, "Domain rule: wind-tunnel technology categories require explicit wind-tunnel evidence.")
    if code == "1A9" and not (project_signals["wind_tunnel"] and project_signals["instrumentation"]):
        return _rule_out_category(category_row, "Domain rule: wind-tunnel measuring categories require both wind-tunnel and instrumentation evidence.")
    if code == "1A5" and not project_signals["high_lift"]:
        return _rule_out_category(category_row, "Domain rule: high-lift categories require explicit slat, flap, or take-off/landing-device evidence.")
    if code == "1A6" and not project_signals["wing_design_terms"]:
        return _rule_out_category(category_row, "Domain rule: wing-design categories require explicit wing, fairing, nacelle, fuselage, or empennage-geometry evidence.")
    if code == "1A11" and not project_signals["external_noise_context"]:
        return _rule_out_category(category_row, "Domain rule: external-noise categories require explicit aeroacoustic or external-noise evidence.")
    if code == "1A2" and not project_signals["unsteady_explicit"]:
        return _rule_out_category(category_row, "Domain rule: unsteady-aerodynamics categories require explicit unsteady, transient, or dynamic-aerodynamic evidence.")
    if project_signals.get("atm_navigation_core") and not project_signals.get("physical_aero_project"):
        if code in {"1A1", "1A2", "1A5", "1A6", "1A7", "1A8", "1A9", "1A10", "1A11", "1B10", "1B11", "1B12", "1B13"}:
            return _rule_out_category(category_row, "Domain rule: navigation or procedure projects without physical aero/test evidence should not use aero-physics or acoustic-test categories.")
    if code == "1F4" and not (project_signals["bench_ground_test"] or project_signals["wind_tunnel"] or (project_signals["testing_validation"] and project_signals["propulsion_core"])):
        return _rule_out_category(category_row, "Domain rule: flight/ground-test categories require explicit bench, campaign, wind-tunnel, or propulsion-test evidence.")
    if code in {"1F32", "1F33"} and not (project_signals["demonstrator_platform"] and (project_signals["bench_ground_test"] or project_signals["testing_validation"])):
        return _rule_out_category(category_row, "Domain rule: large-scale validation categories require explicit demonstrator/platform evidence plus real validation or test activity.")
    if code == "1F27" and not project_signals["systems_simulation_core"]:
        return _rule_out_category(category_row, "Domain rule: numerical-model categories require explicit simulation-environment or numerical-model evidence.")
    if code == "1F22" and not (project_signals["meta_methodology"] or project_signals["study_assessment_core"]):
        return _rule_out_category(category_row, "Domain rule: operational-research methods categories require explicit study, trade-off, methodology, or assessment evidence.")
    if code == "1C10" and not project_signals["bench_ground_test"]:
        return _rule_out_category(category_row, "Domain rule: test-bench categories require explicit bench, rig, or ground-test evidence.")
    if code == "1C15" and not project_signals["electrical_systems_core"]:
        return _rule_out_category(category_row, "Domain rule: electrical-power categories require explicit electrical-power or power-distribution evidence.")
    if code == "1D23" and not project_signals["fuel_systems_core"]:
        return _rule_out_category(category_row, "Domain rule: fuel-system categories require explicit fuel-system evidence.")
    if code in {"1D10", "1D5"} and not (project_signals["avionics_systems"] or project_signals["electrical_systems_core"] or project_signals["systems_simulation_core"]):
        return _rule_out_category(category_row, "Domain rule: avionics or on-board-electronics categories require explicit avionics, electrical-system, or system-simulation evidence.")
    if code == "1D14" and not project_signals["smart"]:
        return _rule_out_category(category_row, "Domain rule: smart-maintenance categories require explicit smart or predictive-maintenance evidence.")
    if code == "1F24" and not project_signals["flight_dynamics"]:
        return _rule_out_category(category_row, "Domain rule: aircraft-performance categories require explicit performance, stability, controllability, or flight-mechanics evidence.")
    if code == "1F25" and not project_signals["airport"]:
        return _rule_out_category(category_row, "Domain rule: airport-performance categories require explicit airport-performance evidence.")
    if code == "1I4" and not project_signals.get("explicit_training"):
        return _rule_out_category(category_row, "Domain rule: selection or training categories require explicit training or selection evidence.")
    if code in {"1G8", "1G9"} and not project_signals["airport"]:
        return _rule_out_category(category_row, "Domain rule: airport-operations categories require explicit airport or airport-traffic evidence.")
    if code == "1C11" and project_signals.get("out_of_domain_project"):
        return _rule_out_category(category_row, "Domain rule: engine-health categories are blocked for non-aerospace weak-fit projects.")
    if code in {"1A1", "1A10"} and not project_signals.get("cfd_explicit"):
        return _rule_out_category(category_row, "Domain rule: computational-fluid or computational-acoustics categories require explicit CFD or numerical-simulation evidence.")
    if code == "1B6" and not project_signals.get("aeroelastic_explicit"):
        return _rule_out_category(category_row, "Domain rule: aeroelasticity requires explicit flutter, aeroelastic, gust-response, or flexible-structure-in-flow evidence.")
    if code == "1B7" and not (project_signals.get("vibration_explicit") or project_signals["acoustics"]):
        return _rule_out_category(category_row, "Domain rule: buckling, vibration, or acoustics categories require explicit vibration, buckling, or acoustic evidence.")
    if "acoustics" in tags and not project_signals["acoustics"]:
        return _rule_out_category(category_row, "Domain rule: the project text does not mention acoustic, noise, or sound objectives.")
    if "smart" in tags and not project_signals["smart"]:
        return _rule_out_category(category_row, "Domain rule: smart materials require explicit sensing, actuation, or adaptability evidence.")
    if "security" in tags and not project_signals["security"]:
        return _rule_out_category(category_row, "Domain rule: aircraft security requires explicit crew/passenger/security measures.")
    if "testing" in tags and not project_signals["testing_validation"]:
        return _rule_out_category(category_row, "Domain rule: testing and validation categories require explicit test, experiment, measurement, or characterisation evidence.")
    if ("flight_dynamics" in tags or code.startswith("1E")) and not project_signals["flight_dynamics"] and not project_signals["failure_analysis"]:
        return _rule_out_category(category_row, "Domain rule: flight-mechanics categories require explicit performance, stability, control, or failure-analysis work.")
    if ("failure_analysis" in tags or "hazard_analysis" in tags) and not project_signals["failure_analysis"]:
        return _rule_out_category(category_row, "Domain rule: failure or hazard-analysis categories require explicit analysis of failures, incidents, or hazards.")
    if "digital_process" in tags and not project_signals["digital_process"]:
        return _rule_out_category(category_row, "Domain rule: digital/lifecycle/collaborative-engineering categories require explicit IT or lifecycle evidence.")
    if "innovation_umbrella" in tags and not project_signals["innovation_concepts"]:
        return _rule_out_category(category_row, "Domain rule: umbrella innovation categories are blocked unless the project explicitly targets concepts or scenarios.")
    if "propulsion" in tags and not project_signals["propulsion_core"]:
        return _rule_out_category(category_row, "Domain rule: propulsion categories are blocked when propulsion is only hazard context.")
    if "human" in tags and not project_signals["human"]:
        return _rule_out_category(category_row, "Domain rule: human-related categories require explicit human/crew/operator evidence.")
    if "airport" in tags and not project_signals["airport"]:
        return _rule_out_category(category_row, "Domain rule: airport-related categories require explicit airport evidence.")
    if "atm" in tags and not project_signals["atm"]:
        return _rule_out_category(category_row, "Domain rule: ATM/navigation categories require explicit air-traffic evidence.")
    if "uas" in tags and not project_signals["uas"]:
        return _rule_out_category(category_row, "Domain rule: UAS categories require explicit unmanned/autonomous evidence.")
    if "metallic" in tags and not project_signals["metallic_materials"]:
        return _rule_out_category(category_row, "Domain rule: metallic-material categories are blocked because the project evidence points to fibres and non-metallic materials.")
    if "nonmetal" in tags and not project_signals["nonmetal_materials"]:
        return _rule_out_category(category_row, "Domain rule: non-metallic-material categories require explicit fibre, membrane, ceramic, or non-metallic evidence.")
    if "composite" in tags and not (project_signals["composite_materials"] or project_signals["nonmetal_materials"]):
        return _rule_out_category(category_row, "Domain rule: composite-material categories require explicit composite or fibre-based material evidence.")

    return None


def _apply_category_domain_rules(category_evaluation, category_row, project_signals):
    adjusted = dict(category_evaluation)
    tags = _category_tags(category_row)
    label_text = _category_label_text(category_row)

    if ("nonmetal" in tags or _contains_any(label_text, ["non-metallic"])) and project_signals["nonmetal_materials"]:
        adjusted["relevance"] = max(adjusted["relevance"], 2)
        adjusted["confidence"] = max(adjusted["confidence"], 95)
        adjusted["reason"] = "Domain rule: the project explicitly mentions fibres, membranes, or other non-metallic materials, which is direct evidence for this materials category."

    if ("composite" in tags or "composite" in label_text) and (project_signals["composite_materials"] or project_signals["nonmetal_materials"]):
        adjusted["relevance"] = max(adjusted["relevance"], 2)
        adjusted["confidence"] = max(adjusted["confidence"], 90)
        adjusted["reason"] = "Domain rule: the project explicitly focuses on fibre- or composite-based material choices, which is strong evidence for this composite-material category."

    if "structural_design" in tags and project_signals["structures_materials_core"] and project_signals["design_work"]:
        adjusted["relevance"] = max(adjusted["relevance"], 2)
        adjusted["confidence"] = max(adjusted["confidence"], 90)
        adjusted["reason"] = "Domain rule: the project explicitly includes structural design or integration work on the aircraft target, which strongly matches this structural-design category."

    if "manufacturing" in tags:
        if project_signals["manufacturing_process"]:
            adjusted["relevance"] = max(adjusted["relevance"], 2)
        elif project_signals["manufacture_explicit"]:
            if adjusted["relevance"] > 0:
                adjusted["relevance"] = 1
                adjusted["confidence"] = max(adjusted["confidence"], 90)
                adjusted["reason"] = "Domain rule: manufacturing is explicit in the project, but no dedicated manufacturing-process research is described, so this category can only be secondary."
        else:
            return _rule_out_category(category_row, "Domain rule: manufacturing categories require explicit manufacturing or production evidence.")

    if "testing" in tags and not project_signals["testing_validation"]:
        return _rule_out_category(category_row, "Domain rule: testing categories are blocked because the project text does not mention tests, experiments, or validation.")

    return adjusted


def _parse_family_evaluation(data, expected_code: str):
    if not isinstance(data, dict):
        return None

    return {
        "family_code": expected_code,
        "relevance": _coerce_int(data.get("relevance"), 0, 3, 0),
        "confidence": _coerce_int(data.get("confidence"), 0, 100, 50),
        "reason": str(data.get("reason", "")).strip() or "No explanation returned.",
    }


def _parse_category_evaluation(data, expected_code: str):
    if not isinstance(data, dict):
        return None

    return {
        "category_code": expected_code,
        "relevance": _coerce_int(data.get("relevance"), 0, 3, 0),
        "confidence": _coerce_int(data.get("confidence"), 0, 100, 50),
        "reason": str(data.get("reason", "")).strip() or "No explanation returned.",
    }


def _parse_category_reason(data, expected_code: str):
    if not isinstance(data, dict):
        return None

    reason = str(data.get("reason", "")).strip()
    if not reason:
        return None

    return {
        "category_code": expected_code,
        "reason": reason,
    }


def _deterministic_reason(category_row, category_evaluation, project_profile, project_signals):
    tags = _category_tags(category_row)
    label_text = _category_label_text(category_row)

    if ("composite" in tags or "composite" in label_text) and (project_signals["composite_materials"] or project_signals["nonmetal_materials"]):
        return "The project relies on fibre-based or composite material choices, so this composite-material category is a strong match."
    if ("nonmetal" in tags or _contains_any(label_text, ["non-metallic"])) and project_signals["nonmetal_materials"]:
        return "The project explicitly mentions fibres, membranes, or other non-metallic material evidence, so this non-metallic-material category is a strong match."
    if "structural_design" in tags and project_signals["design_work"] and project_signals["structural_terms"]:
        return "The project includes explicit structural design, verification, or airframe-integration work, which makes this structural-design category a strong match."
    if "manufacturing" in tags and project_signals["manufacture_explicit"] and not project_signals["manufacturing_process"]:
        return "Manufacturing is explicitly mentioned in the project, but mainly as a supporting activity rather than a process-research topic, so this manufacturing category is secondary."
    if "testing" in tags and not project_signals["testing_validation"]:
        return "This category would only become strong if the project explicitly included test, experiment, or material-characterisation work."

    base_reason = str(category_evaluation.get("reason", "")).strip()
    if base_reason.startswith("Domain rule:"):
        base_reason = base_reason[len("Domain rule:"):].strip()

    if not base_reason:
        return None
    if category_evaluation["relevance"] <= 1:
        return f"Secondary fit: {base_reason[0].lower() + base_reason[1:]}" if len(base_reason) > 1 else f"Secondary fit: {base_reason.lower()}"
    return base_reason[0].upper() + base_reason[1:]


def _tokenize_overlap_text(text: str):
    tokens = re.findall(r"[a-z0-9]+", _normalize_text(text))
    return [token for token in tokens if len(token) > 2 and token not in WEAK_FIT_STOPWORDS]


def _build_project_overlap_tokens(project_text: str, project_profile):
    source_text = " ".join(
        [
            _normalize_text(project_text),
            _normalize_text(project_profile.get("primary_focus", "")),
            _normalize_text(project_profile.get("secondary_focus", "")),
            _normalize_text(project_profile.get("target_object", "")),
            _normalize_text(" ".join(project_profile.get("materials_or_methods", []))),
            _normalize_text(" ".join(project_profile.get("evidence_phrases", []))),
        ]
    )
    return set(_tokenize_overlap_text(source_text))


def _all_taxonomy_categories(taxonomy_df):
    tax = taxonomy_df.fillna("")
    root_lookup = {
        str(row["category_code"]).strip().upper(): str(row["category_label"]).strip()
        for _, row in tax.loc[tax["parent_code"].astype(str).str.strip() == ""].iterrows()
    }

    categories = []
    child_rows = tax.loc[tax["parent_code"].astype(str).str.strip() != ""].sort_values("category_code")
    for _, row in child_rows.iterrows():
        family_code = str(row["parent_code"]).strip().upper()
        row_text = str(row.get("text", "")).replace("\n", " ").strip()
        categories.append(
            {
                "category_code": str(row["category_code"]).strip().upper(),
                "category_label": str(row["category_label"]).strip(),
                "family_code": family_code,
                "family_label": root_lookup.get(family_code, ""),
                "snippet": row_text[:CATEGORY_SNIPPET_CHARS] + ("..." if len(row_text) > CATEGORY_SNIPPET_CHARS else ""),
            }
        )
    return categories


def _weak_fit_score(category_row, project_tokens, project_signals):
    category_text = " ".join(
        [
            _normalize_text(category_row["category_label"]),
            _normalize_text(category_row["snippet"]),
            _normalize_text(category_row["family_label"]),
        ]
    )
    category_tokens = set(_tokenize_overlap_text(category_text))
    label_tokens = set(_tokenize_overlap_text(category_row["category_label"]))
    overlap = sorted(project_tokens & category_tokens)
    label_overlap = project_tokens & label_tokens

    score = len(overlap) + (len(label_overlap) * 3)
    label_text = _category_label_text(category_row)
    family_code = category_row["family_code"]

    if project_signals["meta_nontechnical_project"]:
        if family_code == "1F0":
            score += 6
        if family_code == "1J0":
            score += 3
        if family_code == "1L0":
            score += 2
        if "methodology" in label_text:
            score += 14
        if _contains_any(label_text, ["methodology", "methods", "tools", "operational research", "modelling", "modeling"]):
            score += 10
        if _contains_any(label_text, ["validation", "life-cycle", "lifecycle", "decision support", "scenario"]):
            score += 7
        if _contains_any(label_text, ["methodology"]):
            score += 12
        if _contains_any(label_text, ["operational research", "decision support", "scenarios analysis"]):
            score += 8
        if _contains_any(label_text, ["collaborative product", "process engineering", "life-cycle integration"]):
            score += 6
        if family_code in {"1A0", "1B0", "1C0", "1D0", "1E0"} and len(overlap) <= 1:
            score -= 6

    if project_signals.get("out_of_domain_project"):
        if family_code == "1F0":
            score += 7
        if _contains_any(label_text, ["maintenance", "fault tolerant", "reliability"]):
            score += 10
        if _contains_any(label_text, ["information management", "knowledge management", "information processing"]):
            score += 8
        if _contains_any(label_text, ["methodology", "methods", "tools", "operational research"]):
            score += 7
        if family_code in {"1A0", "1B0", "1C0", "1D0", "1G0", "1H0"}:
            score -= 8

    return score, overlap[:5]


def _deterministic_weak_reason(project_profile, category_row, overlap_terms, project_signals):
    label = category_row["category_label"]
    if project_signals.get("out_of_domain_project"):
        if overlap_terms:
            overlap_text = ", ".join(overlap_terms[:3])
            return f"Weak fit: this project is outside the aerospace core of the taxonomy, and {label} is one of the closest method- or maintenance-oriented proxies because it overlaps on {overlap_text}."
        return f"Weak fit: this project is outside the aerospace core of the taxonomy, and {label} is one of the closest proxy categories available."
    if project_signals["meta_nontechnical_project"]:
        if overlap_terms:
            overlap_text = ", ".join(overlap_terms[:3])
            return f"Weak fit: the project is mainly methodological, and {label} is one of the closest taxonomy proxies because it overlaps on {overlap_text}."
        return f"Weak fit: the project is mainly methodological, and {label} is one of the closest process- or method-oriented categories in the taxonomy."

    if overlap_terms:
        overlap_text = ", ".join(overlap_terms[:3])
        return f"Weak fit: this category is one of the closest matches available in the taxonomy because it overlaps with the project on {overlap_text}."
    return f"Weak fit: this is one of the closest categories available in the taxonomy, even though the project does not map cleanly to a strong technical topic."


def _weak_fit_predictions(project_text, taxonomy_df, project_profile, project_signals, top_k):
    all_categories = _all_taxonomy_categories(taxonomy_df)
    project_tokens = _build_project_overlap_tokens(project_text, project_profile)
    scored = []

    for category_row in all_categories:
        score, overlap_terms = _weak_fit_score(category_row, project_tokens, project_signals)
        if score <= 0:
            continue
        scored.append(
            {
                "category_code": category_row["category_code"],
                "category_label": category_row["category_label"],
                "family_code": category_row["family_code"],
                "family_label": category_row["family_label"],
                "score": score,
                "overlap_terms": overlap_terms,
                "reason": _deterministic_weak_reason(project_profile, category_row, overlap_terms, project_signals),
            }
        )

    if not scored and project_signals["meta_nontechnical_project"]:
        fallback_rows = []
        for row in all_categories:
            label_text = _category_label_text(row)
            if row["family_code"] not in {"1F0", "1J0", "1L0"}:
                continue
            if not _contains_any(
                label_text,
                [
                    "methodology",
                    "methods",
                    "tools",
                    "operational research",
                    "decision support",
                    "life-cycle",
                    "lifecycle",
                    "scenario",
                ],
            ):
                continue
            fallback_rows.append(row)

        fallback_rows.sort(key=lambda row: row["category_code"])
        for row in fallback_rows[: max(top_k * 2, 8)]:
            scored.append(
                {
                    "category_code": row["category_code"],
                    "category_label": row["category_label"],
                    "family_code": row["family_code"],
                    "family_label": row["family_label"],
                    "score": 1,
                    "overlap_terms": [],
                    "reason": _deterministic_weak_reason(project_profile, row, [], project_signals),
                }
            )

    scored.sort(key=lambda item: (-item["score"], item["category_code"]))
    selected = scored[:top_k]
    ranked_predictions = [
        {
            "category_code": item["category_code"],
            "reason": item["reason"],
        }
        for item in selected
    ]

    debug_payload = {
        "project_profile": project_profile,
        "project_signals": project_signals,
        "weak_fit_mode": True,
        "weak_fit_project_tokens": sorted(project_tokens)[:40],
        "weak_fit_candidates": scored[: min(max(top_k * 4, 12), 25)],
        "final_predictions": ranked_predictions,
    }
    selection_note = (
        f"Prompting puro con weak-fit scoring: Top-{len(ranked_predictions)} generado "
        f"por cercania taxonomica al no detectar un encaje tecnico fuerte"
    )
    if len(ranked_predictions) < top_k:
        selection_note += f" de {top_k} solicitadas."
    else:
        selection_note += "."

    return ranked_predictions, selection_note, json.dumps(debug_payload, ensure_ascii=False, indent=2)


def _signal_summary(project_signals) -> list[str]:
    summary = []
    ordered_signals = [
        ("meta_methodology", "methodology / impact-assessment project"),
        ("wind_tunnel", "wind-tunnel work is explicit"),
        ("aerodynamics_physics", "aerodynamic work is explicit"),
        ("high_lift", "slat / flap / take-off / landing work is explicit"),
        ("wing_design_terms", "wing / airframe configuration work is explicit"),
        ("acoustics", "acoustic or noise objectives are explicit"),
        ("external_noise_context", "the acoustic context is external / aeroacoustic"),
        ("internal_noise_context", "the acoustic context is internal / cabin noise"),
        ("testing_validation", "testing / verification / measurements are explicit"),
        ("instrumentation", "probes / instrumentation / gauges are explicit"),
        ("fem_structural", "FEM or structural verification is explicit"),
        ("structures_materials_core", "structures / materials are core to the work"),
        ("nonmetal_materials", "non-metallic materials are explicit"),
        ("composite_materials", "composite materials are explicit"),
        ("shield_protection", "physical protection / impact-resistance is explicit"),
        ("manufacturing_process", "manufacturing-process work is explicit"),
        ("manufacture_explicit", "manufacturing is explicitly mentioned"),
        ("cfd_explicit", "CFD / numerical flow simulation is explicit"),
        ("unsteady_explicit", "unsteady / dynamic aerodynamic behaviour is explicit"),
        ("propulsion_core", "propulsion is a direct technical target"),
        ("atm", "ATM / navigation work is explicit"),
        ("airport", "airport work is explicit"),
        ("human", "human / crew / operator factors are explicit"),
        ("uas", "unmanned / autonomous flight is explicit"),
    ]

    for key, label in ordered_signals:
        if project_signals.get(key):
            summary.append(label)
    return summary[:8]


def _family_score_bonus(family_code: str, project_signals) -> tuple[float, list[str]]:
    bonuses = []
    score = 0.0

    if family_code == "1A0" and project_signals["physical_aero_project"]:
        score += 6.0
        bonuses.append("aerodynamic work is explicit")
    if family_code == "1B0" and (project_signals["structures_materials_core"] or project_signals["fem_structural"]):
        score += 6.0
        bonuses.append("structures or materials work is explicit")
    if family_code == "1B0" and (project_signals.get("process_tooling_manufacturing") or project_signals.get("metallic_process")):
        score += 4.0
        bonuses.append("manufacturing, tooling, or metallic-process work is explicit")
    if family_code == "1C0" and project_signals["propulsion_core"]:
        score += 6.0
        bonuses.append("propulsion is a direct target")
    if family_code == "1C0" and project_signals.get("bench_ground_test") and project_signals["propulsion_core"]:
        score += 3.0
        bonuses.append("propulsion-related bench or ground testing is explicit")
    if family_code == "1D0" and project_signals["avionics_systems"]:
        score += 6.0
        bonuses.append("systems or avionics work is explicit")
    if family_code == "1D0" and (project_signals.get("systems_simulation_core") or project_signals.get("electrical_systems_core")):
        score += 5.0
        bonuses.append("system simulation, electrical, or on-board architecture work is explicit")
    if family_code == "1E0" and project_signals["flight_dynamics"]:
        score += 6.0
        bonuses.append("flight-mechanics work is explicit")
    if family_code == "1F0" and (project_signals["digital_process"] or project_signals["meta_methodology"]):
        score += 6.0 if project_signals["meta_methodology"] else 4.0
        bonuses.append("methods, tools, or methodology work is explicit")
    if family_code == "1F0" and (project_signals.get("systems_simulation_core") or project_signals.get("bench_ground_test")):
        score += 4.0
        bonuses.append("simulation, bench, or test-environment work is explicit")
    if family_code == "1G0" and project_signals["atm"]:
        score += 7.0
        bonuses.append("ATM or navigation work is explicit")
    if family_code == "1G0" and project_signals.get("atm_navigation_core"):
        score += 5.0
        bonuses.append("navigation procedures or GNSS/EGNOS deployment are explicit")
    if family_code == "1F0" and (project_signals.get("flight_procedures") or project_signals.get("certification_work")):
        score += 3.0
        bonuses.append("validation, certification, or operational-method work is explicit")
    if family_code == "1H0" and project_signals["airport"]:
        score += 7.0
        bonuses.append("airport work is explicit")
    if family_code == "1I0" and project_signals["human"]:
        score += 7.0
        bonuses.append("human-factors work is explicit")
    if family_code == "1J0" and project_signals["innovation_concepts"]:
        score += 4.0
        bonuses.append("concepts or scenarios work is explicit")
    if family_code == "1K0" and project_signals["uas"]:
        score += 7.0
        bonuses.append("UAS work is explicit")
    if family_code == "1L0" and project_signals["digital_process"]:
        score += 4.0
        bonuses.append("digital-industry work is explicit")

    return score, bonuses


def _deterministic_candidate_score(category_row, positive_text: str, project_tokens, project_signals):
    blocked = _precheck_category_domain_rules(category_row, project_signals)
    if blocked is not None:
        blocked["family_code"] = category_row["family_code"]
        blocked["family_label"] = category_row["family_label"]
        blocked["score"] = -999.0
        blocked["evidence"] = []
        blocked["overlap_terms"] = []
        return blocked

    category_text = " ".join(
        [
            _normalize_text(category_row["category_label"]),
            _normalize_text(category_row["snippet"]),
            _normalize_text(category_row["family_label"]),
        ]
    )
    category_tokens = set(_tokenize_overlap_text(category_text))
    label_tokens = set(_tokenize_overlap_text(category_row["category_label"]))
    overlap_terms = sorted(project_tokens & category_tokens)
    label_overlap_terms = sorted(project_tokens & label_tokens)

    score = (1.2 * len(overlap_terms)) + (2.6 * len(label_overlap_terms))
    evidence = []

    family_bonus, family_evidence = _family_score_bonus(category_row["family_code"], project_signals)
    score += family_bonus
    evidence.extend(family_evidence)

    label_text = _category_label_text(category_row)
    tags = _category_tags(category_row)

    def add(condition: bool, points: float, note: str):
        nonlocal score
        if condition:
            score += points
            evidence.append(note)

    add("wind tunnel" in label_text and project_signals["wind_tunnel"], 5.0, "wind-tunnel evidence is explicit")
    add("testing" in label_text and project_signals["testing_validation"], 2.5, "testing or verification is explicit")
    add("measurement" in label_text and project_signals["instrumentation"], 2.5, "instrumentation or probes are explicit")
    add("design" in label_text and project_signals["design_work"], 1.5, "design work is explicit")
    add("bench_testing" in tags and project_signals["bench_ground_test"], 3.5, "bench or ground-test evidence is explicit")
    add("simulation_models" in tags and project_signals["systems_simulation_core"], 4.0, "simulation or numerical-model work is explicit")
    add(
        "systems_electrical" in tags and (project_signals["avionics_systems"] or project_signals["electrical_systems_core"] or project_signals["systems_simulation_core"]),
        4.0,
        "systems, avionics, or electrical-architecture work is explicit",
    )
    add("manufacturing" in tags and project_signals.get("process_tooling_manufacturing"), 3.5, "tooling, metrology, or process-manufacturing work is explicit")

    if "generic_methods" in tags and project_signals["strong_technical_topic"] and not project_signals["meta_methodology"] and not project_signals.get("study_assessment_core"):
        score -= 5.0
        evidence.append("generic methods wording is present, but a stronger technical topic dominates the project")
    if "large_validation" in tags and not (project_signals["demonstrator_platform"] and (project_signals["bench_ground_test"] or project_signals["testing_validation"])):
        score -= 6.0

    if _contains_any(label_text, ["wind tunnel testing /technology"]):
        add(project_signals["wind_tunnel"], 14.0, "the project directly builds or tests a wind-tunnel model")
    elif _contains_any(label_text, ["wind tunnel measuring techniques"]):
        add(project_signals["wind_tunnel"], 9.0, "wind-tunnel measurement work is explicit")
        add(project_signals["instrumentation"], 4.0, "pressure instrumentation, probes, or gauges are explicit")
    elif _contains_any(label_text, ["computational fluid dynamics"]):
        add(project_signals["cfd_explicit"], 12.0, "computational flow simulation is explicit")
        if project_signals["wind_tunnel"] and not project_signals["cfd_explicit"]:
            score -= 5.0
    elif _contains_any(label_text, ["unsteady aerodynamics"]):
        add(project_signals["unsteady_explicit"], 8.0, "unsteady or dynamic aerodynamic behaviour is explicit")
    elif _contains_any(label_text, ["high lift devices"]):
        add(project_signals["high_lift"], 12.0, "slats, flaps, or take-off / landing configurations are explicit")
    elif _contains_any(label_text, ["wing design"]):
        add(project_signals["wing_design_terms"], 10.0, "wing, fairing, empennage, or fuselage integration work is explicit")
    elif _contains_any(label_text, ["external noise prediction"]):
        add(project_signals["external_noise_context"], 10.0, "external aerodynamic or engine-noise assessment is explicit")
    elif _contains_any(label_text, ["computational acoustics"]):
        add(project_signals["acoustics"] and project_signals["cfd_explicit"], 8.0, "computational acoustics is explicit")
        if project_signals["acoustics"] and project_signals["wind_tunnel"] and not project_signals["cfd_explicit"]:
            score -= 2.5
    elif _contains_any(label_text, ["acoustics measurements and test technology"]):
        add(project_signals["acoustics"] and project_signals["testing_validation"], 10.0, "acoustic testing or measurement is explicit")
    elif _contains_any(label_text, ["structures behaviour and material testing"]):
        add(project_signals["fem_structural"], 8.0, "structural verification or FEM work is explicit")
        add(project_signals["testing_validation"] and project_signals["instrumentation"], 4.0, "test instrumentation or load measurements are explicit")
    elif "structural_design" in tags:
        add(project_signals["structural_terms"] and project_signals["design_work"], 8.0, "structural design or integration work is explicit")
        add(project_signals["shield_protection"], 5.0, "protective structural constraints are explicit")
    elif "metallic" in tags:
        add(project_signals["metallic_materials"] or project_signals["metallic_process"], 10.0, "metallic-material or metallic-process evidence is explicit")
    elif "composite" in tags:
        add(project_signals["composite_materials"], 12.0, "composite-material evidence is explicit")
        add(project_signals["nonmetal_materials"] and not project_signals["composite_materials"], 4.0, "fibre-based material evidence is explicit")
        if not project_signals["nonmetal_materials"] and not project_signals["composite_materials"]:
            score -= 4.0
    elif "nonmetal" in tags:
        add(project_signals["nonmetal_materials"], 12.0, "fibre, membrane, or non-metallic material evidence is explicit")
        if not project_signals["nonmetal_materials"]:
            score -= 4.0
    elif "manufacturing" in tags:
        add(project_signals["manufacturing_process"], 10.0, "manufacturing-process work is explicit")
        add(project_signals.get("process_tooling_manufacturing"), 5.0, "tooling, metrology, or industrialisation work is explicit")
        add(project_signals["manufacture_explicit"], 3.0, "manufacturing is explicitly mentioned")
        if project_signals["manufacture_explicit"] and not project_signals["manufacturing_process"]:
            evidence.append("manufacturing appears secondary rather than process-centred")
    elif _contains_any(label_text, ["internal noise prediction"]):
        add(project_signals["internal_noise_context"], 10.0, "internal or cabin-noise context is explicit")
        if not project_signals["internal_noise_context"]:
            score -= 8.0
    elif _contains_any(label_text, ["helicopter aeroacoustics"]):
        rotorcraft_context = _contains_any(positive_text, ["helicopter", "rotorcraft"])
        add(project_signals["acoustics"] and rotorcraft_context, 10.0, "rotorcraft acoustic work is explicit")
        if not rotorcraft_context:
            score -= 6.0
    elif _contains_any(label_text, ["noise reduction"]):
        add(
            project_signals["acoustics"]
            and _contains_any(positive_text, ["noise reduction", "noise control", "noise shielding", "noise attenuation"]),
            7.0,
            "a noise-reduction objective is explicit",
        )
    elif _contains_any(label_text, ["methodology"]):
        add(project_signals["meta_methodology"], 14.0, "methodology work is explicit")
    elif _contains_any(label_text, ["operational research methods & tools"]):
        add(
            project_signals["meta_methodology"] or project_signals.get("study_assessment_core"),
            8.0,
            "study, assessment, or operational-research work is explicit",
        )
        add(project_signals.get("flight_procedures") and not project_signals["strong_technical_topic"], 2.0, "procedure design or evaluation is explicit")
    elif _contains_any(label_text, ["collaborative product & process engineering"]):
        add(
            _contains_any(positive_text, ["process", "engineering methods", "engineering tools", "collaborative"]),
            6.0,
            "process or engineering-method work is explicit",
        )
    elif _contains_any(label_text, ["life-cycle integration", "lifecycle integration"]):
        add(_contains_any(positive_text, ["life-cycle", "lifecycle", "process integration"]), 6.0, "process-integration work is explicit")
    elif _contains_any(label_text, ["flight/ ground tests", "flight/ground tests"]):
        add(
            project_signals["bench_ground_test"] or _contains_any(positive_text, ["test campaign", "ground test", "flight test", "trial", "trials", "icing tests"]),
            8.0,
            "bench, campaign, or explicit test activity is present",
        )
    elif _contains_any(label_text, ["system certification"]):
        add(project_signals.get("certification_work"), 9.0, "certification or standards work is explicit")
    elif _contains_any(label_text, ["collaborative decision making"]):
        add(_contains_any(positive_text, ["decision maker", "decision makers", "stakeholders", "users"]), 5.0, "decision-making or stakeholder adoption is explicit")
    elif _contains_any(label_text, ["decision support systems"]):
        add(_contains_any(positive_text, ["support", "guidance", "decision makers", "adoption"]), 4.0, "decision support or adoption support is explicit")
    elif _contains_any(label_text, ["overall atm"]):
        add(project_signals["atm"], 9.0, "ATM operations are explicit")
        add(project_signals.get("flight_procedures"), 3.0, "new or validated flight procedures are explicit")
    elif _contains_any(label_text, ["airspace management"]):
        add(_contains_any(positive_text, ["airspace", "ifr", "curved", "approach", "approaches", "routes", "route"]), 8.0, "airspace, IFR, or route-design work is explicit")
    elif _contains_any(label_text, ["flow and capacity management"]):
        add(_contains_any(positive_text, ["controller workload", "air traffic controllers", "traffic level", "flow", "capacity"]), 8.0, "controller workload or traffic-flow management is explicit")
    elif _contains_any(label_text, ["communications and systems technology"]):
        add(_contains_any(positive_text, ["communications", "guidance", "surveillance/navigation concept"]), 6.0, "communications or support-system technology is explicit")
    elif _contains_any(label_text, ["navigation systems"]):
        add(project_signals.get("egnos_gnss"), 12.0, "EGNOS, GNSS, SBAS, or LPV navigation is explicit")
        add(_contains_any(positive_text, ["approach procedures", "apv", "lpv", "navigation concept"]), 3.0, "navigation procedures are explicit")
    elif _contains_any(label_text, ["atm automated support"]):
        add(_contains_any(positive_text, ["guidance means", "automated support", "simulation platform", "support"],), 7.0, "ATM support or procedure-enablement systems are explicit")
    elif _contains_any(label_text, ["airport traffic management"]):
        add(_contains_any(positive_text, ["airport traffic", "airport surveillance/navigation concept"]), 8.0, "airport traffic-management evidence is explicit")
    elif _contains_any(label_text, ["airport operations"]):
        add(_contains_any(positive_text, ["airport surveillance/navigation concept", "airport operations", "air taxi operations"]), 8.0, "airport-operations evidence is explicit")
    elif _contains_any(label_text, ["airline operations"]):
        add(_contains_any(positive_text, ["operations", "operational deployment", "rotorcraft operations", "airline", "helicopter operations"]), 8.0, "operational deployment or airline/helicopter operations are explicit")
    elif _contains_any(label_text, ["r&d management and co-ordination", "r&d management and coordination"]):
        add(_contains_any(positive_text, ["r&d", "coordination", "consortium", "work packages"]), 4.0, "R&D coordination is explicit")
    elif _contains_any(label_text, ["numerical models"]):
        add(project_signals["systems_simulation_core"], 12.0, "numerical-model or simulation-environment work is explicit")
    elif _contains_any(label_text, ["general purpose equipment"]):
        add(project_signals["bench_ground_test"] or project_signals.get("process_tooling_manufacturing"), 7.0, "equipment or hardware for a test environment is explicit")
    elif _contains_any(label_text, ["reference data for r&d use", "reference data for r&d use and live/rt data use"]):
        add(project_signals["instrumentation"] or _contains_any(positive_text, ["data processing", "test data", "reference data", "data acquisition"]), 7.0, "reference-data or test-data use is explicit")
    elif _contains_any(label_text, ["large scale validation experiments"]):
        add(project_signals["demonstrator_platform"] and (project_signals["bench_ground_test"] or project_signals["testing_validation"]), 7.0, "demonstrator validation experiments are explicit")
    elif _contains_any(label_text, ["large scale validation platforms"]):
        add(project_signals["demonstrator_platform"] and (project_signals["bench_ground_test"] or project_signals["testing_validation"]), 7.0, "demonstrator or validation-platform work is explicit")
    elif _contains_any(label_text, ["test bench calibration"]):
        add(project_signals["bench_ground_test"] and project_signals["instrumentation"], 10.0, "bench, rig, or calibration activity is explicit")
    elif _contains_any(label_text, ["electrical power generation & distribution"]):
        add(project_signals["electrical_systems_core"], 10.0, "electrical-power or power-distribution work is explicit")
    elif _contains_any(label_text, ["avionics integration"]):
        add(project_signals["avionics_systems"] or project_signals["systems_simulation_core"], 9.0, "avionics integration or architecture work is explicit")
    elif _contains_any(label_text, ["electronics & microelectronics for on-board systems"]):
        add(project_signals["electrical_systems_core"] or project_signals["systems_simulation_core"], 9.0, "on-board electronics or electrical-system work is explicit")
    elif _contains_any(label_text, ["fuel systems"]):
        add(project_signals["fuel_systems_core"], 10.0, "fuel-system work is explicit")
    elif _contains_any(label_text, ["scenarios analysis"]):
        add(project_signals["meta_methodology"] or _contains_any(positive_text, ["scenario", "scenarios"]), 5.0, "scenario-oriented analysis is explicit")
    elif _contains_any(label_text, ["unconventional configurations and new aircraft concepts"]):
        add(_contains_any(positive_text, ["unconventional configuration", "new aircraft concept"]), 6.0, "new aircraft-concept work is explicit")
        if project_signals.get("atm_navigation_core"):
            score -= 4.0

    score = round(score, 2)
    return {
        "category_code": category_row["category_code"],
        "category_label": category_row["category_label"],
        "family_code": category_row["family_code"],
        "family_label": category_row["family_label"],
        "snippet": category_row["snippet"],
        "score": score,
        "overlap_terms": overlap_terms[:6],
        "evidence": list(dict.fromkeys(evidence))[:6],
    }


def _select_active_families(scored_candidates, top_k: int):
    family_buckets = {}
    for item in scored_candidates:
        family_buckets.setdefault(item["family_code"], []).append(item)

    family_rows = []
    for family_code, items in family_buckets.items():
        ordered_items = sorted(items, key=lambda row: (-row["score"], row["category_code"]))
        top_scores = [row["score"] for row in ordered_items[:2]]
        family_rows.append(
            {
                "family_code": family_code,
                "family_label": ordered_items[0]["family_label"],
                "family_score": round(sum(top_scores), 2),
                "top_category_score": ordered_items[0]["score"],
                "sample_categories": [row["category_code"] for row in ordered_items[:4]],
            }
        )

    family_rows.sort(key=lambda row: (-row["family_score"], -row["top_category_score"], row["family_code"]))
    if not family_rows:
        return []

    family_budget = _family_budget(top_k)
    top_score = family_rows[0]["family_score"]
    threshold = max(10.0, top_score * 0.45)

    selected = [family_rows[0]]
    for row in family_rows[1:]:
        if len(selected) >= family_budget:
            break
        if row["family_score"] >= threshold:
            selected.append(row)

    return selected


def _build_shortlist(project_text, taxonomy_df, project_profile, project_signals, top_k: int):
    all_categories = _all_taxonomy_categories(taxonomy_df)
    project_tokens = _build_project_overlap_tokens(project_text, project_profile)
    positive_text = _build_positive_project_text(project_text, project_profile)

    scored_candidates = []
    blocked_categories = []
    for category_row in all_categories:
        scored = _deterministic_candidate_score(category_row, positive_text, project_tokens, project_signals)
        if scored["score"] < 0:
            blocked_categories.append(scored)
            continue
        scored_candidates.append(scored)

    scored_candidates.sort(key=lambda row: (-row["score"], row["category_code"]))
    active_families = _select_active_families(scored_candidates, top_k)
    active_family_codes = {row["family_code"] for row in active_families}

    shortlist_limit = min(max(top_k * 2 + 2, 8), SHORTLIST_SIZE_MAX)
    if len(active_families) <= 1:
        shortlist_limit = min(shortlist_limit, max(top_k + 3, 8))
    shortlist = [row for row in scored_candidates if row["family_code"] in active_family_codes][:shortlist_limit]

    top_family_score = active_families[0]["family_score"] if active_families else 0.0
    weak_fit_mode = project_signals["meta_nontechnical_project"] or (
        not project_signals["strong_technical_topic"] and top_family_score < 8.0
    )

    return shortlist, scored_candidates, blocked_categories, active_families, weak_fit_mode


def _candidate_shortlist_reason(candidate) -> str:
    evidence = candidate.get("evidence", [])
    overlap = candidate.get("overlap_terms", [])
    parts = evidence[:2] or overlap[:2]
    if not parts:
        return "generic lexical proximity"
    return "; ".join(parts)


def _build_final_shortlist_prompt(project_text: str, shortlist, project_signals, top_k: int, weak_fit_mode: bool) -> str:
    signal_lines = _signal_summary(project_signals)
    signal_block = "\n".join(f"- {line}" for line in signal_lines) if signal_lines else "- No strong deterministic cues detected"

    candidate_lines = []
    for candidate in shortlist:
        snippet = candidate["snippet"][:SHORTLIST_SNIPPET_CHARS]
        shortlist_reason = _candidate_shortlist_reason(candidate)
        candidate_lines.append(
            f"- {candidate['category_code']} | {candidate['category_label']} | Family {candidate['family_code']} {candidate['family_label']} | "
            f"Why shortlisted: {shortlist_reason} | Taxonomy: {snippet}"
        )

    weak_fit_rule = (
        "This project does not map cleanly to a strong technical topic. Choose the closest taxonomy proxies from the list and make that clear in the reason."
        if weak_fit_mode
        else "Choose the strongest technical matches from the list. Prefer fewer categories over weak or speculative ones."
    )

    return f"""
You are classifying one aerospace project into taxonomy categories.

Task:
Return up to {top_k} ranked categories from the candidate list only.

Rules:
- Use ONLY the candidate categories below.
- Rank strongest to weakest.
- Prefer the real technical target of the work over context words.
- Do not confuse `ATM` with `atmospheric`.
- Do not treat `shielding engine noise` as a physical structural shield.
- Internal-noise categories require cabin, cockpit, passenger, or interior-noise context.
- CFD categories require explicit computational or numerical simulation evidence.
- Manufacturing categories should stay secondary unless the project clearly studies manufacturing processes or production methods.
- Reasons must be short, concrete, and evidence-based.
- {weak_fit_rule}
- Return ONLY JSON in the required format.

JSON format:
{{
  "ranked_predictions": [
    {{
      "category_code": "1A8",
      "reason": "Short concrete reason"
    }}
  ]
}}

Detected project cues:
{signal_block}

Project description:
{project_text}

Candidate categories:
{chr(10).join(candidate_lines)}
"""


def _parse_ranked_predictions_response(data, allowed_codes, top_k: int):
    raw_items = []
    if isinstance(data, dict) and isinstance(data.get("ranked_predictions"), list):
        raw_items = data["ranked_predictions"]
    elif isinstance(data, dict) and data.get("category_code"):
        raw_items = [data]
    elif isinstance(data, list):
        raw_items = data

    ranked_predictions = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("category_code", "")).strip().upper()
        reason = str(item.get("reason", "")).strip()
        if not code or code not in allowed_codes or code in seen:
            continue
        ranked_predictions.append(
            {
                "category_code": code,
                "reason": reason or "Selected from the shortlisted taxonomy candidates.",
            }
        )
        seen.add(code)
        if len(ranked_predictions) >= top_k:
            break
    return ranked_predictions


def _recover_ranked_codes_from_text(raw_text: str, allowed_codes, top_k: int):
    ranked_predictions = []
    seen = set()
    for match in re.finditer(r"\b1[A-L]\d{1,2}\b", str(raw_text).upper()):
        code = match.group(0)
        if code not in allowed_codes or code in seen:
            continue
        ranked_predictions.append({"category_code": code, "reason": ""})
        seen.add(code)
        if len(ranked_predictions) >= top_k:
            break
    return ranked_predictions


def _deterministic_final_reason(candidate, project_signals, weak_fit_mode: bool = False) -> str:
    evidence = candidate.get("evidence", [])
    label_text = _category_label_text(candidate)
    tags = _category_tags(candidate)

    if weak_fit_mode:
        if evidence:
            return f"Weak fit: this is one of the closest taxonomy proxies because {evidence[0]}."
        return "Weak fit: this is one of the closest taxonomy proxies available in the shortlisted categories."

    if _contains_any(label_text, ["wind tunnel testing /technology"]):
        return "The project explicitly designs or tests a low-speed wind-tunnel model, so wind-tunnel technology is a direct match."
    if _contains_any(label_text, ["wind tunnel measuring techniques"]):
        return "The project includes probes, pressure instrumentation, or measurement work in a wind-tunnel model, which fits wind-tunnel measuring techniques."
    if _contains_any(label_text, ["high lift devices"]):
        return "Slats, flaps, and take-off or landing configurations are explicit in the project, so high-lift devices are a strong match."
    if _contains_any(label_text, ["wing design"]):
        return "The project modifies or integrates wing, fuselage, fairing, or empennage geometry, which makes wing design a strong match."
    if _contains_any(label_text, ["external noise prediction"]):
        return "The project explicitly studies aerodynamic or engine-noise effects in a wind-tunnel context, so external noise prediction is a strong match."
    if _contains_any(label_text, ["acoustics measurements and test technology"]):
        return "Acoustic testing or measurement is explicit in the project, which makes acoustics measurements and test technology a strong match."
    if _contains_any(label_text, ["structures behaviour and material testing"]):
        return "Structural verification, FEM, loads, or test instrumentation are explicit in the project, so structures behaviour and material testing is a strong match."
    if "structural_design" in tags:
        if project_signals["shield_protection"]:
            return "The project includes structural design constraints for protecting aircraft components, so structural analysis and design is a strong match."
        return "The project includes explicit structural design, verification, or integration work, which makes structural analysis and design a strong match."
    if "metallic" in tags:
        return "The project explicitly targets metallic materials or metallic manufacturing routes, so this metallic-materials category is a strong match."
    if "composite" in tags:
        return "Composite or fibre-based material choices are explicit in the project, so composite materials are a strong match."
    if "nonmetal" in tags:
        return "The project explicitly mentions fibre, membrane, or other non-metallic material evidence, so non-metallic materials are a direct match."
    if "manufacturing" in tags:
        if project_signals["manufacturing_process"]:
            return "The project explicitly studies manufacturing or production methods, which makes advanced manufacturing processes a strong match."
        return "Manufacturing is explicit in the project, but mainly as a supporting activity, so advanced manufacturing processes is a secondary match."
    if _contains_any(label_text, ["methodology"]):
        return "The project is explicitly centred on methodology development, so Methodology is the closest direct taxonomy match."
    if _contains_any(label_text, ["operational research methods & tools"]):
        return "The project develops evaluation or operational-research methods, which makes operational research methods and tools a strong proxy."
    if _contains_any(label_text, ["flight/ ground tests", "flight/ground tests"]):
        return "The project explicitly includes bench, campaign, or ground-test activity, so flight or ground tests is a strong supporting match."
    if _contains_any(label_text, ["system certification"]):
        return "Standards, certification, or safety-enablement work is explicit in the project, so system certification is a strong supporting match."
    if _contains_any(label_text, ["numerical models"]):
        return "The project explicitly develops numerical or fast-time simulation models, so numerical models are a direct match."
    if _contains_any(label_text, ["general purpose equipment"]):
        return "The project develops or tests dedicated hardware or equipment for a test environment, which makes general purpose equipment a relevant match."
    if _contains_any(label_text, ["reference data for r&d use", "reference data for r&d use and live/rt data use"]):
        return "Reference data, test data, or data-acquisition outputs are explicit in the project, so this reference-data category is a relevant match."
    if _contains_any(label_text, ["large scale validation experiments"]):
        return "The project includes explicit demonstrator or validation-experiment activity, so large-scale validation experiments is a plausible supporting match."
    if _contains_any(label_text, ["large scale validation platforms"]):
        return "The project uses or builds a demonstrator or validation platform, so large-scale validation platforms is a plausible supporting match."
    if _contains_any(label_text, ["test bench calibration"]):
        return "The project explicitly involves bench, rig, or calibration work, so test-bench calibration is a strong match."
    if _contains_any(label_text, ["electrical power generation & distribution"]):
        return "Electrical-power or power-distribution behaviour is explicit in the project, so this electrical-power category is a strong match."
    if _contains_any(label_text, ["avionics integration"]):
        return "The project explicitly studies avionics architecture or avionics integration, so avionics integration is a strong match."
    if _contains_any(label_text, ["electronics & microelectronics for on-board systems"]):
        return "On-board electronics or electrical-system architecture is explicit in the project, so this electronics category is a strong match."
    if _contains_any(label_text, ["fuel systems"]):
        return "Fuel-system functionality is an explicit part of the project, so fuel systems is a direct match."
    if _contains_any(label_text, ["overall atm"]):
        return "The project focuses on ATM operations and procedure deployment, so Overall ATM is a strong match."
    if _contains_any(label_text, ["airspace management"]):
        return "The project defines IFR, curved, or route-management procedures, which makes airspace management a strong match."
    if _contains_any(label_text, ["flow and capacity management"]):
        return "The project explicitly addresses controller workload, traffic level, or capacity effects, which fits flow and capacity management."
    if _contains_any(label_text, ["communications and systems technology"]):
        return "The project relies on guidance, surveillance, or communications support to enable the new procedures, so communications and systems technology is a strong match."
    if _contains_any(label_text, ["navigation systems"]):
        return "EGNOS, GNSS, SBAS, or LPV-enabled navigation is explicit in the project, so navigation systems is a direct match."
    if _contains_any(label_text, ["atm automated support"]):
        return "The project uses ATM support, simulation, or guidance-enablement mechanisms to deploy the procedures, so ATM automated support is a strong match."
    if _contains_any(label_text, ["airport traffic management"]):
        return "The project includes explicit airport-traffic or airport surveillance/navigation operations, which makes airport traffic management a strong match."
    if _contains_any(label_text, ["airport operations"]):
        return "The project includes explicit airport or air-taxi operational deployment around navigation procedures, which makes airport operations a strong match."
    if _contains_any(label_text, ["airline operations"]):
        return "The project is centred on operational deployment of new helicopter, rotorcraft, or airline procedures, so airline operations is a strong match."

    if evidence:
        if len(evidence) >= 2:
            return f"The project matches this category because {evidence[0]} and {evidence[1]}."
        return f"The project matches this category because {evidence[0]}."
    return "This category is one of the strongest deterministic matches in the shortlisted taxonomy candidates."


def _reason_needs_rewrite(reason: str) -> bool:
    lowered = str(reason).strip().lower()
    if not lowered:
        return True
    if len(lowered.split()) < 6:
        return True
    generic_markers = [
        "explicit",
        "implicit",
        "implied",
        "may be",
        "might",
        "could be",
        "possible",
        "general purpose",
    ]
    return any(marker in lowered for marker in generic_markers)


def _select_families(family_evaluations, family_budget: int):
    ordered = sorted(
        family_evaluations,
        key=lambda item: (-item["relevance"], item["order"]),
    )
    strong = [item for item in ordered if item["relevance"] >= 2]
    moderate = [item for item in ordered if item["relevance"] == 1]

    if strong:
        return strong[:family_budget]
    if moderate:
        return moderate[:1]
    return []


def _is_indirect_match(reason: str) -> bool:
    lowered = str(reason).strip().lower()
    indirect_markers = [
        "indirect",
        "indirectly",
        "not direct",
        "not directly",
        "potentially",
        "could",
        "may ",
        "might",
        "tangential",
        "plausible",
    ]
    return any(marker in lowered for marker in indirect_markers)


def _select_categories(category_evaluations, top_k: int):
    positive = []
    for item in category_evaluations:
        if item["relevance"] <= 0:
            continue
        if item["family_relevance"] <= 1 and _is_indirect_match(item["reason"]):
            continue
        if item["relevance"] == 1 and _is_indirect_match(item["reason"]):
            continue
        positive.append(item)

    ordered = sorted(
        positive,
        key=lambda item: (-item["relevance"], -item["family_relevance"], item["order"]),
    )
    return ordered[:top_k]


def llm_predict_prompting(text, taxonomy_df, model_name: str, top_k):
    project_text = _prepare_project_text(text, MAX_PROJECT_CHARS)
    project_profile = _fallback_project_profile(project_text)
    project_signals = _build_project_signals(project_text, project_profile)

    if project_signals["meta_nontechnical_project"]:
        return _weak_fit_predictions(project_text, taxonomy_df, project_profile, project_signals, top_k)

    shortlist, scored_candidates, blocked_categories, active_families, weak_fit_mode = _build_shortlist(
        project_text,
        taxonomy_df,
        project_profile,
        project_signals,
        top_k,
    )

    if weak_fit_mode or not shortlist:
        return _weak_fit_predictions(project_text, taxonomy_df, project_profile, project_signals, top_k)

    final_prompt = _build_final_shortlist_prompt(project_text, shortlist, project_signals, top_k, weak_fit_mode=False)
    final_data, final_raw_json = _run_llm_json(
        final_prompt,
        model_name,
        max_new_tokens=FINAL_GEN_TOKENS,
        response_schema=RANKED_PREDICTIONS_SCHEMA,
    )

    shortlist_lookup = {row["category_code"]: row for row in shortlist}
    allowed_codes = set(shortlist_lookup)
    ranked_predictions = _parse_ranked_predictions_response(final_data, allowed_codes, top_k)
    used_deterministic_fallback = False
    shortlist_completion_count = 0

    if not ranked_predictions and final_raw_json:
        ranked_predictions = _recover_ranked_codes_from_text(final_raw_json, allowed_codes, top_k)

    if ranked_predictions:
        normalized_predictions = []
        seen = set()
        for item in ranked_predictions:
            code = item["category_code"]
            if code in seen:
                continue
            reason = str(item.get("reason", "")).strip()
            if _reason_needs_rewrite(reason):
                reason = _deterministic_final_reason(shortlist_lookup[code], project_signals)
            normalized_predictions.append({"category_code": code, "reason": reason})
            seen.add(code)
            if len(normalized_predictions) >= top_k:
                break
        ranked_predictions = normalized_predictions

        if len(ranked_predictions) < top_k and shortlist:
            top_shortlist_score = shortlist[0]["score"]
            completion_threshold = max(14.0, top_shortlist_score * 0.60)
            existing_codes = {item["category_code"] for item in ranked_predictions}
            for candidate in shortlist:
                if candidate["category_code"] in existing_codes:
                    continue
                if candidate["score"] < completion_threshold:
                    continue
                ranked_predictions.append(
                    {
                        "category_code": candidate["category_code"],
                        "reason": _deterministic_final_reason(candidate, project_signals),
                    }
                )
                existing_codes.add(candidate["category_code"])
                shortlist_completion_count += 1
                if len(ranked_predictions) >= top_k:
                    break

        ranked_predictions.sort(
            key=lambda item: (
                -shortlist_lookup[item["category_code"]]["score"],
                item["category_code"],
            )
        )
    else:
        used_deterministic_fallback = True
        ranked_predictions = [
            {
                "category_code": row["category_code"],
                "reason": _deterministic_final_reason(row, project_signals),
            }
            for row in shortlist[:top_k]
        ]

    selection_note = (
        f"Prompting puro con shortlist determinista + 1 llamada LLM: "
        f"{len(active_families)} familia(s) activas, {len(shortlist)} candidata(s) finalistas"
    )
    if blocked_categories:
        selection_note += f", {len(blocked_categories)} bloqueadas por reglas"
    if shortlist_completion_count:
        selection_note += f", completado con {shortlist_completion_count} candidata(s) fuerte(s) del shortlist"
    if used_deterministic_fallback:
        selection_note += ", se uso el ranking determinista por salida LLM incompleta o invalida"
    selection_note += f" y Top-{len(ranked_predictions)} generado"
    if len(ranked_predictions) < top_k:
        selection_note += f" de {top_k} solicitadas."
    else:
        selection_note += "."

    debug_payload = {
        "project_profile": project_profile,
        "project_signals": project_signals,
        "active_families": active_families,
        "blocked_categories": blocked_categories[:20],
        "deterministic_candidates": scored_candidates[: min(max(top_k * 4, 14), 24)],
        "llm_raw_output": final_raw_json,
        "used_deterministic_fallback": used_deterministic_fallback,
        "final_predictions": ranked_predictions,
    }

    return ranked_predictions, selection_note, json.dumps(debug_payload, ensure_ascii=False, indent=2)
