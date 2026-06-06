import json
from ollama_client import ollama_generate
from ranking_utils import rerank_predictions_with_domain_bias


def extract_last_json_object(text: str):
    end = text.rfind("}")
    if end == -1:
        return ""

    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                return text[i:end + 1]

    return ""


def build_prompt_hybrid(text, predictions, taxonomy_df, top_k):

    rows = []

    for c in predictions:
        code = c["category_code"]
        row = taxonomy_df.loc[taxonomy_df["category_code"] == code].iloc[0]
        label = row["category_label"]
        rows.append(f"{code}: {label} (embedding_score={c['score']:.3f})")

    cand_block = "\n".join(rows)

    return f"""
You are an expert aerospace research project classifier.

Re-rank ALL {top_k} candidate categories by relevance to the project description.

STRICT RULES:
1. ranked_predictions MUST contain ALL {top_k} categories, just re-ordered by relevance
2. ranked_predictions MUST include category_code AND a short reason explaining why it's relevant
3. All category_code values MUST be different
4. Do NOT invent category codes
5. Do NOT return explanations outside the JSON
6. Return ONLY the JSON object

JSON format:
{{
  "ranked_predictions": [
    {{"category_code": "CODE1", "reason": "Short explanation"}},
    {{"category_code": "CODE2", "reason": "Short explanation"}},
    ...
  ]
}}

Candidate categories (pre-selected by embeddings):
{cand_block}

Project description:
{text}
"""
    

def _parse_model_ranked_predictions(clean_json, valid_codes):
    parsed = []
    if not clean_json:
        return parsed

    data = json.loads(clean_json)
    seen = set()
    for item in data.get("ranked_predictions", []):
        code = str(item.get("category_code", "")).strip().upper()
        reason = str(item.get("reason", "")).strip()
        if code in valid_codes and code not in seen:
            parsed.append({"category_code": code, "reason": reason})
            seen.add(code)

    return parsed


def _complete_with_embedding_fallback(ranked_predictions, original_order, top_k):
    seen = {item["category_code"] for item in ranked_predictions}
    for code in original_order:
        if len(ranked_predictions) == top_k:
            break
        if code not in seen:
            ranked_predictions.append(
                {
                    "category_code": code,
                    "reason": "Retrieved from embedding similarity (not re-ranked by model).",
                }
            )
            seen.add(code)

    return ranked_predictions[:top_k]


def _parse_hybrid_predictions(clean_json, predictions, top_k):
    original_order = [str(c["category_code"]).strip().upper() for c in predictions]
    valid_codes = set(original_order)

    ranked_predictions = []
    selection_note = ""

    if clean_json:
        try:
            ranked_predictions = _parse_model_ranked_predictions(clean_json, valid_codes)
            ranked_predictions = _complete_with_embedding_fallback(ranked_predictions, original_order, top_k)
        except Exception:
            selection_note = "No se pudo interpretar correctamente la salida JSON del modelo."
    else:
        selection_note = "El modelo no devolvio un bloque JSON valido."

    return ranked_predictions, selection_note


def llm_predict_hybrid(text, predictions, taxonomy_df, model_name: str, top_k):

    prompt = build_prompt_hybrid(text, predictions, taxonomy_df, top_k)

    try:
        decoded = ollama_generate(prompt, model_name, num_predict=320)
    except RuntimeError:
        decoded = ""
    clean_json = extract_last_json_object(decoded)

    ranked_predictions, selection_note = _parse_hybrid_predictions(clean_json, predictions, top_k)
    if ranked_predictions:
        ranked_predictions = rerank_predictions_with_domain_bias(text, ranked_predictions, taxonomy_df)

    return ranked_predictions, selection_note, clean_json
