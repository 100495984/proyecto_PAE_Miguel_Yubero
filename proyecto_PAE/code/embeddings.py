import hashlib
import os
import logging
import warnings
import numpy as np
import streamlit as st
import torch

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings(
    "ignore",
    message=r"Accessing `__path__` from `\.models\..*",
)
warnings.filterwarnings(
    "ignore",
    message=r"You are sending unauthenticated requests to the HF Hub.*",
)

from sentence_transformers import SentenceTransformer, util

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from ranking_utils import build_project_query, score_taxonomy_row


# Directorio donde se guardan los embeddings precalculados de la taxonomía.
# Se crea automáticamente si no existe.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxonomy_cache")
os.makedirs(CACHE_DIR, exist_ok=True)



# Carga el modelo SBERT una sola vez por sesión (Streamlit cache_resource).
@st.cache_resource(show_spinner=False)
def load_sbert(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")



# Genera una clave única (hash) que identifica la combinación
# taxonomía + modelo. Si cambia cualquiera de los dos, el hash cambia
# y se recalculan y guardan los embeddings automáticamente.
def _taxonomy_cache_key(tax_df, model_name: str) -> str:
    # Serializamos el DataFrame como CSV en memoria (sin índice) para hashear
    csv_bytes = tax_df.to_csv(index=False).encode("utf-8")
    model_bytes = model_name.encode("utf-8")
    return hashlib.sha256(csv_bytes + model_bytes).hexdigest()


# Construye la ruta del archivo de caché a partir de la clave y el nombre del modelo.
def _cache_path(cache_key: str, model_name: str = "") -> str:
    # Usar el nombre del modelo y el hash para el archivo de caché
    short_name = model_name.split("/")[-1] if model_name else "cache"
    filename = f"{short_name}__{cache_key[:8]}.npz"
    return os.path.join(CACHE_DIR, filename)


# Construye los strings que representan cada categoría de la taxonomía.
# Se separa en su propia función para que sea fácil ajustar el formato
# sin tocar el resto de la lógica.
def _build_tax_strings(tax_df) -> list[str]:
    return [
        f"{r['category_code']} - {r['category_label']}. {r['text']}"
        for _, r in tax_df.iterrows()
    ]



# Devuelve (tax_strings, tax_emb) intentando leer primero del disco.
# Si no existe la caché, calcula los embeddings, los guarda y los devuelve.
@st.cache_resource(show_spinner=False)
def _get_taxonomy_embeddings(cache_key: str, model_name: str, tax_df):
    path = _cache_path(cache_key, model_name)


    # Intento de lectura desde disco
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        tax_strings = data["tax_strings"].tolist()
        tax_emb_np  = data["tax_emb"]              
        tax_emb     = torch.tensor(tax_emb_np)
        return tax_strings, tax_emb

    # No existe caché: calcular y guardar
    model      = load_sbert(model_name)
    tax_strings = _build_tax_strings(tax_df)
    tax_emb    = model.encode(
        tax_strings,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


    # Eliminar archivos antiguos del mismo modelo (sin hash) para evitar duplicados
    short_name = model_name.split("/")[-1] if model_name else "cache"
    old_path = os.path.join(CACHE_DIR, f"{short_name}.npz")
    if os.path.exists(old_path):
        os.remove(old_path)

    # Guardamos en disco: strings como array de objetos, embeddings como float32
    np.savez_compressed(
        path,
        tax_strings=np.array(tax_strings, dtype=object),
        tax_emb=tax_emb.cpu().numpy().astype(np.float32),
    )

    return tax_strings, tax_emb



# Función para obtener los embeddings de la taxonomía con caché automática.
def compute_taxonomy_embeddings(tax_df, model_name: str):
    cache_key = _taxonomy_cache_key(tax_df, model_name)
    return _get_taxonomy_embeddings(cache_key, model_name, tax_df)


# Función para obtener las top-k categorías más similares a un texto dado.
def embeddings_topk(text: str, tax_df, model_name: str, top_k: int) -> list[dict]:
    model = load_sbert(model_name)

    _, tax_emb = compute_taxonomy_embeddings(tax_df, model_name)

    # Embedding del proyecto con prefijo de instrucción para modelos bi-encoder
    query_text = build_project_query(text)
    proj_emb = model.encode(
        ["Represent this aerospace research project for classification: " + query_text],
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    sims = util.cos_sim(proj_emb, tax_emb)[0].cpu().numpy()
    lexical_boost = np.array([score_taxonomy_row(text, row) for _, row in tax_df.iterrows()], dtype=np.float32)
    if lexical_boost.size:
        max_boost = float(lexical_boost.max())
        if max_boost > 0:
            lexical_boost = lexical_boost / max_boost
        sims = (0.85 * sims) + (0.15 * lexical_boost)

    idx  = np.argsort(sims)[-top_k:][::-1]

    return [
        {
            "category_code":  tax_df.iloc[i]["category_code"],
            "category_label": tax_df.iloc[i]["category_label"],
            "score":          float(sims[i]),
        }
        for i in idx
    ]
