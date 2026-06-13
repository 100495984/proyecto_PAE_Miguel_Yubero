import os
import time
import streamlit as st
import pandas as pd
import torch
import gc
import streamlit.components.v1 as components  # noqa: F401 (kept for potential future use)

from embeddings import embeddings_topk
from prompting import llm_predict_prompting
from hybrid import llm_predict_hybrid

from csv_manager import (save_classification_to_projects_csv, load_classification_from_projects_df, project_is_classified,)
from ui import (collapsible_section, render_topk_table, render_final_classification_table, add_to_final, move_in_final,)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ---------------------- NORMALIZACIÓN DE PROYECTOS ----------------------------------

def normalize_projects(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "project_id" not in df.columns:
        df["project_id"] = [f"P{i+1:05d}" for i in range(len(df))]
    df["full_text"] = df["title"].str.strip() + "\n\n" + df["description"].str.strip()
    df["display_title"] = df["project_id"] + " — " + df["title"].str[:80]
    return df



# ------------------------- CONFIGURACIÓN DE PÁGINA ------------------------------------

st.set_page_config(page_title="PAE Classifier", layout="wide")

st.markdown("""<h1 style='text-align: center;'>🚀 PAE Project Classifier</h1>""",
            unsafe_allow_html=True)
st.markdown("---")

section = "Clasificador"



# ------------------------------- SECCIÓN PRINCIPAL: CLASIFICADOR ----------------------------------

if section == "Clasificador":

    # Toast de confirmación tras rerun (aparece arriba a la derecha y desaparece solo)
    if st.session_state.get("_just_saved_pid"):
        pid = st.session_state.pop("_just_saved_pid")
        st.toast(f"✅ Project {pid} successfully saved!")
        time.sleep(0.3)


    # ---------------------- CARGA DE INPUTS (UPLOAD DATA) --------------------------------

    st.markdown('<div id="upload-data-anchor"></div>', unsafe_allow_html=True)
    visible = collapsible_section("📂 Upload data", "show_upload")

    upload_container = st.container(key="upload_data_container")
    with upload_container:
        tax_file = st.file_uploader("Taxonomy CSV", type=["csv"], key="tax_file")

    if not visible:
        st.markdown("""
            <style>
            [class*="st-key-upload_data_container"] div[data-testid="stFileUploader"] { display: none; }
            </style>
        """, unsafe_allow_html=True)

    if st.session_state.get("tax_file") is None:
        st.warning("⚠️ Please upload the taxonomy to continue.")
        st.stop()



    # --------------------------- CARGA Y VALIDACIÓN DEL DATASET DE LA TAXONOMÍA -----------------------------------------

    tax_raw = pd.read_csv(tax_file)
    TAX_REQUIRED = {"category_code", "category_label"}
    if not TAX_REQUIRED.issubset(tax_raw.columns):
        st.error(f"⚠️ The taxonomy CSV is missing required columns: " f"{TAX_REQUIRED - set(tax_raw.columns)}. " 
                 f"Found: {list(tax_raw.columns)}. " f"Have you uploaded the wrong file?")
        st.stop()
    taxonomy_df = tax_raw.sort_values("category_code").reset_index(drop=True)

    projects_df = None


    st.markdown("---")



    # ---------------------- SELECT ANALYSIS TYPE ----------------------------------

    visible_modes = collapsible_section("🔧 Select analysis type", "show_modes")

    selected_modes = st.multiselect( 
        "Select analysis types:", ["Sentence Embeddings", "Prompting LLM", "Hybrid approach"], key="selected_modes")

    if not visible_modes:
        st.markdown("""
            <style>
            div[data-testid="stMultiSelect"] { display: none; }
            </style>
        """, unsafe_allow_html=True)

    selected_modes = st.session_state.get("selected_modes", [])
    if len(selected_modes) == 0:
        st.info("Select at least one analysis type to start.")
        st.stop()

    st.markdown("---")


    # ----------------------------- CONFIGURATION ----------------------------------------

    visible_config = collapsible_section("⚙ Configuration", "show_config")
    config_container = st.container(key="config_container")

    with config_container:

        st.markdown("### 🤖 Model selection")
        llm_options_prompting = ["mistral:7b", "qwen2.5:3b", "llama3.2:3b"]
        llm_options_default = ["qwen2.5:3b", "llama3.2:3b", "mistral:7b"]

        if len(selected_modes) > 1:
            col1, col2 = st.columns(2)
            with col1:
                embedding_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5",
                                                                   "sentence-transformers/all-mpnet-base-v2"], key="embedding_model")
            with col2:
                llm_choices = llm_options_prompting if "Prompting LLM" in selected_modes else llm_options_default
                llm_model = st.selectbox("LLM model (Ollama)", llm_choices, key="llm_model")
        else:
            if "Sentence Embeddings" in selected_modes:
                embedding_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5",
                                                                   "sentence-transformers/all-mpnet-base-v2"], key="embedding_model")
            elif "Prompting LLM" in selected_modes:
                llm_model = st.selectbox("LLM model (Ollama)", llm_options_prompting, key="llm_model")
            elif "Hybrid approach" in selected_modes:
                col1, col2 = st.columns(2)
                with col1:
                    embedding_model = st.selectbox("Embedding model", ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5",
                                                                        "sentence-transformers/all-mpnet-base-v2"], key="embedding_model")
                with col2:
                    llm_model = st.selectbox("LLM model (Ollama)", llm_options_default, key="llm_model")


        st.markdown("### 🎯 Prediction settings")
        slider_col, _ = st.columns([3, 2])
        with slider_col:
            top_k = st.slider("Number of categories to assign (Top-K)", 1, 10, 5, key="top_k")
        st.caption("Select how many categories will be assigned to each project.")

    if not visible_config:
        st.markdown("""
            <style>
            [class*="st-key-config_container"] { display: none !important; }
            </style>
        """, unsafe_allow_html=True)

    st.markdown("---")



    # ------------------------------ MODO DE CLASIFICACIÓN Y SELECCIÓN DE PROYECTO -----------------------------------------

    visible_classification = collapsible_section("📄 Classification", "show_classification")
    classification_container = st.container()

    with classification_container:

        input_mode = st.radio("Select mode:", ["From CSV", "Manual"], index=0, key="input_mode")

        if input_mode == "Manual":
            title = st.text_input("Project Title")
            description = st.text_area("Project Description", height=200)
            text = (title.strip() + "\n\n" + description.strip()).strip() \
                if title.strip() and description.strip() else ""

        else:
            proj_file = st.file_uploader("Projects CSV", type=["csv"], key="proj_file")

            # VALIDACIÓN BÁSICA DEL CSV DE PROYECTOS
            if proj_file:
                # Guardar el CSV en inputs/ para poder sobreescribirlo al guardar clasificaciones
                INPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inputs")
                os.makedirs(INPUTS_DIR, exist_ok=True)
                proj_csv_path = os.path.join(INPUTS_DIR, proj_file.name)
                if st.session_state.get("proj_csv_path") != proj_csv_path or not os.path.exists(proj_csv_path):
                    proj_bytes = proj_file.read()
                    with open(proj_csv_path, "wb") as f:
                        f.write(proj_bytes)
                    proj_file.seek(0)
                st.session_state["proj_csv_path"] = proj_csv_path
            else:
                # El uploader se ha vaciado: limpiar estado para que no se pueda seguir usando
                st.session_state.pop("proj_csv_path", None)

            proj_csv_path = st.session_state.get("proj_csv_path", "")
            if proj_csv_path and os.path.exists(proj_csv_path):
                proj_raw = pd.read_csv(proj_csv_path, dtype=str)
                PROJ_REQUIRED = {"title", "description"}
                if not PROJ_REQUIRED.issubset(proj_raw.columns):
                    st.error(f"⚠️ The projects CSV is missing required columns: "
                             f"{PROJ_REQUIRED - set(proj_raw.columns)}. "
                             f"Found: {list(proj_raw.columns)}. "
                             f"Have you uploaded the wrong file?")
                    st.stop()
                projects_df = normalize_projects(proj_raw)
            else:
                projects_df = None
 
            if projects_df is None:
                text = ""
            else:
                labels = projects_df["display_title"].tolist()
                sel = st.selectbox("Select project", labels)
                row = projects_df.iloc[labels.index(sel)]
                text = row["full_text"]
 
                selected_pid = row["project_id"]
                if st.session_state.get("active_project_id") != selected_pid:
                    st.session_state["active_project_id"] = selected_pid
                    st.session_state["active_project_title"] = row["title"]
                    for k in ["embeddings_results", "prompting_results", "hybrid_results"]:
                        st.session_state.pop(k, None)
                    # Cargar clasificación previa si existe
                    saved = load_classification_from_projects_df(projects_df, selected_pid, taxonomy_df)
                    st.session_state.final_classification = saved
 
                st.markdown("### Project Title")
                st.markdown(f"**{row['title']}**")
                st.markdown("### Project Description")
                st.write(text)

                # Indicador visual de si el proyecto ya está clasificado
                if project_is_classified(projects_df, selected_pid):
                    st.success(f"✅ This project already has a saved classification.")

    if not visible_classification:
        st.markdown("""
            <style>
            div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stRadio"]) {
                display: none;
            }
            </style>
        """, unsafe_allow_html=True)

    if visible_classification:
        if input_mode == "Manual":
            if not title.strip() or not description.strip():
                st.info("Both title and description are required to analyze the project.")
                st.stop()
        elif input_mode == "From CSV" and projects_df is None:
            st.stop()
    elif not text:
        st.stop()

    st.markdown("---")




    # -------------------------------- ANÁLISIS DEL PROYECTO ----------------------------------------------

    if st.button("🚀 Analyze Project"):
        for k in ["embeddings_results", "prompting_results", "hybrid_results",
                  "show_embeddings_table", "show_prompting_table", "show_hybrid_table"]:
            st.session_state.pop(k, None)
        st.session_state.final_classification = []
        st.session_state["_do_analyze"] = True
        st.rerun()

    if st.session_state.get("_do_analyze"):
        st.session_state.pop("_do_analyze")
        with st.spinner("⏳ Analyzing project... please wait"):

            st.session_state.final_classification = []

            if input_mode == "Manual":
                st.session_state["active_project_id"] = title.strip()[:20].replace(" ", "_") if title.strip() else "manual"
                st.session_state["active_project_title"] = title.strip()

            # Embeddings
            if "Sentence Embeddings" in selected_modes:
                predictions = embeddings_topk(text, taxonomy_df, embedding_model, top_k)
                ranked_df = pd.DataFrame(predictions)
                ranked_df = ranked_df.merge(taxonomy_df[["category_code", "category_label"]], 
                                            on="category_code", how="left", suffixes=("", "_tax"))
                if "category_label_tax" in ranked_df.columns:
                    ranked_df["category_label"] = ranked_df["category_label_tax"].fillna(ranked_df.get("category_label", ""))
                    ranked_df = ranked_df.drop(columns=["category_label_tax"])
                ranked_df = ranked_df[["category_code", "category_label", "score"]]
                st.session_state.embeddings_results = ranked_df
                st.session_state.show_embeddings_table = True

            # Prompting
            if "Prompting LLM" in selected_modes:
                gc.collect()
                ranked_predictions, _selection_note, _raw_output = llm_predict_prompting(text, taxonomy_df, llm_model, top_k)
                ranked_df = pd.DataFrame(ranked_predictions)
                if ranked_df.empty:
                    ranked_df = pd.DataFrame(columns=["category_code", "category_label", "reason"])
                else:
                    ranked_df = ranked_df.merge(taxonomy_df[["category_code", "category_label"]], on="category_code", how="left")
                    ranked_df = ranked_df[["category_code", "category_label", "reason"]]
                st.session_state.prompting_results = ranked_df
                st.session_state.show_prompting_table = True

            # Hybrid
            if "Hybrid approach" in selected_modes:
                predictions = embeddings_topk(text, taxonomy_df, embedding_model, top_k)
                emb_df = pd.DataFrame(predictions)
                emb_df = emb_df.merge(taxonomy_df[["category_code", "category_label"]], on="category_code", how="left", suffixes=("", "_tax"))
                if "category_label_tax" in emb_df.columns:
                    emb_df["category_label"] = emb_df["category_label_tax"].fillna(emb_df.get("category_label", ""))
                    emb_df = emb_df.drop(columns=["category_label_tax"])
                emb_df = emb_df[["category_code", "category_label", "score"]]
                gc.collect()
                ranked_predictions, selection_note, raw_output = llm_predict_hybrid(text, predictions, taxonomy_df, llm_model, top_k)
                ranked_df = pd.DataFrame(ranked_predictions)
                ranked_df = ranked_df.merge(taxonomy_df[["category_code", "category_label"]], on="category_code", how="left")
                ranked_df = ranked_df[["category_code", "category_label", "reason"]]
                st.session_state.hybrid_results = ranked_df
                st.session_state.show_hybrid_table = True



    # ------------------------- MOSTRAR RESULTADOS DE PREDICCIONES -----------------------------------

    active_pid = st.session_state.get("active_project_id", "")

    if "embeddings_results" in st.session_state:
        st.markdown("---")
        embeddings_count = len(st.session_state.embeddings_results.index)
        visible = collapsible_section(f"Top {embeddings_count} predictions (Sentence Embeddings) · {active_pid}", "show_embeddings_table")
        if visible:
            render_topk_table(st.session_state.embeddings_results.reset_index(drop=True), "emb", score_col="score")

    if "prompting_results" in st.session_state:
        st.markdown("---")
        prompting_count = len(st.session_state.prompting_results.index)
        visible = collapsible_section(f"Top {prompting_count} predictions (Prompting LLM) · {active_pid}", "show_prompting_table")
        if visible:
            render_topk_table(st.session_state.prompting_results.reset_index(drop=True), "prom", score_col="reason")
            if 0 < prompting_count < top_k:
                st.caption(f"Only {prompting_count} categories were returned, so fewer than the requested top {top_k} are shown.")
            if st.session_state.prompting_results.empty:
                st.caption("The prompting model did not return valid categories for this project.")
    if "hybrid_results" in st.session_state:
        st.markdown("---")
        hybrid_count = len(st.session_state.hybrid_results.index)
        visible = collapsible_section(f"Top {hybrid_count} predictions (Hybrid approach) · {active_pid}", "show_hybrid_table")
        if visible:
            render_topk_table(st.session_state.hybrid_results.reset_index(drop=True), "hyb", score_col="reason")


    
    # --------------------------- CLASIFICACIÓN FINAL-------------------------------------

    if "final_classification" not in st.session_state:
        st.session_state.final_classification = []

    any_results = any(k in st.session_state for k in ["embeddings_results", "prompting_results", "hybrid_results"])
    has_saved_classification = len(st.session_state.get("final_classification", [])) > 0

    if any_results or has_saved_classification:

        st.markdown("<hr style='margin:8px 0;border:none;border-top:0.5px solid #333'>", unsafe_allow_html=True)

        visible_final = collapsible_section("📋 Final Classification", "show_final_table")

        if visible_final:
            final_list = st.session_state.final_classification



            # --------------------- BÚSQUEDA OPCIONAL EN CSV DE TAXONOMÍA ----------------------------------

            with st.expander("🔍 Search and add a category from the full taxonomy"):
                search_query = st.text_input("Search taxonomy", placeholder="Type to filter by code or label…", key="taxonomy_search", 
                                             label_visibility="collapsed")
                if search_query.strip():
                    mask = (taxonomy_df["category_code"].str.contains(search_query, case=False, na=False) | 
                            taxonomy_df["category_label"].str.contains(search_query, case=False, na=False))
                    filtered_tax = taxonomy_df[mask]
                else:
                    filtered_tax = taxonomy_df

                tax_options = [f"{row['category_code']} — {row['category_label']}" for _, row in filtered_tax.iterrows()]
                if tax_options:
                    sel_col, add_col = st.columns([5, 1])
                    with sel_col:
                        selected_tax = st.selectbox("Select category", tax_options, key="taxonomy_selectbox", label_visibility="collapsed")
                    with add_col:
                        if st.button("➕ Add", key="add_from_taxonomy"):
                            sel_code = selected_tax.split(" — ")[0].strip()
                            sel_label = " — ".join(selected_tax.split(" — ")[1:]).strip()
                            add_to_final(sel_code, sel_label)
                            st.rerun()
                else:
                    st.caption("No categories match your search.")

            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)



            # ----------------------------- TABLA DE CLASIFICACIÓN FINAL ---------------------------------------

            if not final_list:
                st.info("No categories added yet. Use the ➕ button in the results tables above, or search the taxonomy above.")
            else:
                render_final_classification_table(final_list)


            #  ------------- Botones de acción: Guardar y Limpiar ---------------------------------

            st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)

            current_project_id = st.session_state.get("active_project_id", "unknown")
            save_disabled = len(final_list) == 0

            st.markdown("""
                <style>
                [class*="st-key-action_bar"] [data-testid="stHorizontalBlock"] { gap: 6px !important; }
                [class*="st-key-action_bar"] [data-testid="stColumn"] {
                    padding: 0 !important;
                    width: fit-content !important;
                    flex: unset !important;
                }
                [class*="st-key-action_bar"] button,
                [class*="st-key-action_bar"] [data-testid="stDownloadButton"] button {
                    width: fit-content !important;
                    white-space: nowrap !important;
                }
                </style>
            """, unsafe_allow_html=True)

            action_bar = st.container(key="action_bar")
            with action_bar:
                btn_c1, btn_c2 = st.columns([1.6, 1.5])
                with btn_c1:
                    proj_csv_path = st.session_state.get("proj_csv_path", "")
                    can_save = not save_disabled and bool(proj_csv_path) and os.path.exists(proj_csv_path)
                    if st.button("💾 Save classification", type="primary", disabled=not can_save):
                        saving_placeholder = st.empty()
                        saving_placeholder.markdown("⏳ Saving classification... please wait")
                        save_classification_to_projects_csv(proj_csv_path, current_project_id, final_list)
                        st.session_state["force_reload_projects"] = True
                        saving_placeholder.empty()
                        st.session_state["_just_saved_pid"] = current_project_id
                        st.rerun()
                with btn_c2:
                    if st.button("🗑️ Clear classification", type="secondary"):
                        st.session_state.final_classification = []
                        st.rerun()
