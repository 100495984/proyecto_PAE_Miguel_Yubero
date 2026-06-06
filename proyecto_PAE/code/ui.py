import streamlit as st


# ----------------------------- SECCIÓN COLAPSABLE PARA MOSTRAR/OCULTAR --------------------------------------

def collapsible_section(title: str, key: str) -> bool:
    
    if key not in st.session_state:
        st.session_state[key] = True

    arrow = "▼" if st.session_state[key] else "▶"

    col_btn, col_title = st.columns([1, 30])

    with col_btn:
        if st.button(arrow, key=f"toggle_{key}", type="tertiary"):
            st.session_state[key] = not st.session_state[key]

    with col_title:
        if st.button(title, key=f"title_{key}", type="tertiary"):
            st.session_state[key] = not st.session_state[key]
        st.markdown(f"""
            <style>
            div[data-testid="stColumn"]:has(button[kind="tertiary"] p) button p {{
                font-size: 35px !important;
                font-weight: 700 !important;
            }}
            div[data-testid="stColumn"]:has(button[kind="tertiary"] p) button:hover {{
                color: #FAFAFA !important;
                opacity: 0.8 !important;
            }}
            button[kind="tertiary"]:hover {{
                color: #FAFAFA !important;
                background: none !important;
            }}
            </style>
        """, unsafe_allow_html=True)

    return st.session_state[key]



# ----------------------------- HELPERS DE CLASIFICACIÓN FINAL (session_state) -----------------------------
# Estos helpers gestionan la lista de categorías seleccionadas en la clasificación final, permitiendo añadir, eliminar y reordenar categorías.

def add_to_final(code: str, label: str):
    # Añade una categoría a la clasificación final si no está ya
    already = any(r["category_code"] == code for r in st.session_state.final_classification)
    if not already:
        st.session_state.final_classification.append({"category_code": code, "category_label": label})

def remove_from_final(code: str):
    #Elimina una categoría de la clasificación final
    st.session_state.final_classification = [r for r in st.session_state.final_classification if r["category_code"] != code]


def move_in_final(code: str, direction: int):
    # Mueve una categoría hacia arriba (-1) o hacia abajo (+1) en la clasificación final
    lst = st.session_state.final_classification
    idx = next((i for i, r in enumerate(lst) if r["category_code"] == code), None)
    if idx is None:
        return
    new_idx = idx + direction
    if 0 <= new_idx < len(lst):
        lst[idx], lst[new_idx] = lst[new_idx], lst[idx]
        st.session_state.final_classification = lst



# ------------------------------- TABLA TOP-K DE PREDICCIONES ---------------------------------------

def render_topk_table(df, source_key: str, score_col: str = None):
    # Renderiza una tabla interactiva con las predicciones top-k.
    final_codes = {r["category_code"] for r in st.session_state.final_classification}

    if score_col == "score":
        cols_header = st.columns([0.5, 2, 4, 2, 0.7, 0.9])
        headers = ["Rank", "Code", "Label", "Score", "Add"]
    else:
        cols_header = st.columns([0.5, 2, 4, 3, 0.7, 0.9])
        headers = ["Rank", "Code", "Label", "Reason", "Add"]

    for col, h in zip(cols_header, headers):
        col.markdown(f"**{h}**")

    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for i, row in df.iterrows():
        code = row["category_code"]
        label = row["category_label"]
        extra = row[score_col] if score_col else ""
        rank = i + 1
        in_final = code in final_codes

        if score_col == "score":
            c_rank, c_code, c_label, c_extra, c_add, c_del = st.columns([0.5, 2, 4, 2, 0.7, 0.9])
            c_extra.write(f"{extra:.4f}" if isinstance(extra, float) else extra)
        else:
            c_rank, c_code, c_label, c_extra, c_add, c_del = st.columns([0.5, 2, 4, 3, 0.7, 0.9])
            c_extra.write(str(extra)[:120] + ("…" if len(str(extra)) > 120 else ""))

        c_rank.write(str(rank))
        c_code.write(code)
        c_label.write(label)

        btn_label = "✔" if in_final else "➕"
        if c_add.button(btn_label, key=f"add_{source_key}_{code}", help="Add to final classification"):
            add_to_final(code, label)
            st.rerun()



# -------------------------------- TABLA DE CLASIFICACIÓN FINAL (con reordenación) ----------------------------------

def render_final_classification_table(final_list: list):
    # Renderiza la tabla de clasificación final con botones de reorden y borrado.
    n = len(final_list)
    final_table_container = st.container(key="final_cls_table")

    with final_table_container:
        st.markdown("""
        <style>
        [class*="st-key-final_cls_table"] button[data-testid="stBaseButton-secondary"] {
            background: transparent !important;
            border: none !important;
            padding: 0 2px !important;
            min-height: 0 !important;
            box-shadow: none !important;
            color: #777 !important;
            font-size: 11px !important;
            width: auto !important;
            min-width: 0 !important;
        }
        [class*="st-key-final_cls_table"] button[data-testid="stBaseButton-secondary"]:hover {
            color: #ffffff !important;
            background: transparent !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.caption(f"**{n} category/ies selected.** Use ▲ ▼ to reorder.")

        h_num, h_code, h_label, h_up, h_down, h_del = st.columns([0.35, 1.4, 5.5, 0.38, 0.38, 0.45])
        for col, txt in zip([h_num, h_code, h_label], ["Rank", "Code", "Label"]):
            col.markdown(f"<span style='font-size:13px;color:#ffffff;font-weight:700;"
                         f"text-transform:uppercase;letter-spacing:.06em'>{txt}</span>", unsafe_allow_html=True)
        st.markdown(
            "<hr style='margin:2px 0 2px 0;border:none;border-top:0.5px solid #333'>",unsafe_allow_html=True)

        for idx, r in enumerate(final_list):
            code = r["category_code"]
            label = r["category_label"]
            is_first = (idx == 0)
            is_last = (idx == n - 1)

            c_num, c_code, c_label, c_up, c_down, c_del = st.columns([0.35, 1.4, 5.5, 0.38, 0.38, 0.45])

            c_num.markdown(f"<span style='font-size:14px;color:#ffffff;font-weight:400;line-height:2.4'>{idx + 1}</span>", 
                        unsafe_allow_html=True)
            
            c_code.markdown(f"<span style='font-family:\"SFMono-Regular\",Consolas,monospace;"
                            f"font-size:15px;color:#ffffff;font-weight:400;line-height:2.4'>{code}</span>", unsafe_allow_html=True)
            
            c_label.markdown(f"<span style='font-size:15px;color:#ffffff;font-weight:400;line-height:2.4'>{label}</span>",
                             unsafe_allow_html=True)

            if not is_first:
                if c_up.button("▲", key=f"up_{code}", help="Move up"):
                    move_in_final(code, -1)
                    st.rerun()
            else:
                c_up.markdown("<span style='opacity:.12;font-size:13px;line-height:2.4;display:inline-block'>▲</span>", 
                              unsafe_allow_html=True)

            if not is_last:
                if c_down.button("▼", key=f"down_{code}", help="Move down"):
                    move_in_final(code, +1)
                    st.rerun()
            else:
                c_down.markdown("<span style='opacity:.12;font-size:13px;line-height:2.4;display:inline-block'>▼</span>",
                    unsafe_allow_html=True)

            if c_del.button("🗑️", key=f"final_remove_{code}", help=f"Remove {code}"):
                remove_from_final(code)
                st.rerun()

        st.markdown("<hr style='margin:4px 0 10px 0;border:none;border-top:0.5px solid #333'>", unsafe_allow_html=True)
