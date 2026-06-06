import os
import csv
import datetime
import pandas as pd
import streamlit as st


# ---------------------- GUARDADO EN CSV DE PROYECTOS ----------------------------------
# Las clasificaciones se almacenan directamente en el CSV de proyectos,
# en las columnas "labels" (códigos separados por comas) y "timestamp".

def ensure_classification_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Asegura que el DataFrame tiene las columnas "labels" y "timestamp"
    if "labels" not in df.columns:
        df["labels"] = ""
    if "timestamp" not in df.columns:
        df["timestamp"] = ""
    return df


def save_classification_to_projects_csv(proj_csv_path: str, project_id: str, final_list: list):
    # Guarda la clasificación de un proyecto (solo códigos, separados por comas)
    # directamente en el CSV de proyectos, en las columnas "labels" y "timestamp".
    df = pd.read_csv(proj_csv_path, dtype=str)
    df = ensure_classification_columns(df)

    labels_str = ", ".join(r["category_code"] for r in final_list)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    mask = df["project_id"] == str(project_id)
    if mask.any():
        df.loc[mask, "labels"] = labels_str
        df.loc[mask, "timestamp"] = timestamp
    else:
        new_row = {col: "" for col in df.columns}
        new_row["project_id"] = str(project_id)
        new_row["labels"] = labels_str
        new_row["timestamp"] = timestamp
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.to_csv(proj_csv_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


def load_classification_from_projects_df(projects_df: pd.DataFrame, project_id: str, taxonomy_df: pd.DataFrame) -> list:
    # Lee la clasificación guardada de un proyecto desde el DataFrame de proyectos.
    # Devuelve una lista de dicts {category_code, category_label} o [] si no hay nada.
    if "labels" not in projects_df.columns:
        return []

    mask = projects_df["project_id"] == str(project_id)
    if not mask.any():
        return []

    labels_str = projects_df.loc[mask, "labels"].values[0]
    if pd.isna(labels_str) or str(labels_str).strip() == "":
        return []

    codes = [c.strip() for c in str(labels_str).split(",") if c.strip()]
    result = []
    for code in codes:
        match = taxonomy_df[taxonomy_df["category_code"] == code]
        label = match["category_label"].values[0] if not match.empty else code
        result.append({"category_code": code, "category_label": label})
    return result


def project_is_classified(projects_df: pd.DataFrame, project_id: str) -> bool:
    # Devuelve True si el proyecto ya tiene una clasificación guardada
    if "labels" not in projects_df.columns:
        return False
    mask = projects_df["project_id"] == str(project_id)
    if not mask.any():
        return False
    val = projects_df.loc[mask, "labels"].values[0]
    return not (pd.isna(val) or str(val).strip() == "")