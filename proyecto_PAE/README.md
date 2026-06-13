# PAE Project Classifier
Aplicación web en Streamlit para clasificar proyectos aeronáuticos con tres modos: **Sentence Embeddings**, **Prompting LLM** e **Hybrid approach**.

---

## Requisitos

- Python 3.10 o 3.11
- [Ollama](https://ollama.com/download) instalado y en ejecución

### Hardware recomendado para LLM

- Modelos 3B: >= 8 GB RAM
- Modelos 7B: >= 12 GB RAM

---

## Instalación

### 1. Clonar el repositorio y crear entorno virtual

**Windows (PowerShell)**
```powershell
git clone <url-del-repo>
cd proyecto_PAE
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**
```bash
git clone <url-del-repo>
cd proyecto_PAE
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Descargar modelos de Ollama

```bash
ollama pull mistral:7b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

---

## Ejecución

Con el entorno virtual activado:

```bash
streamlit run code/app.py
```

La app abre en `http://localhost:8501`.

> Ollama debe estar corriendo antes de lanzar la app. Si `ollama serve` da error de puerto en uso, ya está activo.

---

## Uso básico

1. Subir taxonomía CSV (`category_code`, `category_label`)
2. Elegir modo(s) de análisis
3. Configurar modelos y Top-K
4. Cargar proyecto (CSV o manual)
5. Pulsar **Analyze Project**

---

## Reglas de dominio

Las reglas que el sistema aplica sobre el LLM para clasificar proyectos aeroespaciales están documentadas en [prompt_rules.md](prompt_rules.md). Incluye instrucciones por fase del pipeline (perfil del proyecto, evaluación de familias, selección de categorías, etc.).
