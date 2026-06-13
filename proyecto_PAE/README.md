# PAE Project Classifier

Aplicación web en Streamlit para clasificar proyectos aeronáuticos usando tres modos: Sentence Embeddings, Prompting LLM e Hybrid approach.

---

## Requisitos

Para ejecutar la app necesitas Python 3.10 o 3.11 (no versiones superiores, ya que algunas dependencias como PyTorch no son compatibles todavía) y [Ollama](https://ollama.com/download) instalado y en ejecución si quieres usar los modos LLM o Hybrid.

En cuanto al hardware, los modelos de 3B parámetros funcionan con 8 GB de RAM. Para los de 7B se recomiendan al menos 12 GB.

---

## Instalación

Clona el repositorio y crea un entorno virtual con Python 3.11. Es importante usar explícitamente esa versión para evitar problemas de compatibilidad.

En Windows:
```powershell
git clone <url-del-repo>
cd proyecto_PAE
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux o macOS:
```bash
git clone <url-del-repo>
cd proyecto_PAE
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Si vas a usar los modos LLM o Hybrid, descarga al menos uno de los modelos de Ollama:
```bash
ollama pull mistral:7b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

Si ya tienes Ollama instalado de antes, los modelos son globales al sistema y no hace falta volver a descargarlos. Puedes comprobarlo con `ollama list`.

---

## Ejecución

Con el entorno virtual activado, lanza la app desde la raíz del proyecto:

```bash
streamlit run code/app.py
```

La app abre en `http://localhost:8501`. Ollama tiene que estar corriendo antes de lanzarla si usas LLM o Hybrid. Si `ollama serve` da error de puerto en uso, es que ya está activo.

La primera vez que uses el modo Sentence Embeddings, la app descargará automáticamente el modelo de embeddings desde HuggingFace. Puede tardar unos minutos dependiendo de la conexión. Las siguientes ejecuciones lo cargan desde caché y arrancan de forma inmediata.

---

## Uso básico

El flujo normal de la app es: subir la taxonomía en CSV (con columnas `category_code` y `category_label`), elegir el modo o modos de análisis, configurar los modelos y el Top-K, cargar el proyecto a clasificar (desde CSV o introduciéndolo manualmente) y pulsar Analyze Project.

---

## Datos de ejemplo

En la carpeta `inputs/` hay dos archivos listos para usar directamente con la app.

`aerotax_transformado.csv` es la taxonomía aeronáutica que se sube en el primer paso. Contiene los códigos y etiquetas de categoría junto con una descripción extendida de cada una.

`pae_aero_corpus.csv` es un corpus de proyectos aeronáuticos ya etiquetados que se puede usar para probar la app. Cada fila es un proyecto con su título, descripción y las categorías asignadas.

---

## Estructura del código

Todo el código de la app está en la carpeta `code/`. El punto de entrada es `app.py`, que orquesta el flujo principal y conecta el resto de módulos.

`embeddings.py` implementa el modo Sentence Embeddings: genera embeddings del proyecto y de la taxonomía, calcula similitud coseno y devuelve las categorías más cercanas.

`prompting.py` implementa el modo Prompting LLM: construye los prompts por fases y llama al modelo via Ollama para que seleccione las categorías más relevantes.

`hybrid.py` combina los resultados de los dos modos anteriores aplicando un re-ranking final.

`ollama_client.py` es el cliente HTTP que se comunica con la API local de Ollama.

`csv_manager.py` gestiona la lectura y escritura de proyectos en CSV, incluyendo el guardado de etiquetas y timestamps.

`ranking_utils.py` contiene las utilidades de re-ranking: sesgo de dominio, tokenización y scoring de candidatos.

`ui.py` tiene componentes de interfaz reutilizables.

La carpeta `taxonomy_cache/` almacena los embeddings de la taxonomía ya calculados para no tener que recomputarlos en cada sesión.

---

## Reglas de dominio

Las reglas que el sistema aplica para clasificar proyectos están documentadas en [prompt_rules.md](prompt_rules.md). Incluye las instrucciones por fase del pipeline: perfil del proyecto, evaluación de familias, selección de categorías, etc.
