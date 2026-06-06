# PAE Project Classifier

Aplicacion web en Streamlit para clasificar proyectos aeronauticos con tres modos:

- Sentence Embeddings
- Prompting LLM
- Hybrid approach

## Estado actual de arquitectura (importante)

- El backend LLM de la app esta fijado en Ollama.
- Ya no se usa Hugging Face para los LLM de prompting/hybrid.
- Hugging Face se usa para modelos de Sentence Transformers (embeddings).

## Requisitos

### Software

- Python 3.10 o 3.11 (recomendado: 3.11)
- Git
- pip actualizado
- Ollama instalado

### Hardware recomendado

- Para embeddings: CPU suficiente (tambien puede usar GPU)
- Para LLM local con Ollama:
	- Modelos 3B cuantizados: >= 8 GB RAM/VRAM recomendados
	- Modelos 7B cuantizados: >= 12 GB RAM/VRAM recomendados

### CUDA y drivers (opcional)

- Si tienes GPU NVIDIA, instala drivers correctos
- El entorno actual usa ruedas CUDA 12.1 para PyTorch
- Sin GPU funciona, pero mas lento

## Instalacion del proyecto

### Windows (PowerShell)

```powershell
git clone https://github.com/tu-usuario/proyecto_PAE.git
cd proyecto_PAE/proyecto_PAE

python -m venv venv
.\venv\Scripts\Activate.ps1

cd code
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/tu-usuario/proyecto_PAE.git
cd proyecto_PAE/proyecto_PAE

python3 -m venv venv
source venv/bin/activate

cd code
pip install -r requirements.txt
```

## Instalacion y uso de Ollama

### 1) Instalar Ollama

- Web oficial: https://ollama.com

# Proyecto PAE - Clasificación de Proyectos con LLM y Embeddings

Aplicación web en Streamlit para clasificar proyectos aeronáuticos usando tres modos:

- **Sentence Embeddings**
- **Prompting LLM**
- **Hybrid approach**

---

## Requisitos

- **Python** 3.10 o 3.11 recomendado
- **pip** actualizado
- **Ollama** instalado y ejecutándose para usar modelos LLM locales (Qwen, Llama, Mistral, etc.)
- Modelos de embeddings soportados: `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`, `BAAI/bge-large-en-v1.5`

### Hardware recomendado

- Para embeddings: CPU suficiente (puede usar GPU)
- Para LLM local con Ollama:
  - Modelos 3B cuantizados: >= 8 GB RAM/VRAM
  - Modelos 7B cuantizados: >= 12 GB RAM/VRAM

### Soporte GPU (opcional)

- Instala drivers NVIDIA y versiones CUDA si tienes GPU
- El entorno soporta PyTorch con CUDA 12.1

---

## Instalación del proyecto

### Windows (PowerShell)

```powershell
git clone https://github.com/tu-usuario/proyecto_PAE.git
cd proyecto_PAE/proyecto_PAE
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/tu-usuario/proyecto_PAE.git
cd proyecto_PAE/proyecto_PAE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Instalación y uso de Ollama

1. Instala Ollama desde https://ollama.com/download
2. Descarga los modelos necesarios, por ejemplo:
	```powershell
	ollama pull qwen:7b
	ollama pull llama3
	ollama pull mistral
	```
3. Comprueba modelos descargados:
	```powershell
	ollama list
	```
4. Arranca el servicio de Ollama:
	```powershell
	ollama serve
	```
	- Si aparece error de puerto en uso (11434), Ollama ya está corriendo.
	- Solo necesitas el servicio activo, no la ventana abierta.

---

## Ejecución de la app

Con entorno virtual activado:

```powershell
streamlit run code/app.py
```

URL local por defecto: http://localhost:8501

---

## Flujo de uso en la UI

1. Subir taxonomía CSV
2. Elegir modo(s): Embeddings, Prompting LLM, Hybrid
3. Elegir modelos en Configuration:
	- Embedding model
	- LLM model (de Ollama)
4. Ajustar Top-K
5. Cargar proyecto (CSV o Manual)
6. Pulsar Analyze Project
7. Revisar y guardar clasificación

---

## Formatos de entrada CSV

### Taxonomía (obligatorio)

Columnas mínimas:

- category_code
- category_label

Recomendado:
- text (descripción larga de categoría)

### Proyectos (modo From CSV)

Columnas mínimas:

- title
- description

Columnas gestionadas por la app:

- project_id
- labels
- timestamp

---

## Comportamiento de cada modo

### Sentence Embeddings

- Calcula similitud semántica con la taxonomía
- Devuelve top-k por score de similitud

### Prompting LLM (modo estricto)

- Prompting puro sobre taxonomía (sin usar embeddings para decidir resultado)
- Normaliza y valida códigos devueltos por el LLM
- Si el LLM no devuelve suficientes códigos válidos, puede mostrar menos de Top-K
- Esto es intencional para priorizar calidad frente a relleno artificial

### Hybrid approach

- Usa embeddings para shortlist inicial
- LLM reordena y justifica candidatos

---

## Errores y mensajes habituales

### "Could not reach Ollama"

- Revisa que Ollama esté activo en `http://localhost:11434`
- Comprueba con:
  ```powershell
  Invoke-WebRequest -Uri http://127.0.0.1:11434/api/tags -UseBasicParsing
  ```

### "HTTP Error 404" al consultar Ollama

- El modelo indicado no existe localmente
- Comprueba nombre exacto con `ollama list`

### "El modelo respondió pero no devolvió códigos válidos"

- El LLM respondió en formato no útil o con códigos no válidos
- En prompting estricto, la app evita rellenar con ruido

### Warning de HF_TOKEN (Hugging Face)

- Mensaje esperado en embeddings si no hay autenticación de Hugging Face
- No bloquea el funcionamiento normal

---

## Dependencias y versiones recomendadas

Las dependencias principales están en `requirements.txt`.

- Python >=3.10, <3.12
- torch >=2.0.0
- sentence-transformers >=2.2.2
- streamlit >=1.25.0
- ollama-client >=0.1.0

> Si usas GPU, instala torch y sentence-transformers con la versión adecuada para tu hardware.

---

## Notas sobre caché y reproducibilidad

- El sistema sobrescribe automáticamente el archivo de caché de embeddings por modelo (por ejemplo, `all-MiniLM-L6-v2.npz`).
- Si cambias la taxonomía o el dataset, se regeneran los embeddings y se sobrescribe el archivo.
- Si tienes problemas con dependencias, revisa las versiones en `requirements.txt`.

---

## Recomendaciones prácticas

- Usa `qwen:7b` o `llama3` para iterar rápido
- Usa `mistral` si quieres más calidad y tienes recursos
- Para comparaciones justas, analiza el mismo proyecto con los tres modos y el mismo Top-K

---

## Contacto y soporte

Para dudas o problemas, contacta con el responsable del proyecto o abre un issue en el repositorio.
