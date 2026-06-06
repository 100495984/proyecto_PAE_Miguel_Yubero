import json
import urllib.error
import urllib.request


DEFAULT_OLLAMA_URL = "http://localhost:11434"


RANKED_PREDICTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category_code": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["category_code", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ranked_predictions"],
    "additionalProperties": False,
}



class OllamaError(RuntimeError):
    pass

# Wrapper para compatibilidad con prompting.py
def ollama_generate(prompt: str, model: str, num_predict: int = 384, base_url: str = DEFAULT_OLLAMA_URL, temperature: float = 0.0, timeout: int = 300, response_format: dict | str = "json") -> str:
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(
        model=model,
        messages=messages,
        base_url=base_url,
        num_predict=num_predict,
        temperature=temperature,
        timeout=timeout,
        response_format=response_format,
    )


def list_local_models(base_url: str = DEFAULT_OLLAMA_URL) -> list[str]:
    endpoint = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(endpoint, method="GET")

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"Could not reach Ollama at {base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned a malformed model list.") from exc

    models = body.get("models", [])
    return [str(item.get("name", "")).strip() for item in models if str(item.get("name", "")).strip()]


def chat_completion(
    model: str,
    messages: list[dict],
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    num_predict: int = 384,
    temperature: float = 0.0,
    timeout: int = 300,
    response_format: dict | str = "json",
) -> str:
    if not model:
        raise OllamaError("Ollama model name is required.")

    endpoint = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": response_format,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        if exc.code == 404:
            raise OllamaError(
                f"Ollama returned 404 for model '{model}'. Check that the model exists in 'ollama list'. {detail}"
            ) from exc
        raise OllamaError(f"Ollama HTTP error {exc.code} at {base_url}: {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"Could not reach Ollama at {base_url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned a malformed JSON response.") from exc

    content = body.get("message", {}).get("content", "")
    if not content:
        raise OllamaError("Ollama returned an empty completion.")

    return content