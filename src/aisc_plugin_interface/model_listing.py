import json
import ssl
import urllib.error
import urllib.request
from typing import Any


class ModelListingError(RuntimeError):
    """Raised when listing models from an OpenAI-compatible endpoint fails."""


def list_openai_models(
    base_url: str,
    api_key: str | None = None,
    verify_ssl: bool = True,
) -> list[str]:
    """
    Return the model ids exposed by an OpenAI-compatible `/models` endpoint.

    ``verify_ssl`` may be disabled for endpoints using self-signed certificates
    (e.g. a local LLM gateway running in a private network).

    Raises :class:`ModelListingError` when the endpoint is unreachable or the
    credentials are rejected, so callers can fall back to a free-text model
    input in the UI.
    """
    url = f"{base_url.rstrip('/')}/v1/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    context = None if verify_ssl else ssl._create_unverified_context()

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, context=context, timeout=15) as resp:
            payload: Any = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ModelListingError(
            f"Model listing endpoint returned HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelListingError(
            f"Failed to reach model listing endpoint {url}: {exc.reason}"
        ) from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        return [item.get("id") for item in data if isinstance(item, dict) and item.get("id")]
    if isinstance(payload, list):
        return [item.get("id") for item in payload if isinstance(item, dict) and item.get("id")]
    return []
