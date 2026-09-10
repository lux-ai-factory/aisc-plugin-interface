import json
import urllib.error
import urllib.request
from typing import Any


class OpenAICompatibleClientError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint call fails."""


class _Completions:
    """Namespace mirroring `openai.resources.chat.completions`."""

    def __init__(self, client: "OpenAICompatibleClient"):
        self._client = client

    def create(self, **kwargs: Any) -> dict[str, Any]:
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        if model is None or messages is None:
            raise OpenAICompatibleClientError("'model' and 'messages' are required")
        body: dict[str, Any] = {"model": model, "messages": messages}
        for key in ("temperature", "max_tokens", "max_completion_tokens", "top_p",
                    "stream", "tools", "tool_choice", "response_format"):
            value = kwargs.get(key)
            if value is not None:
                body[key] = value
        return self._client._request("chat/completions", body)


class OpenAICompatibleClient:
    """
    Minimal client for OpenAI-compatible `/chat/completions` endpoints.

    Uses the standard library by default. If the `openai` package is installed
    it is preferred, since it exposes the full feature set (tools, streaming,
    async). Both paths expose an `openai`-style `chat.completions` namespace.

    Example:
        client = OpenAICompatibleClient("https://api.example.com/v1", "sk-...")
        reply = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._sdk = None

        try:
            import openai  # type: ignore

            self._sdk = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key or "missing",
            )
        except ImportError:
            self._sdk = None

    @property
    def chat(self):
        """Accessor mirroring the `openai` SDK's `client.chat` namespace."""
        if self._sdk is not None:
            return self._sdk.chat
        return self

    @property
    def completions(self) -> Any:
        """Namespace mirroring `openai.resources.chat.completions` (stdlib path)."""
        return _Completions(self)

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OpenAICompatibleClientError(
                f"OpenAI-compatible endpoint returned HTTP {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAICompatibleClientError(
                f"Failed to reach OpenAI-compatible endpoint {url}: {exc.reason}"
            ) from exc
