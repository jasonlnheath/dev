"""Custom / Ollama (local) provider profile.
Thinking capped at 2K tokens for main session, disabled for sub-agents.
"""
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

_THINKING_BUDGET_MAIN = 2048


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — thinking capped, num_ctx support."""

    def build_api_args_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        ollama_num_ctx: int | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}

        if ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = ollama_num_ctx
            extra_body["options"] = options

        if reasoning_config and reasoning_config.get("enabled") == False:
            extra_body["think"] = False
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            extra_body["think"] = True
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": True,
                "thinking_budget_tokens": _THINKING_BUDGET_MAIN,
            }

        return extra_body, {}

    def fetch_models(self, *, api_key: str | None = None, timeout: float = 8.0) -> list[str] | None:
        if not self.base_url:
            return None
        return super().fetch_models(api_key=api_key, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
    env_vars=(),
    base_url="",
)

register_provider(custom)
