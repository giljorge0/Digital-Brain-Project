"""
Multi-LLM Provider Layer
------------------------
Supports multiple providers AND multiple named accounts per provider.
Each account is a separate API key / endpoint config stored in the
profile registry.

Supported providers:
  claude       — Anthropic (claude-opus-4-5, claude-sonnet-4-5, ...)
  openai       — OpenAI (gpt-4o, gpt-4o-mini, ...)
  deepseek     — DeepSeek (deepseek-chat, deepseek-reasoner)
  gemini       — Google Gemini (gemini-1.5-pro, gemini-flash, ...)
  perplexity   — Perplexity (sonar, sonar-pro, ...)
  ollama       — Local Ollama (any model you have pulled)

Config lives in configs/llm_profiles.yaml.  Example:

  profiles:
    - name: claude_primary
      provider: claude
      api_key: sk-ant-...
      model: claude-opus-4-5
      role: heavy          # heavy | daily | embed

    - name: claude_research
      provider: claude
      api_key: sk-ant-...  # second account
      model: claude-sonnet-4-5
      role: daily

    - name: deepseek_primary
      provider: deepseek
      api_key: sk-...
      model: deepseek-chat
      role: daily

    - name: local_ollama
      provider: ollama
      base_url: http://localhost:11434
      model: mistral
      role: daily

  defaults:
    heavy: claude_primary
    daily: deepseek_primary
    embed: local_ollama
    gap_analysis: claude_primary
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import yaml

log = logging.getLogger("brain.llm")

# ─── Profile dataclass ────────────────────────────────────────────────────────

@dataclass
class LLMProfile:
    name: str
    provider: str                   # claude | openai | deepseek | gemini | perplexity | ollama
    model: str
    api_key: str = ""
    base_url: str = ""              # for ollama or custom endpoints
    role: str = "daily"             # heavy | daily | embed
    extra: dict = field(default_factory=dict)

    def client(self):
        """Return a thin client wrapper for this profile."""
        p = self.provider.lower()
        if p == "claude":
            return _ClaudeClient(self)
        elif p == "openai":
            return _OpenAIClient(self)
        elif p == "deepseek":
            return _DeepSeekClient(self)
        elif p == "gemini":
            return _GeminiClient(self)
        elif p == "perplexity":
            return _PerplexityClient(self)
        elif p == "ollama":
            return _OllamaClient(self)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


# ─── Registry ─────────────────────────────────────────────────────────────────

class LLMRegistry:
    """
    Loads all profiles from configs/llm_profiles.yaml.
    Falls back to environment variables if yaml is absent.
    """

    def __init__(self, config_path: str | Path = "configs/llm_profiles.yaml"):
        self.profiles: dict[str, LLMProfile] = {}
        self.defaults: dict[str, str] = {}
        self._load(Path(config_path))

    def _load(self, path: Path):
        if path.exists():
            data = yaml.safe_load(path.read_text())
            for p in data.get("profiles", []):
                profile = LLMProfile(
                    name=p["name"],
                    provider=p["provider"],
                    model=p.get("model", ""),
                    api_key=p.get("api_key") or os.environ.get(
                        f"{p['provider'].upper()}_API_KEY", ""
                    ),
                    base_url=p.get("base_url", ""),
                    role=p.get("role", "daily"),
                    extra=p.get("extra", {}),
                )
                self.profiles[profile.name] = profile
            self.defaults = data.get("defaults", {})
            log.info(f"Loaded {len(self.profiles)} LLM profiles from {path}")
        else:
            log.warning(f"No llm_profiles.yaml found at {path}. Using env vars.")
            self._load_from_env()

    def _load_from_env(self):
        """Minimal env-var fallback so the system works with zero config."""
        env_map = [
            ("claude_default",     "claude",     "claude-sonnet-4-5",   "ANTHROPIC_API_KEY"),
            ("openai_default",     "openai",     "gpt-4o",              "OPENAI_API_KEY"),
            ("deepseek_default",   "deepseek",   "deepseek-chat",       "DEEPSEEK_API_KEY"),
            ("gemini_default",     "gemini",     "gemini-1.5-pro",      "GEMINI_API_KEY"),
            ("perplexity_default", "perplexity", "sonar-pro",           "PERPLEXITY_API_KEY"),
        ]
        for name, provider, model, env_key in env_map:
            key = os.environ.get(env_key, "")
            if key:
                self.profiles[name] = LLMProfile(
                    name=name, provider=provider, model=model,
                    api_key=key, role="daily"
                )

        # Ollama doesn't need an API key
        self.profiles["ollama_local"] = LLMProfile(
            name="ollama_local", provider="ollama",
            model="mistral", base_url="http://localhost:11434", role="daily"
        )

        # Set sensible defaults
        available = list(self.profiles.keys())
        if available:
            self.defaults = {
                "heavy": available[0],
                "daily": available[0],
                "embed": "ollama_local",
                "gap_analysis": available[0],
            }

    def get(self, name: str) -> LLMProfile:
        if name not in self.profiles:
            raise KeyError(f"LLM profile '{name}' not found. Available: {list(self.profiles)}")
        return self.profiles[name]

    def get_for_role(self, role: str) -> LLMProfile:
        """Get the default profile for a role (heavy | daily | embed | gap_analysis)."""
        name = self.defaults.get(role)
        if not name:
            # fallback: first available
            name = next(iter(self.profiles))
        return self.get(name)

    def list_profiles(self) -> list[dict]:
        return [
            {
                "name": p.name,
                "provider": p.provider,
                "model": p.model,
                "role": p.role,
                "has_key": bool(p.api_key),
            }
            for p in self.profiles.values()
        ]


# ─── Thin client wrappers ─────────────────────────────────────────────────────
# Each returns {"text": str, "model": str, "tokens_used": int}

class _ClaudeClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import anthropic
        client = anthropic.Anthropic(api_key=self.profile.api_key)
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"model": self.profile.model, "max_tokens": max_tokens, "messages": messages}
        if system:
            kwargs["system"] = system
        resp = client.messages.create(**kwargs)
        return {
            "text": resp.content[0].text,
            "model": resp.model,
            "tokens_used": resp.usage.input_tokens + resp.usage.output_tokens,
        }


class _OpenAIClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import openai
        client = openai.OpenAI(api_key=self.profile.api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.profile.model, messages=messages, max_tokens=max_tokens
        )
        return {
            "text": resp.choices[0].message.content,
            "model": resp.model,
            "tokens_used": resp.usage.total_tokens,
        }


class _DeepSeekClient:
    """DeepSeek uses an OpenAI-compatible API."""
    BASE_URL = "https://api.deepseek.com/v1"

    def __init__(self, profile: LLMProfile):
        self.profile = profile

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import openai
        client = openai.OpenAI(
            api_key=self.profile.api_key,
            base_url=self.profile.base_url or self.BASE_URL,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.profile.model, messages=messages, max_tokens=max_tokens
        )
        return {
            "text": resp.choices[0].message.content,
            "model": resp.model,
            "tokens_used": resp.usage.total_tokens,
        }


class _GeminiClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import google.generativeai as genai
        genai.configure(api_key=self.profile.api_key)
        model = genai.GenerativeModel(
            model_name=self.profile.model,
            system_instruction=system if system else None,
        )
        resp = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens},
        )
        return {
            "text": resp.text,
            "model": self.profile.model,
            "tokens_used": resp.usage_metadata.total_token_count if hasattr(resp, "usage_metadata") else 0,
        }


class _PerplexityClient:
    """Perplexity uses an OpenAI-compatible API with online/sonar models."""
    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, profile: LLMProfile):
        self.profile = profile

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import openai
        client = openai.OpenAI(
            api_key=self.profile.api_key,
            base_url=self.profile.base_url or self.BASE_URL,
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = client.chat.completions.create(
            model=self.profile.model, messages=messages, max_tokens=max_tokens
        )
        return {
            "text": resp.choices[0].message.content,
            "model": resp.model,
            "tokens_used": resp.usage.total_tokens,
        }


class _OllamaClient:
    def __init__(self, profile: LLMProfile):
        self.profile = profile
        self.base_url = profile.base_url or "http://localhost:11434"

    def complete(self, prompt: str, system: str = "", max_tokens: int = 2048) -> dict:
        import requests
        payload = {
            "model": self.profile.model,
            "prompt": f"{system}\n\n{prompt}" if system else prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("response", ""),
            "model": self.profile.model,
            "tokens_used": data.get("eval_count", 0),
        }
