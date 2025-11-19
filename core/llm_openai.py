"""OpenAI LLM client with robust JSON extraction.

Implements BaseLLM interface for OpenAI chat completions API.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .llm_base import BaseLLM, LLMConfig


class JsonExtractor:
    """Extracts and parses JSON from LLM text responses."""

    MAX_RESPONSE_LENGTH = 40000

    @classmethod
    def extract(cls, raw: str) -> Dict[str, Any]:
        """Extract valid JSON object from potentially messy LLM output.

        Args:
            raw: Raw text response from LLM

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found or response too long
        """
        cls._validate_length(raw)
        cleaned = cls._remove_code_blocks(raw)
        candidates = cls._find_json_candidates(cleaned)

        if candidates:
            return cls._select_best_candidate(candidates)

        return cls._fallback_extraction(cleaned)

    @classmethod
    def _validate_length(cls, text: str) -> None:
        """Reject excessively long responses."""
        if len(text) > cls.MAX_RESPONSE_LENGTH:
            raise ValueError(
                f"Response too long ({len(text)} chars, "
                f"max {cls.MAX_RESPONSE_LENGTH}). "
                "Model likely generated explanatory text instead of JSON."
            )

    @staticmethod
    def _remove_code_blocks(text: str) -> str:
        """Strip markdown code block markers."""
        text = re.sub(r'```[\w]*\s*\n', '', text)
        text = re.sub(r'\n```\s*', '', text)
        return text.strip()

    @classmethod
    def _find_json_candidates(cls, text: str) -> List[Tuple[int, int, Dict]]:
        """Find all valid JSON objects in text."""
        candidates = []
        start_positions = [i for i, c in enumerate(text) if c == '{']

        for start in start_positions:
            end = cls._find_matching_brace(text, start)
            if end == -1:
                continue

            json_str = text[start:end + 1]
            parsed = cls._try_parse(json_str)

            if parsed is not None:
                candidates.append((start, end, parsed))

        return candidates

    @staticmethod
    def _find_matching_brace(text: str, start: int) -> int:
        """Find matching closing brace using stack-based tracking."""
        stack = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                stack += 1
            elif text[i] == '}':
                stack -= 1
                if stack == 0:
                    return i
        return -1

    @staticmethod
    def _try_parse(json_str: str) -> Optional[Dict]:
        """Attempt to parse JSON with trailing comma correction."""
        fixed = re.sub(r',\s*(?=[}\]])', '', json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _select_best_candidate(
        candidates: List[Tuple[int, int, Dict]]
    ) -> Dict:
        """Select the last (most complete) valid JSON object."""
        candidates.sort(key=lambda x: x[1])
        return candidates[-1][2]

    @classmethod
    def _fallback_extraction(cls, text: str) -> Dict[str, Any]:
        """Last resort: extract first valid JSON or raise."""
        start = text.find("{")
        if start == -1:
            raise ValueError(
                f"No JSON object found. Response: {text[:200]}..."
            )

        end = cls._find_matching_brace(text, start)
        if end == -1:
            raise ValueError(
                f"Unbalanced braces. Response: {text[:200]}..."
            )

        json_str = text[start:end + 1]
        parsed = cls._try_parse(json_str)

        if parsed is None:
            raise cls._build_parse_error(json_str)

        return parsed

    @staticmethod
    def _build_parse_error(json_str: str) -> ValueError:
        """Build detailed error message for failed parsing."""
        try:
            json.loads(json_str)
        except json.JSONDecodeError as e:
            lines = json_str.split('\n')
            error_line = min(e.lineno - 1, len(lines) - 1)
            context_start = max(0, error_line - 2)
            context_end = min(len(lines), error_line + 3)

            context = '\n'.join(
                f"{i+1}: {lines[i]}"
                for i in range(context_start, context_end)
            )

            return ValueError(
                f"JSON parsing failed at line {e.lineno}, col {e.colno}: {e.msg}\n"
                f"Context:\n{context}\n\n"
                f"Full response (first 500 chars): {json_str[:500]}"
            )

        return ValueError("Unknown JSON parsing error")


class OpenAIRequestBuilder:
    """Builds OpenAI API request parameters with adaptive retry logic."""

    RETRIABLE_ERROR_PATTERNS = [
        'rate limit',
        'invalid_request_error',
        'maximum context length',
        'timeout'
    ]

    def __init__(self, client: OpenAI, config: LLMConfig) -> None:
        """Initialize request builder.

        Args:
            client: OpenAI client instance
            config: LLM configuration parameters
        """
        self._client = client
        self._config = config

    def execute(
        self,
        model: str,
        content: str
    ) -> str:
        """Execute request with adaptive parameter handling.

        Args:
            model: OpenAI model identifier
            content: User message content

        Returns:
            Model response text

        Raises:
            RuntimeError: On API errors or invalid model
        """
        params = self._build_base_params(model, content)

        try:
            return self._send_request(params)
        except Exception as exc:
            return self._handle_error(exc, params)

    def _build_base_params(
        self,
        model: str,
        content: str
    ) -> Dict[str, Any]:
        """Construct initial request parameters."""
        params = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "timeout": self._config.timeout_s,
            "max_completion_tokens": self._config.max_tokens
        }

        if self._should_include_temperature():
            params["temperature"] = self._config.temperature

        return params

    def _should_include_temperature(self) -> bool:
        """Check if temperature should be included in request."""
        return self._config.temperature not in (0.0, 1.0)

    def _send_request(self, params: Dict[str, Any]) -> str:
        """Send request and extract response text."""
        response = self._client.chat.completions.create(**params)
        return response.choices[0].message.content

    def _handle_error(
        self,
        exc: Exception,
        params: Dict[str, Any]
    ) -> str:
        """Handle API errors with adaptive retry strategies."""
        error_msg = str(exc).lower()

        if "v1/responses" in error_msg:
            raise self._build_model_error(params["model"], exc)

        if "max_completion_tokens" in error_msg:
            return self._retry_with_max_tokens(params, exc)

        if "temperature" in error_msg:
            return self._retry_without_temperature(params, exc)

        if self._is_retriable_error(error_msg):
            raise RuntimeError(f"OpenAI API retriable error: {exc}") from exc

        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    @staticmethod
    def _build_model_error(model: str, exc: Exception) -> RuntimeError:
        """Build detailed error for invalid model name."""
        return RuntimeError(
            f"Model '{model}' is not a valid OpenAI chat model. "
            "Valid models: gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini, "
            "gpt-3.5-turbo, o1-preview, o1-mini. "
            f"Original error: {exc}"
        )

    def _retry_with_max_tokens(
        self,
        params: Dict[str, Any],
        original_exc: Exception
    ) -> str:
        """Retry request using max_tokens instead of max_completion_tokens."""
        params.pop("max_completion_tokens", None)
        params["max_tokens"] = self._config.max_tokens

        try:
            return self._send_request(params)
        except Exception as exc:
            raise RuntimeError(f"Retry failed: {exc}") from original_exc

    def _retry_without_temperature(
        self,
        params: Dict[str, Any],
        original_exc: Exception
    ) -> str:
        """Retry request without temperature parameter."""
        params.pop("temperature", None)

        try:
            return self._send_request(params)
        except Exception as exc:
            raise RuntimeError(f"Retry failed: {exc}") from original_exc

    def _is_retriable_error(self, error_msg: str) -> bool:
        """Check if error indicates a retriable condition."""
        return any(
            pattern in error_msg
            for pattern in self.RETRIABLE_ERROR_PATTERNS
        )


class OpenAILLM(BaseLLM):
    """LLM client for OpenAI chat completions API.

    Provides robust request handling with adaptive parameter negotiation
    and comprehensive error recovery.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_tokens: int = 4096,
        timeout_s: int = 60
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Default model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response length
            timeout_s: Request timeout in seconds
        """
        config = LLMConfig(
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s
        )

        super().__init__(config=config)

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._request_builder = OpenAIRequestBuilder(self._client, config)

    def _raw_request(self, *, content: str, model: Optional[str] = None) -> str:
        """Execute OpenAI API request.

        Args:
            content: User message
            model: Model override (uses instance default if None)

        Returns:
            Model response text
        """
        target_model = model or self._model
        return self._request_builder.execute(target_model, content)

    @staticmethod
    def extract_json_from_text(raw: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text.

        Convenience method delegating to JsonExtractor.

        Args:
            raw: Raw LLM response

        Returns:
            Parsed JSON dictionary
        """
        return JsonExtractor.extract(raw)
