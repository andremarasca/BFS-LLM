"""Local LLM client using llama-cpp-python.

Implements BaseLLM interface for GGUF models with minimal overhead.
"""

from pathlib import Path
from typing import Any, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .llm_base import BaseLLM, LLMConfig


class ModelValidator:
    """Validates model availability and compatibility."""

    @staticmethod
    def check_library_available() -> None:
        """Ensure llama_cpp is installed."""
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python not found. "
                "Install with: pip install llama-cpp-python"
            )

    @staticmethod
    def check_model_exists(model_path: str) -> None:
        """Ensure model file exists at given path."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path is not a file: {model_path}"
            )


class LocalQwenLLM(BaseLLM):
    """Local LLM client for GGUF models via llama-cpp-python.

    Provides efficient inference on local hardware with minimal configuration.
    """

    DEFAULT_CONTEXT_SIZE = 4096
    STOP_TOKENS = ["</s>", "<|endoftext|>"]

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.6,
        context_size: int = DEFAULT_CONTEXT_SIZE
    ) -> None:
        """Initialize local model client.

        Args:
            model_path: Path to GGUF model file
            temperature: Sampling temperature
            context_size: Model context window size

        Raises:
            RuntimeError: If llama-cpp-python not installed
            FileNotFoundError: If model file not found
        """
        ModelValidator.check_library_available()
        ModelValidator.check_model_exists(model_path)

        config = LLMConfig(
            temperature=temperature,
            timeout_s=60
        )

        super().__init__(config=config)

        self._model_path = model_path
        self._context_size = context_size
        self._model = self._load_model()

    def _load_model(self) -> Llama:
        """Load GGUF model into memory."""
        return Llama(
            model_path=self._model_path,
            n_ctx=self._context_size,
            temperature=self.config.temperature
        )

    def _raw_request(self, *, content: str, model: Optional[str] = None) -> str:
        """Generate completion using local model.

        Args:
            content: Input prompt
            model: Ignored (single model per instance)

        Returns:
            Generated text completion
        """
        result = self._generate(content)
        return self._extract_text(result)

    def _generate(self, prompt: str) -> Any:
        """Execute model inference."""
        return self._model(
            prompt,
            temperature=self.config.temperature,
            stop=self.STOP_TOKENS
        )

    @staticmethod
    def _extract_text(result: Any) -> str:
        """Extract text from llama-cpp response format."""
        if not isinstance(result, dict):
            return str(result)

        choices = result.get("choices", [])
        if not choices:
            return ""

        return choices[0].get("text", "")
