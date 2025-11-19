"""Base abstractions for LLM clients.

Provides abstract interface and shared logging for all LLM backends.
Concrete implementations (OpenAI, local models) inherit from BaseLLM.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class TrafficLogger:
    """Handles logging of LLM request/response pairs to disk."""

    def __init__(self, root_dir: str, run_id: Optional[str] = None) -> None:
        """Initialize logger with directory structure.

        Args:
            root_dir: Base directory for all traffic logs
            run_id: Unique identifier for this session (auto-generated if None)
        """
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

        self._run_id = run_id or self._generate_run_id()
        self._run_dir = self._root_dir / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._counter = 0

    @staticmethod
    def _generate_run_id() -> str:
        """Generate timestamped run identifier."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _next_index(self) -> str:
        """Return zero-padded sequential index."""
        self._counter += 1
        return f"{self._counter:04d}"

    def log(self, kind: str, payload: Any) -> None:
        """Persist payload to JSON file.

        Args:
            kind: File suffix describing payload type (e.g., 'Entrada', 'Saida')
            payload: Data to log (dict or string)

        Note:
            Logging failures are silently ignored to never interrupt main flow.
        """
        try:
            index = self._next_index()
            filename = f"{index}_{kind}.json"
            filepath = self._run_dir / filename

            self._write_payload(filepath, payload)
        except Exception:
            pass

    @staticmethod
    def _write_payload(filepath: Path, payload: Any) -> None:
        """Write payload to file based on type."""
        if isinstance(payload, str):
            filepath.write_text(payload, encoding="utf-8")
            return

        filepath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


class LLMConfig:
    """Encapsulates LLM configuration parameters."""

    def __init__(
        self,
        temperature: float = 0.6,
        timeout_s: int = 60
    ) -> None:
        """Initialize configuration with validated parameters.

        Args:
            temperature: Sampling temperature (0.0-2.0)
            timeout_s: Request timeout in seconds
        """
        self._validate_temperature(temperature)
        self._validate_timeout(timeout_s)

        self.temperature = temperature
        self.timeout_s = timeout_s

    @staticmethod
    def _validate_temperature(value: float) -> None:
        """Ensure temperature is within valid range."""
        if not 0.0 <= value <= 2.0:
            raise ValueError(f"Temperature must be in [0.0, 2.0], got {value}")

    @staticmethod
    def _validate_timeout(value: int) -> None:
        """Ensure timeout is positive."""
        if value <= 0:
            raise ValueError(f"timeout_s must be positive, got {value}")


class BaseLLM(ABC):
    """Abstract base for LLM clients with automatic traffic logging.

    Subclasses must implement _raw_request to integrate specific backends.
    All logging is handled transparently by the base class.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        traffic_root_dir: str = "llm_traffic",
        run_id: Optional[str] = None
    ) -> None:
        """Initialize LLM client with configuration and logging.

        Args:
            config: LLM parameters (uses defaults if None)
            traffic_root_dir: Base directory for traffic logs
            run_id: Session identifier for logs
        """
        self._config = config or LLMConfig()
        self._logger = TrafficLogger(traffic_root_dir, run_id)

    @property
    def config(self) -> LLMConfig:
        """Access current configuration."""
        return self._config

    def send_request(self, content: str, model: Optional[str] = None) -> str:
        """Send request to LLM with automatic logging.

        Args:
            content: Input prompt or message
            model: Model identifier (backend-specific)

        Returns:
            Raw text response from LLM
        """
        self._logger.log("Entrada", content)

        response = self._raw_request(content=content, model=model)

        self._logger.log("Saida", response)
        self._logger.log("Conjunto", {
            "Entrada": content,
            "Saida": response
        })

        return response

    @abstractmethod
    def _raw_request(self, *, content: str, model: Optional[str] = None) -> str:
        """Execute backend-specific request.

        Args:
            content: Input prompt or message
            model: Model identifier (backend-specific)

        Returns:
            Raw text response from LLM

        Note:
            Subclasses must implement this. Do NOT add logging here.
        """
        raise NotImplementedError
