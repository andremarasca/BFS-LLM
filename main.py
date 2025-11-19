#!/usr/bin/env python3
"""BFS-LLM: Concept tree expansion using LLM with BFS traversal.

Entry point for the BFS tree expansion system.
Loads configuration, initializes LLM client, and runs expansion motor.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from core import LocalQwenLLM, OpenAILLM
from core.llm_bfs_motor import BFSMotor


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment variable.

    Returns:
        API key or None if not found
    """
    return os.getenv('OPENAI_API_KEY')


def create_llm_client(config: dict):
    """Create appropriate LLM client based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized LLM client (OpenAILLM or LocalQwenLLM)

    Raises:
        RuntimeError: If OpenAI API key not found when required
    """
    llm_modos = config.get('llm_modos', {})
    usar_local = llm_modos.get('usar_local', False)

    if usar_local:
        logging.info("Using local Qwen model")
        model_path = llm_modos.get('caminho_modelo_local')

        if not model_path:
            raise ValueError("caminho_modelo_local not specified in config")

        return LocalQwenLLM(
            model_path=model_path,
            temperature=config['llm_cliente'].get('temperatura', 0.6),
            max_tokens=config['llm_cliente'].get('max_tokens', 1024)
        )

    logging.info("Using OpenAI API")
    llm_config = config.get('llm_cliente', {})

    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )

    return OpenAILLM(
        api_key=api_key,
        model=llm_config.get('modelo', 'gpt-4o-mini'),
        temperature=llm_config.get('temperatura', 0.1),
        max_tokens=llm_config.get('max_tokens', 16384),
        timeout_s=llm_config.get('timeout_s', 60)
    )


def check_dry_run(config: dict) -> bool:
    """Check if dry-run mode is enabled.

    Args:
        config: Configuration dictionary

    Returns:
        True if dry-run mode enabled
    """
    return config.get('execucao', {}).get('dry_run', False)


def main() -> int:
    """Main entry point for BFS tree expansion.

    Returns:
        Exit code (0 = success, 1 = error)
    """
    try:
        logging.info("=" * 60)
        logging.info("BFS-LLM: Concept Tree Expansion System")
        logging.info("=" * 60)

        config = load_config()

        if check_dry_run(config):
            logging.warning("DRY RUN MODE: No LLM calls will be made")
            logging.info("Set execucao.dry_run: false in config.yaml to run normally")
            return 0

        llm_client = create_llm_client(config)

        motor = BFSMotor(config_path="config.yaml")
        motor.set_llm_client(llm_client)

        logging.info("Starting BFS expansion...")
        final_tree = motor.run()

        logging.info("=" * 60)
        logging.info("Expansion completed successfully!")
        logging.info(f"Final tree saved to: {config['arquivos']['arvore_saida']}")
        logging.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logging.warning("\nProcess interrupted by user")
        return 1

    except Exception as exc:
        logging.error(f"Fatal error: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
