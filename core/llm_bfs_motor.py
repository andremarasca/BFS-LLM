"""BFS-based concept tree expansion engine.

Implements breadth-first traversal with LLM-powered node expansion,
validation, retry policies, and partial persistence.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import yaml

from .llm_base import BaseLLM
from .llm_openai import JsonExtractor


class JsonValidator:
    """Validates JSON data against JSON Schema specification."""

    def __init__(self, schema_path: str) -> None:
        """Initialize validator with schema file.

        Args:
            schema_path: Path to JSON Schema file
        """
        self._schema = self._load_schema(schema_path)

    @staticmethod
    def _load_schema(path: str) -> Dict[str, Any]:
        """Load JSON Schema from file."""
        schema_file = Path(path)
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")

        with schema_file.open('r', encoding='utf-8') as f:
            return json.load(f)

    def validate(self, data: Any) -> None:
        """Validate data against loaded schema.

        Args:
            data: Data to validate

        Raises:
            jsonschema.ValidationError: If validation fails
        """
        jsonschema.validate(instance=data, schema=self._schema)


class PartialPersistence:
    """Handles incremental tree persistence for fault tolerance."""

    def __init__(self, snapshot_dir: str = "snapshots") -> None:
        """Initialize persistence handler.

        Args:
            snapshot_dir: Directory for snapshot files
        """
        self._snapshot_dir = Path(snapshot_dir)
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, tree: Dict[str, Any], level: int) -> None:
        """Save tree snapshot for current BFS level.

        Args:
            tree: Complete concept tree
            level: Current BFS depth level
        """
        filename = f"snapshot_level_{level:03d}.json"
        filepath = self._snapshot_dir / filename

        with filepath.open('w', encoding='utf-8') as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)

        logging.info(f"Snapshot saved: {filepath}")

    def save_final(self, tree: Dict[str, Any], output_path: str) -> None:
        """Save final expanded tree.

        Args:
            tree: Complete concept tree
            output_path: Destination file path
        """
        output_file = Path(output_path)

        with output_file.open('w', encoding='utf-8') as f:
            json.dump(tree, f, ensure_ascii=False, indent=2)

        logging.info(f"Final tree saved: {output_path}")


class RetryPolicy:
    """Implements configurable retry logic for LLM requests."""

    def __init__(
        self,
        max_attempts: int = 3,
        timeout_between: int = 2,
        continue_on_failure: bool = True
    ) -> None:
        """Initialize retry policy.

        Args:
            max_attempts: Maximum retry attempts per node
            timeout_between: Seconds between attempts
            continue_on_failure: Continue BFS even if node expansion fails
        """
        self._max_attempts = max_attempts
        self._timeout_between = timeout_between
        self._continue_on_failure = continue_on_failure

    @property
    def max_attempts(self) -> int:
        """Maximum retry attempts."""
        return self._max_attempts

    @property
    def timeout_between(self) -> int:
        """Timeout between attempts in seconds."""
        return self._timeout_between

    @property
    def continue_on_failure(self) -> bool:
        """Whether to continue BFS on node expansion failure."""
        return self._continue_on_failure

    def execute_with_retry(
        self,
        operation: callable,
        node_name: str
    ) -> Optional[Any]:
        """Execute operation with retry logic.

        Args:
            operation: Callable to execute
            node_name: Node identifier for logging

        Returns:
            Operation result or None if all attempts fail
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                logging.warning(
                    f"Attempt {attempt}/{self._max_attempts} failed for '{node_name}': {exc}"
                )

                if attempt < self._max_attempts:
                    time.sleep(self._timeout_between)
                else:
                    logging.error(f"All retry attempts exhausted for '{node_name}'")
                    if not self._continue_on_failure:
                        raise
                    return None

        return None


class BFSTreeExpander:
    """Manages BFS traversal and node expansion logic."""

    def __init__(
        self,
        base_prompt: Dict[str, Any],
        validator: Optional[JsonValidator] = None,
        max_depth: int = -1,
        max_nodes_per_level: int = -1
    ) -> None:
        """Initialize BFS expander.

        Args:
            base_prompt: Template with prompt and concept tree structure
            validator: JSON Schema validator (optional)
            max_depth: Maximum tree depth (-1 = unlimited)
            max_nodes_per_level: Maximum nodes to expand per level (-1 = unlimited)
        """
        self._base_prompt = base_prompt
        self._validator = validator
        self._max_depth = max_depth
        self._max_nodes_per_level = max_nodes_per_level

    def is_eligible_node(self, node: Dict[str, Any]) -> bool:
        """Check if node is eligible for expansion.

        Args:
            node: Node to check

        Returns:
            True if node should be expanded
        """
        if node.get('is_leaf_node', False):
            return False

        sub_concepts = node.get('sub_concepts')
        if sub_concepts is None:
            return False

        return isinstance(sub_concepts, list) and len(sub_concepts) == 0

    def build_prompt(self, node: Dict[str, Any], current_tree: Dict[str, Any]) -> str:
        """Build expansion prompt for given node.

        Args:
            node: Node to expand
            current_tree: Current state of concept tree

        Returns:
            Formatted prompt string
        """
        prompt_data = deepcopy(self._base_prompt)
        prompt_data['concept_tree'] = current_tree
        prompt_data['node_to_expand'] = node

        return json.dumps(prompt_data, ensure_ascii=False, indent=2)

    def process_llm_response(
        self,
        raw_response: str,
        target_node: Dict[str, Any]
    ) -> bool:
        """Process LLM response and update node.

        Args:
            raw_response: Raw LLM text response
            target_node: Node being expanded (modified in place)

        Returns:
            True if expansion succeeded, False otherwise
        """
        try:
            extracted = JsonExtractor.extract(raw_response)
        except Exception as exc:
            logging.error(f"JSON extraction failed: {exc}")
            return False

        if not isinstance(extracted, list):
            logging.error(f"Expected list, got {type(extracted).__name__}")
            return False

        if self._validator:
            try:
                self._validator.validate(extracted)
            except jsonschema.ValidationError as exc:
                logging.error(f"Schema validation failed: {exc}")
                return False

        if len(extracted) == 0:
            target_node['sub_concepts'] = []
            target_node['is_leaf_node'] = True
            logging.info(f"Node '{target_node.get('name')}' marked as leaf (empty expansion)")
            return True

        for child in extracted:
            child['sub_concepts'] = []
            child.pop('is_leaf_node', None)

        target_node['sub_concepts'] = extracted
        target_node.pop('is_leaf_node', None)

        logging.info(f"Node '{target_node.get('name')}' expanded with {len(extracted)} children")
        return True

    def collect_nodes_at_level(
        self,
        tree: Dict[str, Any],
        target_depth: int
    ) -> List[Dict[str, Any]]:
        """Collect all nodes at specific depth level.

        Args:
            tree: Concept tree root
            target_depth: Depth level to collect

        Returns:
            List of nodes at target depth
        """
        if target_depth == 0:
            return [tree]

        nodes_at_level = []
        queue = deque([(tree, 0)])

        while queue:
            node, depth = queue.popleft()

            if depth == target_depth:
                nodes_at_level.append(node)
                continue

            if depth < target_depth:
                sub_concepts = node.get('sub_concepts', [])
                for child in sub_concepts:
                    queue.append((child, depth + 1))

        return nodes_at_level

    def get_eligible_nodes_at_level(
        self,
        tree: Dict[str, Any],
        level: int
    ) -> List[Dict[str, Any]]:
        """Get eligible nodes at specific BFS level.

        Args:
            tree: Concept tree root
            level: BFS depth level

        Returns:
            List of eligible nodes (limited by max_nodes_per_level if set)
        """
        all_nodes = self.collect_nodes_at_level(tree, level)
        eligible = [n for n in all_nodes if self.is_eligible_node(n)]

        if self._max_nodes_per_level > 0:
            eligible = eligible[:self._max_nodes_per_level]

        return eligible

    def get_max_depth(self, tree: Dict[str, Any]) -> int:
        """Calculate maximum depth of tree.

        Args:
            tree: Concept tree root

        Returns:
            Maximum depth (root = 0)
        """
        if not tree.get('sub_concepts'):
            return 0

        max_child_depth = 0
        for child in tree['sub_concepts']:
            child_depth = self.get_max_depth(child)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth + 1


class BFSMotor:
    """Main orchestrator for BFS-based tree expansion."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        run_id: Optional[str] = None
    ) -> None:
        """Initialize BFS motor with configuration.

        Args:
            config_path: Path to YAML configuration file
            run_id: Unique run identifier for logging
        """
        self._config = self._load_config(config_path)
        self._run_id = run_id
        self._setup_logging()

        self._validator = self._create_validator()
        self._persistence = PartialPersistence()
        self._retry_policy = self._create_retry_policy()
        self._expander = self._create_expander()

        self._tree = self._load_tree()
        self._llm_client = None

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with config_file.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> None:
        """Configure logging based on config settings."""
        log_config = self._config.get('logging', {})
        level = getattr(logging, log_config.get('nivel', 'INFO'))

        # Clear existing handlers to ensure clean configuration
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        handlers = []

        if log_config.get('salvar_em_arquivo', True):
            arquivo = log_config.get('arquivo_log', 'bfs_motor.log')
            handlers.append(logging.FileHandler(arquivo, encoding='utf-8'))

        handlers.append(logging.StreamHandler())

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
            force=True
        )

    def _create_validator(self) -> Optional[JsonValidator]:
        """Create JSON validator if validation enabled."""
        validacao = self._config.get('validacao', {})

        if not validacao.get('validar_schema', True):
            return None

        schema_path = self._config['arquivos']['json_schema']
        return JsonValidator(schema_path)

    def _create_retry_policy(self) -> RetryPolicy:
        """Create retry policy from config."""
        retry_config = self._config.get('retry', {})

        return RetryPolicy(
            max_attempts=retry_config.get('max_tentativas', 3),
            timeout_between=retry_config.get('timeout_entre_tentativas', 2),
            continue_on_failure=retry_config.get('continuar_em_falha', True)
        )

    def _create_expander(self) -> BFSTreeExpander:
        """Create BFS tree expander."""
        base_prompt_path = self._config['arquivos']['base_prompt']

        with open(base_prompt_path, 'r', encoding='utf-8') as f:
            base_prompt = json.load(f)

        bfs_config = self._config.get('bfs', {})

        return BFSTreeExpander(
            base_prompt=base_prompt,
            validator=self._validator,
            max_depth=bfs_config.get('max_profundidade', -1),
            max_nodes_per_level=bfs_config.get('max_nos_por_nivel', -1)
        )

    def _load_tree(self) -> Dict[str, Any]:
        """Load initial concept tree from base prompt."""
        base_prompt_path = self._config['arquivos']['base_prompt']

        with open(base_prompt_path, 'r', encoding='utf-8') as f:
            base_prompt = json.load(f)

        return deepcopy(base_prompt['concept_tree'])

    def set_llm_client(self, client: BaseLLM) -> None:
        """Set LLM client for expansion.

        Args:
            client: Initialized LLM client
        """
        self._llm_client = client

    def _ensure_sub_concepts_exist(self, node: Dict[str, Any]) -> None:
        """Recursively ensure all nodes have sub_concepts field.

        Args:
            node: Node to process (modified in place)
        """
        if 'sub_concepts' not in node:
            node['sub_concepts'] = []

        for child in node.get('sub_concepts', []):
            self._ensure_sub_concepts_exist(child)

    def expand_node(self, node: Dict[str, Any]) -> bool:
        """Expand single node using LLM.

        Args:
            node: Node to expand (modified in place)

        Returns:
            True if expansion succeeded
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm_client() first.")

        prompt = self._expander.build_prompt(node, self._tree)

        def _request_operation():
            raw_response = self._llm_client.send_request(prompt)
            success = self._expander.process_llm_response(raw_response, node)
            if not success:
                raise RuntimeError("LLM response processing failed")
            return success

        node_name = node.get('name', 'unknown')
        result = self._retry_policy.execute_with_retry(_request_operation, node_name)

        return result is not None

    def run(self) -> Dict[str, Any]:
        """Execute BFS expansion loop.

        Returns:
            Final expanded tree
        """
        if self._llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm_client() first.")

        logging.info("Starting BFS tree expansion")

        self._ensure_sub_concepts_exist(self._tree)

        max_depth = self._expander._max_depth
        current_level = 0

        performance_config = self._config.get('performance', {})
        delay = performance_config.get('delay_entre_requests', 0.5)

        while True:
            all_nodes_at_level = self._expander.collect_nodes_at_level(
                self._tree,
                current_level
            )

            if not all_nodes_at_level:
                logging.info(f"No nodes at level {current_level}, BFS completed")
                break

            if max_depth >= 0 and current_level >= max_depth:
                logging.info(
                    f"Level {current_level}: at max_depth limit "
                    f"({len(all_nodes_at_level)} nodes not expanded)"
                )
                current_level += 1
                continue

            eligible_nodes = self._expander.get_eligible_nodes_at_level(
                self._tree,
                current_level
            )

            if not eligible_nodes:
                logging.info(
                    f"Level {current_level}: no eligible nodes "
                    f"({len(all_nodes_at_level)} nodes already expanded), skipping"
                )
                current_level += 1
                continue

            logging.info(
                f"Level {current_level}: expanding {len(eligible_nodes)} nodes"
            )

            for i, node in enumerate(eligible_nodes, 1):
                node_name = node.get('name', 'unknown')
                logging.info(f"Expanding node {i}/{len(eligible_nodes)}: '{node_name}'")

                self.expand_node(node)

                if delay > 0 and i < len(eligible_nodes):
                    time.sleep(delay)

                self._persistence.save_snapshot(self._tree, current_level)

            current_level += 1

        self._ensure_sub_concepts_exist(self._tree)

        output_path = self._config['arquivos']['arvore_saida']
        self._persistence.save_final(self._tree, output_path)

        logging.info("BFS expansion completed successfully")

        return self._tree
