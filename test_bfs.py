#!/usr/bin/env python3
"""Test script for BFS motor functionality.

Tests core components without requiring actual LLM calls.
"""

import json
import logging
from pathlib import Path

from core import BFSTreeExpander, JsonValidator, PartialPersistence, RetryPolicy

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_json_validator():
    """Test JSON Schema validation."""
    print("\n=== Testing JsonValidator ===")

    validator = JsonValidator("json_schema.json")

    valid_data = [
        {"name": "Test Concept", "definition": "A test definition"}
    ]

    try:
        validator.validate(valid_data)
        print("[OK] Valid data passed validation")
    except Exception as e:
        print(f"[FAIL] Validation failed: {e}")
        return False

    invalid_data = [
        {"name": "", "definition": "Missing name"}
    ]

    try:
        validator.validate(invalid_data)
        print("[FAIL] Invalid data should have failed validation")
        return False
    except Exception:
        print("[OK] Invalid data correctly rejected")

    return True


def test_partial_persistence():
    """Test snapshot saving functionality."""
    print("\n=== Testing PartialPersistence ===")

    test_dir = Path("test_snapshots")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    persistence = PartialPersistence(snapshot_dir="test_snapshots")

    test_tree = {
        "name": "Root",
        "definition": "Test root",
        "sub_concepts": []
    }

    try:
        persistence.save_snapshot(test_tree, 0)
        snapshot_file = test_dir / "snapshot_level_000.json"

        if snapshot_file.exists():
            print("[OK] Snapshot file created successfully")
        else:
            print("[FAIL] Snapshot file not created")
            return False

        with snapshot_file.open('r') as f:
            loaded = json.load(f)
            if loaded == test_tree:
                print("[OK] Snapshot content matches original tree")
            else:
                print("[FAIL] Snapshot content mismatch")
                return False

    except Exception as e:
        print(f"[FAIL] Persistence test failed: {e}")
        return False
    finally:
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

    return True


def test_retry_policy():
    """Test retry logic."""
    print("\n=== Testing RetryPolicy ===")

    retry_policy = RetryPolicy(max_attempts=3, timeout_between=0, continue_on_failure=True)

    attempts = []

    def failing_operation():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("Simulated failure")
        return "Success"

    result = retry_policy.execute_with_retry(failing_operation, "test_node")

    if result == "Success" and len(attempts) == 2:
        print(f"[OK] Retry succeeded after {len(attempts)} attempts")
        return True
    else:
        print(f"[FAIL] Retry test failed (attempts: {len(attempts)}, result: {result})")
        return False


def test_bfs_tree_expander():
    """Test BFS tree expansion logic."""
    print("\n=== Testing BFSTreeExpander ===")

    with open("base_prompt.json", 'r', encoding='utf-8') as f:
        base_prompt = json.load(f)

    validator = JsonValidator("json_schema.json")
    expander = BFSTreeExpander(
        base_prompt=base_prompt,
        validator=validator,
        max_depth=5
    )

    eligible_node = {
        "name": "Test Node",
        "definition": "A test node",
        "sub_concepts": []
    }

    if expander.is_eligible_node(eligible_node):
        print("[OK] Eligible node correctly identified")
    else:
        print("[FAIL] Eligible node not recognized")
        return False

    leaf_node = {
        "name": "Leaf Node",
        "definition": "A leaf node",
        "sub_concepts": [],
        "is_leaf_node": True
    }

    if not expander.is_eligible_node(leaf_node):
        print("[OK] Leaf node correctly excluded")
    else:
        print("[FAIL] Leaf node should be excluded")
        return False

    test_tree = {
        "name": "Root",
        "definition": "Root concept",
        "sub_concepts": [
            {
                "name": "Child 1",
                "definition": "First child",
                "sub_concepts": []
            },
            {
                "name": "Child 2",
                "definition": "Second child",
                "sub_concepts": []
            }
        ]
    }

    level_0 = expander.collect_nodes_at_level(test_tree, 0)
    level_1 = expander.collect_nodes_at_level(test_tree, 1)

    if len(level_0) == 1 and len(level_1) == 2:
        print(f"[OK] Level collection correct (L0: {len(level_0)}, L1: {len(level_1)})")
    else:
        print(f"[FAIL] Level collection incorrect (L0: {len(level_0)}, L1: {len(level_1)})")
        return False

    max_depth = expander.get_max_depth(test_tree)
    if max_depth == 1:
        print(f"[OK] Max depth calculation correct: {max_depth}")
    else:
        print(f"[FAIL] Max depth incorrect: {max_depth} (expected 1)")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("BFS-LLM Component Tests")
    print("=" * 60)

    tests = [
        ("JSON Validator", test_json_validator),
        ("Partial Persistence", test_partial_persistence),
        ("Retry Policy", test_retry_policy),
        ("BFS Tree Expander", test_bfs_tree_expander),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] {name} raised exception: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
