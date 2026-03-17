"""Tests for the shibing624/sharegpt_gpt4 dataset from HuggingFace."""

import pytest
from datasets import load_dataset
import torch


DATASET_NAME = "shibing624/sharegpt_gpt4"


@pytest.fixture(scope="module")
def dataset():
    """Load the full dataset (train split)."""
    ds = load_dataset(DATASET_NAME, split="train")
    return ds


@pytest.fixture(scope="module")
def sample(dataset):
    """A small sample for faster per-entry tests."""
    return dataset.select(range(min(100, len(dataset))))


# ---------------------------------------------------------------------------
# Dataset-level tests
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    def test_loads_successfully(self, dataset):
        assert dataset is not None

    def test_has_rows(self, dataset):
        assert len(dataset) > 0

    def test_expected_row_count(self, dataset):
        # The dataset page reports ~103k rows
        assert len(dataset) > 100_000

    def test_has_conversations_column(self, dataset):
        assert "conversations" in dataset.column_names

    def test_only_expected_columns(self, dataset):
        # The dataset should only have the 'conversations' column
        assert dataset.column_names == ["conversations"]


# ---------------------------------------------------------------------------
# Conversation structure tests
# ---------------------------------------------------------------------------

class TestConversationStructure:
    def test_conversations_is_list(self, sample):
        for row in sample:
            assert isinstance(row["conversations"], list)

    def test_conversations_non_empty(self, sample):
        for row in sample:
            assert len(row["conversations"]) > 0

    def test_each_turn_has_required_keys(self, sample):
        for row in sample:
            for turn in row["conversations"]:
                assert "from" in turn, f"Turn missing 'from' key: {turn}"
                assert "value" in turn, f"Turn missing 'value' key: {turn}"

    def test_from_field_valid_roles(self, sample):
        valid_roles = {"human", "gpt", "system"}
        for row in sample:
            for turn in row["conversations"]:
                assert turn["from"] in valid_roles, (
                    f"Unexpected role: {turn['from']}"
                )

    def test_value_field_is_string(self, sample):
        for row in sample:
            for turn in row["conversations"]:
                assert isinstance(turn["value"], str)

    def test_most_values_non_empty(self, sample):
        total, empty = 0, 0
        for row in sample:
            for turn in row["conversations"]:
                total += 1
                if len(turn["value"].strip()) == 0:
                    empty += 1
        # Allow a small fraction of empty values (dataset has a few)
        assert empty / total < 0.05, (
            f"Too many empty values: {empty}/{total}"
        )

    def test_first_turn_is_human_or_system(self, sample):
        for row in sample:
            first_role = row["conversations"][0]["from"]
            assert first_role in {"human", "system"}, (
                f"First turn should be human or system, got: {first_role}"
            )


# ---------------------------------------------------------------------------
# Conversation quality / statistics tests
# ---------------------------------------------------------------------------

class TestConversationStatistics:
    def test_multi_turn_conversations_exist(self, dataset):
        multi_turn = sum(1 for row in dataset if len(row["conversations"]) > 2)
        assert multi_turn > 0, "Expected some multi-turn conversations"

    def test_average_turns_reasonable(self, sample):
        avg_turns = sum(
            len(row["conversations"]) for row in sample
        ) / len(sample)
        # Conversations should average at least 2 turns (one Q, one A)
        assert avg_turns >= 2.0

    def test_conversation_lengths_vary(self, sample):
        lengths = {len(row["conversations"]) for row in sample}
        # Expect at least a few different conversation lengths in 100 samples
        assert len(lengths) > 1, "All conversations have the same length"


# ---------------------------------------------------------------------------
# Indexing / slicing tests
# ---------------------------------------------------------------------------

class TestDatasetOperations:
    def test_index_first_row(self, dataset):
        row = dataset[0]
        assert "conversations" in row

    def test_index_last_row(self, dataset):
        row = dataset[-1]
        assert "conversations" in row

    def test_slice_returns_correct_length(self, dataset):
        subset = dataset.select(range(10))
        assert len(subset) == 10

    def test_shuffle_preserves_length(self, dataset):
        shuffled = dataset.shuffle(seed=42)
        assert len(shuffled) == len(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
