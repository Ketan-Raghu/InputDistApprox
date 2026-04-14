"""Tests for the LLM-LAT/harmful-dataset from HuggingFace."""

import pytest
from datasets import load_dataset


DATASET_NAME = "" \
""


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
        # The dataset page reports ~4.9k rows
        assert len(dataset) > 4_000

    def test_has_expected_columns(self, dataset):
        for col in ("prompt", "rejected", "chosen"):
            assert col in dataset.column_names

    def test_only_expected_columns(self, dataset):
        assert set(dataset.column_names) == {"prompt", "rejected", "chosen"}


# ---------------------------------------------------------------------------
# Field type tests
# ---------------------------------------------------------------------------

class TestFieldTypes:
    def test_prompt_is_string(self, sample):
        for row in sample:
            assert isinstance(row["prompt"], str)

    def test_rejected_is_string(self, sample):
        for row in sample:
            assert isinstance(row["rejected"], str)

    def test_chosen_is_string(self, sample):
        for row in sample:
            assert isinstance(row["chosen"], str)

    def test_prompt_non_empty(self, sample):
        for row in sample:
            assert len(row["prompt"].strip()) > 0

    def test_rejected_non_empty(self, sample):
        for row in sample:
            assert len(row["rejected"].strip()) > 0

    def test_chosen_non_empty(self, sample):
        for row in sample:
            assert len(row["chosen"].strip()) > 0


# ---------------------------------------------------------------------------
# Content quality tests
# ---------------------------------------------------------------------------

class TestContentQuality:
    def test_chosen_and_rejected_differ(self, sample):
        same = 0
        for row in sample:
            if row["chosen"].strip() == row["rejected"].strip():
                same += 1
        # Chosen and rejected should almost always differ
        assert same / len(sample) < 0.05, (
            f"Too many identical chosen/rejected pairs: {same}/{len(sample)}"
        )

    def test_prompt_lengths_vary(self, sample):
        lengths = {len(row["prompt"]) for row in sample}
        assert len(lengths) > 1, "All prompts have the same length"

    def test_average_prompt_length_reasonable(self, sample):
        avg_len = sum(len(row["prompt"]) for row in sample) / len(sample)
        # Prompts should have some substance
        assert avg_len > 10

    def test_average_chosen_length_reasonable(self, sample):
        avg_len = sum(len(row["chosen"]) for row in sample) / len(sample)
        # Chosen responses should have some substance
        assert avg_len > 10

    def test_average_rejected_length_reasonable(self, sample):
        avg_len = sum(len(row["rejected"]) for row in sample) / len(sample)
        # Rejected responses should have some substance
        assert avg_len > 10


# ---------------------------------------------------------------------------
# Indexing / slicing tests
# ---------------------------------------------------------------------------

class TestDatasetOperations:
    def test_index_first_row(self, dataset):
        row = dataset[0]
        assert "prompt" in row

    def test_index_last_row(self, dataset):
        row = dataset[-1]
        assert "prompt" in row

    def test_slice_returns_correct_length(self, dataset):
        subset = dataset.select(range(10))
        assert len(subset) == 10

    def test_shuffle_preserves_length(self, dataset):
        shuffled = dataset.shuffle(seed=42)
        assert len(shuffled) == len(dataset)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
