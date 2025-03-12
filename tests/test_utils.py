import os

import numpy as np
import pytest

from experiments.utils import create_subdir, parse_range, truncate_param


class TestTruncateParam:
    @pytest.mark.parametrize(
        "name, char, n, expected",
        [
            # Test default behavior with '_' delimiter and n=4 tokens.
            (
                "single_qubit_gate_depolar_prob",
                "_",
                4,
                "Single qubit gate depolar",
            ),
            # Test custom delimiter.
            (
                "single-qubit-gate-depolar-prob",
                "-",
                4,
                "Single qubit gate depolar",
            ),
            # Test custom token count.
            (
                "single_qubit_gate_depolar_prob",
                "_",
                3,
                "Single qubit gate",
            ),
        ],
    )
    def test_various_cases(self, name, char, n, expected):
        """Test with different inputs (base case, given delimiter, token count)
        """

        result = truncate_param(name, char, n)
        np.testing.assert_string_equal(result, expected)

    def test_empty_string(self):
        """Test an empty string."""

    result = truncate_param("")
    np.testing.assert_string_equal(result, "")

    def test_zero_token(self):
        """Test with n=0 aka 0 token."""

    result = truncate_param("single_qubit_gate_depolar_prob", n=0)
    np.testing.assert_string_equal(result, "")


class TestParseRange:
    def test_range_linspace(self):
        """Test base case range linear space."""

        parse_range.__globals__["LOG_SCALE_PARAMS"] = []
        range = "0,10,5"
        param = "single_qubit_gate_depolar_prob"
        result = parse_range(range, param)
        expected = np.linspace(0, 10, 5)
        np.testing.assert_allclose(result, expected)

    def test_range_logspace(self):
        """Test base case range log space."""

        parse_range.__globals__["LOG_SCALE_PARAMS"] = ["T1"]
        range = "0,2,5"
        param = "T1"
        result = parse_range(range, param)
        expected = np.logspace(0, 2, 5)
        np.testing.assert_allclose(result, expected)


class TestCreateSubdir:
    def test_duplicate(self, tmp_path):
        """Test create_subdir when the experiment directory already exists.
        """
        directory = str(tmp_path / "results")
        experiment = "experiment"
        sweep_params = "Param"
        expected_dir = os.path.join(directory, f"{experiment}_Param")

        first_subdir = create_subdir(directory, experiment, sweep_params)
        np.testing.assert_equal(first_subdir, expected_dir)
        np.testing.assert_equal(os.path.exists(first_subdir), True)

        second_subdir = create_subdir(directory, experiment, sweep_params)
        expected_second = f"{expected_dir}_1"
        np.testing.assert_equal(second_subdir, expected_second)
        np.testing.assert_equal(os.path.exists(second_subdir), True)
