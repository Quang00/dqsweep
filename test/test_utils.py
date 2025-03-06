import pytest

from experiments.utils import truncate_param


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
        """
        Test with different inputs (base case, given delimiter, token count).
        """

        result = truncate_param(name, char, n)
        assert result == expected

    def test_empty_string(self):
        """
        Test an empty string.
        """

    result = truncate_param("")
    assert result == ""

    def test_zero_token(self):
        """
        Test with n=0 aka 0 token.
        """

    result = truncate_param("single_qubit_gate_depolar_prob", n=0)
    assert result == ""
