# SPDX-License-Identifier: MIT
"""
Data schema definitions and validation for protein sequences.

Defines expected CSV columns and validation rules.
"""

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# Standard amino acids (20 canonical)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Extended set including some non-standard
EXTENDED_AMINO_ACIDS = STANDARD_AMINO_ACIDS | set("XBZJUO")


class ProteinSequenceRecord(BaseModel):
    """
    Schema for a single protein sequence record.

    Expected CSV columns: id, sequence, label
    """

    id: str = Field(..., description="Unique protein identifier")
    sequence: str = Field(..., min_length=30, description="Amino acid sequence")
    label: int = Field(..., ge=0, le=5, description="Enzyme class label (0-5)")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate sequence contains only valid amino acids."""
        v = v.upper()
        invalid_chars = set(v) - EXTENDED_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains invalid amino acids: {invalid_chars}. "
                f"Valid amino acids: {sorted(EXTENDED_AMINO_ACIDS)}"
            )
        return v


@dataclass
class DatasetMetadata:
    """Metadata about the dataset."""

    num_samples: int
    num_classes: int
    label_distribution: dict
    sequence_length_stats: dict
    invalid_sequences: int = 0


def validate_sequence(sequence: str, min_length: int = 30) -> tuple[bool, Optional[str]]:
    """
    Validate a protein sequence.

    Args:
        sequence: Amino acid sequence
        min_length: Minimum required length

    Returns:
        (is_valid, error_message)
    """
    sequence = sequence.upper()

    # Check length
    if len(sequence) < min_length:
        return False, f"Sequence too short: {len(sequence)} < {min_length}"

    # Check characters
    invalid_chars = set(sequence) - EXTENDED_AMINO_ACIDS
    if invalid_chars:
        return False, f"Invalid amino acids: {invalid_chars}"

    return True, None


def clean_sequence(sequence: str, replace_unknown: str = "X") -> str:
    """
    Clean a protein sequence by replacing unknown amino acids.

    Args:
        sequence: Raw amino acid sequence
        replace_unknown: Character to use for unknown amino acids (default: X)

    Returns:
        Cleaned sequence
    """
    sequence = sequence.upper()
    cleaned = "".join(c if c in EXTENDED_AMINO_ACIDS else replace_unknown for c in sequence)
    return cleaned


def get_label_mapping() -> dict:
    """
    Get label mapping from integers to enzyme class names.

    Returns:
        Dictionary mapping label indices to class names
    """
    return {
        0: "EC1_Oxidoreductases",
        1: "EC2_Transferases",
        2: "EC3_Hydrolases",
        3: "EC4_Lyases",
        4: "EC5_Isomerases",
        5: "EC6_Ligases",
    }
