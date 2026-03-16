import ast
import logging
import math
import re
from typing import Any

from slime.utils.types import Sample

logger = logging.getLogger(__name__)

CODE_BLOCK_RE = re.compile(r"```(?:text|txt|python|json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
BRACKET_MATRIX_RE = re.compile(r"\[\s*\[(.*?)\]\s*\]", re.DOTALL)
TAGGED_MATRIX_RE = re.compile(r"<matrix>(.*?)</matrix>", re.DOTALL | re.IGNORECASE)
BIN_ROW_RE = re.compile(r"(?<![01])([01](?:[\s,]+[01]){2,}|[01]{3,})(?![01])")
MAX_ENUM_K = 20


def _metadata(sample: Sample) -> dict[str, Any]:
    return sample.metadata if isinstance(sample.metadata, dict) else {}


def _normalize_bit(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        if value in (0, 1):
            return value
        raise ValueError(f"Invalid GF(2) entry {value}.")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
    raise ValueError(f"Invalid GF(2) entry {value!r}.")


def _parse_matrix_literal(text: str) -> list[list[int]] | None:
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None

    if not isinstance(parsed, list) or not parsed:
        return None

    rows: list[list[int]] = []
    for row in parsed:
        if isinstance(row, str):
            cleaned = row.replace(" ", "")
            if not cleaned or any(ch not in "01" for ch in cleaned):
                return None
            rows.append([int(ch) for ch in cleaned])
            continue

        if not isinstance(row, (list, tuple)) or not row:
            return None
        rows.append([_normalize_bit(item) for item in row])
    return rows


def _parse_binary_rows(text: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in text.splitlines():
        line = line.strip().strip("[]()")
        if not line:
            continue
        if all(ch in "01 " for ch in line) and any(ch in "01" for ch in line):
            cleaned = line.replace(" ", "")
            if len(cleaned) >= 2:
                rows.append([int(ch) for ch in cleaned])
            continue

        if set(line) <= {"0", "1", ",", " "} and any(ch in "01" for ch in line):
            items = [item for item in re.split(r"[\s,]+", line) if item]
            if len(items) >= 2 and all(item in {"0", "1"} for item in items):
                rows.append([int(item) for item in items])
    return rows


def _find_matrix_candidates(text: str) -> list[str]:
    candidates = []

    candidates.extend(match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text))
    candidates.extend(match.group(1).strip() for match in TAGGED_MATRIX_RE.finditer(text))
    candidates.extend(match.group(0).strip() for match in BRACKET_MATRIX_RE.finditer(text))

    if not candidates:
        candidates.append(text)

    return candidates


def parse_generator_matrix(response: str) -> list[list[int]]:
    for candidate in _find_matrix_candidates(response):
        matrix = _parse_matrix_literal(candidate)
        if matrix is not None:
            return _validate_matrix(matrix)

        rows = _parse_binary_rows(candidate)
        if rows:
            return _validate_matrix(rows)

        for match in BIN_ROW_RE.finditer(candidate):
            snippet = match.group(1)
            rows = _parse_binary_rows(snippet)
            if rows:
                return _validate_matrix(rows)

    raise ValueError(
        "Could not parse a GF(2) generator matrix from the model response. "
        "Supported formats include [[1,0,1],[0,1,1]] or line-based binary rows."
    )


def _validate_matrix(matrix: list[list[int]]) -> list[list[int]]:
    if not matrix:
        raise ValueError("Generator matrix is empty.")

    n = len(matrix[0])
    if n == 0:
        raise ValueError("Generator matrix has zero columns.")

    normalized = []
    for row in matrix:
        if len(row) != n:
            raise ValueError("Generator matrix rows have inconsistent lengths.")
        normalized.append([_normalize_bit(item) for item in row])

    return normalized


def _gf2_rank(matrix: list[list[int]]) -> int:
    rows = [row[:] for row in matrix]
    rank = 0
    num_rows = len(rows)
    num_cols = len(rows[0])

    for col in range(num_cols):
        pivot = None
        for row in range(rank, num_rows):
            if rows[row][col]:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        for row in range(num_rows):
            if row != rank and rows[row][col]:
                rows[row] = [a ^ b for a, b in zip(rows[row], rows[rank], strict=False)]
        rank += 1
        if rank == num_rows:
            break

    return rank


def compute_minimum_distance(matrix: list[list[int]]) -> int:
    k = len(matrix)
    if k == 0:
        raise ValueError("Generator matrix must contain at least one row.")
    if k > MAX_ENUM_K:
        raise ValueError(
            f"Generator matrix has k={k}, which is too large for exhaustive minimum-distance enumeration. "
            f"Current limit is k<={MAX_ENUM_K}."
        )

    min_weight = math.inf
    for mask in range(1, 1 << k):
        codeword = [0] * len(matrix[0])
        for row_idx in range(k):
            if (mask >> row_idx) & 1:
                codeword = [a ^ b for a, b in zip(codeword, matrix[row_idx], strict=False)]
        weight = sum(codeword)
        if 0 < weight < min_weight:
            min_weight = weight
            if min_weight == 1:
                return 1

    if min_weight is math.inf:
        raise ValueError("All generated codewords are zero; generator matrix is degenerate.")
    return int(min_weight)


def _resolve_target_distance(sample: Sample) -> int | None:
    metadata = _metadata(sample)

    for key in ("target_min_distance", "minimum_distance", "min_distance", "target_d", "d"):
        value = metadata.get(key)
        if value is not None:
            return int(value)

    label = sample.label
    if isinstance(label, int):
        return label
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.isdigit():
            return int(stripped)

    return None


def _resolve_expected_shape(sample: Sample) -> tuple[int | None, int | None]:
    metadata = _metadata(sample)
    expected_n = metadata.get("n")
    expected_k = metadata.get("k")
    return (int(expected_n) if expected_n is not None else None, int(expected_k) if expected_k is not None else None)


def _score_distance(actual_distance: int, target_distance: int | None, rank_ok: bool, shape_ok: bool) -> float:
    if not rank_ok or not shape_ok:
        return 0.0
    if target_distance is None:
        return float(actual_distance)
    return 1.0 if actual_distance >= target_distance else 0.0


def _set_diagnostics(sample: Sample, **diagnostics: Any) -> None:
    metadata = _metadata(sample)
    metadata.update(diagnostics)
    sample.metadata = metadata


async def custom_rm(args, sample: Sample | list[Sample], **kwargs):
    if isinstance(sample, list):
        return [await custom_rm(args, item, **kwargs) for item in sample]

    try:
        matrix = parse_generator_matrix(sample.response)
        rank = _gf2_rank(matrix)
        expected_n, expected_k = _resolve_expected_shape(sample)
        shape_ok = (expected_n is None or len(matrix[0]) == expected_n) and (expected_k is None or len(matrix) == expected_k)
        rank_ok = rank == len(matrix)
        min_distance = compute_minimum_distance(matrix) if rank_ok else 0
        target_distance = _resolve_target_distance(sample)
        reward = _score_distance(min_distance, target_distance, rank_ok=rank_ok, shape_ok=shape_ok)

        _set_diagnostics(
            sample,
            ecc_matrix=matrix,
            ecc_rank=rank,
            ecc_target_min_distance=target_distance,
            ecc_computed_min_distance=min_distance,
            ecc_shape_ok=shape_ok,
            ecc_rank_ok=rank_ok,
            rollout_correct=(reward == 1.0 if target_distance is not None else reward > 0.0),
        )
        return reward
    except Exception as exc:
        logger.info("ECC reward parsing/verification failed: %s", exc)
        _set_diagnostics(
            sample,
            ecc_error=str(exc),
            ecc_shape_ok=False,
            ecc_rank_ok=False,
            ecc_computed_min_distance=0,
            rollout_correct=False,
        )
        return 0.0
