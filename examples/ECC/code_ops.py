"""GF(2) linear code operations on generator matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np

DEFAULT_MAX_K_ENUM = 16


# ---------------------------------------------------------------------------
# Basic GF(2) linear algebra
# ---------------------------------------------------------------------------

def normalize_binary_matrix(matrix: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Return a 2-D uint8 matrix with entries reduced modulo 2.

    Parameters
    ----------
    matrix:
        Any 2-D array-like object containing integer-like values.

    Returns
    -------
    np.ndarray
        A copy in shape (rows, cols), dtype uint8, with values in {0, 1}.
    """
    arr = np.array(matrix, dtype=np.uint8, copy=True)
    if arr.ndim != 2:
        raise ValueError("Generator matrix must be 2-dimensional.")
    return arr % 2


def rref_gf2(matrix: np.ndarray | Sequence[Sequence[int]]) -> tuple[np.ndarray, List[int]]:
    """Compute the reduced row echelon form of a matrix over GF(2).

    Parameters
    ----------
    matrix:
        2-D array-like matrix over GF(2).

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (rref_matrix, pivot_columns) where the matrix is in reduced row echelon form.
    """
    a = normalize_binary_matrix(matrix)
    rows, cols = a.shape
    pivot_cols: List[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        pivot_candidates = np.where(a[r:, c] == 1)[0]
        if pivot_candidates.size == 0:
            continue
        pivot = r + int(pivot_candidates[0])
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for rr in range(rows):
            if rr != r and a[rr, c]:
                a[rr] ^= a[r]
        pivot_cols.append(c)
        r += 1
    return a, pivot_cols


def row_basis_gf2(matrix: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Return an independent row basis spanning the input row space over GF(2)."""
    rref, pivots = rref_gf2(matrix)
    if not pivots:
        return np.zeros((0, normalize_binary_matrix(matrix).shape[1]), dtype=np.uint8)
    basis = []
    for row in rref:
        if np.any(row):
            basis.append(row)
    return np.array(basis, dtype=np.uint8)


def rank_gf2(matrix: np.ndarray | Sequence[Sequence[int]]) -> int:
    """Return the rank of a matrix over GF(2)."""
    _, pivots = rref_gf2(matrix)
    return len(pivots)


def in_row_space_gf2(
    vector: np.ndarray | Sequence[int], matrix: np.ndarray | Sequence[Sequence[int]]
) -> bool:
    """Test whether ``vector`` belongs to the row space of ``matrix`` over GF(2)."""
    m = normalize_binary_matrix(matrix)
    v = np.array(vector, dtype=np.uint8).reshape(1, -1) % 2
    if m.shape[1] != v.shape[1]:
        raise ValueError("Vector length must match matrix width.")
    rank_before = rank_gf2(m)
    rank_after = rank_gf2(np.vstack([m, v]))
    return rank_after == rank_before


def kronecker_gf2(
    a: np.ndarray | Sequence[Sequence[int]], b: np.ndarray | Sequence[Sequence[int]]
) -> np.ndarray:
    """Compute the Kronecker product over GF(2)."""
    aa = normalize_binary_matrix(a)
    bb = normalize_binary_matrix(b)
    return (np.kron(aa, bb) % 2).astype(np.uint8)


@dataclass(frozen=True)
class LinearCode:
    """Binary linear code represented by a GF(2) generator matrix."""

    G: np.ndarray

    def __post_init__(self) -> None:
        g = normalize_binary_matrix(self.G)
        g = row_basis_gf2(g)
        object.__setattr__(self, "G", g)

    @property
    def n(self) -> int:
        """Return code length."""
        return int(self.G.shape[1])

    @property
    def k(self) -> int:
        """Return code dimension."""
        return int(self.G.shape[0])

    def copy(self) -> "LinearCode":
        """Return a deep copy of this code."""
        return LinearCode(self.G.copy())

    def enumerate_codewords(self, max_k: int = DEFAULT_MAX_K_ENUM) -> np.ndarray:
        """Enumerate all codewords by linear combinations of generator rows.

        Parameters
        ----------
        max_k:
            Maximum supported dimension for full enumeration.

        Returns
        -------
        np.ndarray
            Matrix with one codeword per row.
        """
        if self.k > max_k:
            raise ValueError(
                f"Enumeration requires k <= {max_k}, but this code has k={self.k}."
            )
        if self.k == 0:
            return np.zeros((1, self.n), dtype=np.uint8)
        coeffs = np.array(
            [[(i >> j) & 1 for j in range(self.k)] for i in range(2**self.k)],
            dtype=np.uint8,
        )
        return (coeffs @ self.G) % 2


def from_generator_matrix(matrix: np.ndarray | Sequence[Sequence[int]]) -> LinearCode:
    """Construct a binary linear code from a generator matrix."""
    return LinearCode(matrix)


# ---------------------------------------------------------------------------
# Polynomial helpers for cyclic constructions
# ---------------------------------------------------------------------------

def _poly_trim_binary(poly: np.ndarray | Sequence[int]) -> np.ndarray:
    p = np.array(poly, dtype=np.uint8).reshape(-1) % 2
    i = len(p) - 1
    while i > 0 and p[i] == 0:
        i -= 1
    return p[: i + 1]


def _poly_degree_binary(poly: np.ndarray | Sequence[int]) -> int:
    p = _poly_trim_binary(poly)
    return len(p) - 1


def _poly_divmod_binary(
    f: np.ndarray | Sequence[int], g: np.ndarray | Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Polynomial division over GF(2), low-to-high coefficient convention."""
    ff = _poly_trim_binary(f)
    gg = _poly_trim_binary(g)
    if len(gg) == 0 or (len(gg) == 1 and gg[0] == 0):
        raise ZeroDivisionError("Polynomial division by zero.")
    df = _poly_degree_binary(ff)
    dg = _poly_degree_binary(gg)
    if df < dg:
        return np.array([0], dtype=np.uint8), ff
    q = np.zeros(df - dg + 1, dtype=np.uint8)
    r = ff.copy()
    while _poly_degree_binary(r) >= dg and not (len(r) == 1 and r[0] == 0):
        dr = _poly_degree_binary(r)
        shift = dr - dg
        q[shift] = 1
        r[shift : shift + dg + 1] ^= gg
        r = _poly_trim_binary(r)
    return _poly_trim_binary(q), _poly_trim_binary(r)


# ---------------------------------------------------------------------------
# Internal code-space helpers
# ---------------------------------------------------------------------------

def _ensure_same_length(c1: LinearCode, c2: LinearCode) -> None:
    if c1.n != c2.n:
        raise ValueError(f"Codes must have same length, got {c1.n} and {c2.n}.")


def _pad_to_length(code: LinearCode, n: int) -> LinearCode:
    if code.n > n:
        raise ValueError("Target length must be >= code length.")
    if code.n == n:
        return code.copy()
    pad = np.zeros((code.k, n - code.n), dtype=np.uint8)
    return LinearCode(np.hstack([code.G, pad]))


def _remove_columns(matrix: np.ndarray, positions: Sequence[int]) -> np.ndarray:
    if not positions:
        return matrix.copy()
    cols = matrix.shape[1]
    sorted_pos = sorted(positions)
    if len(set(sorted_pos)) != len(sorted_pos):
        raise ValueError("Positions must be distinct.")
    if sorted_pos[0] < 0 or sorted_pos[-1] >= cols:
        raise IndexError("Position out of range.")
    keep = [i for i in range(cols) if i not in set(sorted_pos)]
    return matrix[:, keep]


def _orthogonal_complement_basis(code: LinearCode) -> np.ndarray:
    """Return a basis of C^perp as rows."""
    rref, pivots = rref_gf2(code.G)
    free_cols = [c for c in range(code.n) if c not in pivots]
    if not free_cols:
        return np.zeros((0, code.n), dtype=np.uint8)
    basis = []
    for f in free_cols:
        x = np.zeros(code.n, dtype=np.uint8)
        x[f] = 1
        for r, p in enumerate(pivots):
            x[p] = rref[r, f]
        basis.append(x)
    return np.array(basis, dtype=np.uint8)


def _intersection(c1: LinearCode, c2: LinearCode) -> LinearCode:
    _ensure_same_length(c1, c2)
    h1 = _orthogonal_complement_basis(c1)
    h2 = _orthogonal_complement_basis(c2)
    h = row_basis_gf2(np.vstack([h1, h2])) if h1.size or h2.size else np.zeros((0, c1.n), dtype=np.uint8)
    return LinearCode(_orthogonal_complement_basis(LinearCode(h)))


def _span_from_words(words: np.ndarray, n: int) -> LinearCode:
    if words.size == 0:
        return LinearCode(np.zeros((0, n), dtype=np.uint8))
    return LinearCode(words)


def _is_subcode(subcode: LinearCode, code: LinearCode) -> bool:
    _ensure_same_length(subcode, code)
    return all(in_row_space_gf2(row, code.G) for row in subcode.G)


def _sum_codes(c1: LinearCode, c2: LinearCode) -> LinearCode:
    _ensure_same_length(c1, c2)
    return LinearCode(np.vstack([c1.G, c2.G]))


def _solve_left_coeffs_gf2(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Solve coeffs such that coeffs @ basis = vector over GF(2)."""
    v = np.array(vector, dtype=np.uint8).reshape(-1) % 2
    b = normalize_binary_matrix(basis)
    if b.shape[1] != v.shape[0]:
        raise ValueError("Vector length must match basis width.")
    t, _ = b.shape
    if t == 0:
        if np.any(v):
            raise ValueError("Vector is not in span of empty basis.")
        return np.zeros(0, dtype=np.uint8)
    aug = np.hstack([b.T.copy(), v.reshape(-1, 1)])
    rows, _ = aug.shape
    vars_count = t
    r = 0
    pivots: list[int] = []
    for c in range(vars_count):
        if r >= rows:
            break
        pivot_candidates = np.where(aug[r:, c] == 1)[0]
        if pivot_candidates.size == 0:
            continue
        p = r + int(pivot_candidates[0])
        if p != r:
            aug[[r, p]] = aug[[p, r]]
        for rr in range(rows):
            if rr != r and aug[rr, c]:
                aug[rr] ^= aug[r]
        pivots.append(c)
        r += 1
    for rr in range(rows):
        if not np.any(aug[rr, :vars_count]) and aug[rr, vars_count]:
            raise ValueError("Vector is not in span of basis.")
    sol = np.zeros(vars_count, dtype=np.uint8)
    for rr, c in enumerate(pivots):
        sol[c] = aug[rr, vars_count]
    return sol


def _map_rows_to_tail(rows: np.ndarray, source_basis: np.ndarray, tail_basis: np.ndarray) -> np.ndarray:
    if rows.size == 0:
        return np.zeros((0, tail_basis.shape[1]), dtype=np.uint8)
    mapped = []
    for row in rows:
        coeffs = _solve_left_coeffs_gf2(row, source_basis)
        mapped.append((coeffs @ tail_basis) % 2)
    return np.array(mapped, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Family constructors
# ---------------------------------------------------------------------------

def cyclic_code_binary(n: int, generator_poly: Sequence[int]) -> LinearCode:
    """Construct a binary cyclic code of length ``n`` by generator polynomial.

    ``generator_poly`` uses low-to-high coefficient order.
    """
    g = _poly_trim_binary(generator_poly)
    if len(g) == 1 and g[0] == 0:
        raise ValueError("generator_poly must be non-zero.")
    xn_minus_1 = np.zeros(n + 1, dtype=np.uint8)
    xn_minus_1[0] = 1  # -1 == 1 in GF(2)
    xn_minus_1[n] = 1
    _, rem = _poly_divmod_binary(xn_minus_1, g)
    if np.any(rem != 0):
        raise ValueError("generator_poly must divide x^n - 1 over GF(2).")
    k = n - _poly_degree_binary(g)
    rows = []
    for i in range(k):
        row = np.zeros(n, dtype=np.uint8)
        row[i : i + len(g)] = g
        rows.append(row)
    return LinearCode(np.array(rows, dtype=np.uint8))


def constacyclic_code_binary(
    n: int, generator_poly: Sequence[int], shift_constant: int
) -> LinearCode:
    """Construct a binary constacyclic code by generator polynomial.

    In GF(2), the only non-zero shift constant is 1, so this reduces to cyclic.
    """
    lam = int(shift_constant) % 2
    if lam == 0:
        raise ValueError("In GF(2), shift_constant must be 1 for non-trivial constacyclic codes.")
    g = _poly_trim_binary(generator_poly)
    xn_minus_lam = np.zeros(n + 1, dtype=np.uint8)
    xn_minus_lam[0] = lam
    xn_minus_lam[n] = 1
    _, rem = _poly_divmod_binary(xn_minus_lam, g)
    if np.any(rem != 0):
        raise ValueError("generator_poly must divide x^n - shift_constant over GF(2).")
    deg_g = _poly_degree_binary(g)
    k = n - deg_g
    rows = []
    for i in range(k):
        ext_len = max(n, i + len(g))
        row = np.zeros(ext_len, dtype=np.uint8)
        row[i : i + len(g)] = g
        for j in range(ext_len - 1, n - 1, -1):
            row[j - n] ^= row[j]  # lam=1
        rows.append(row[:n])
    return LinearCode(np.array(rows, dtype=np.uint8))


def _hamming_parity_check_matrix_binary(r: int) -> np.ndarray:
    if r <= 0:
        raise ValueError("r must be positive.")
    n = (1 << r) - 1
    h = np.zeros((r, n), dtype=np.uint8)
    for j in range(n):
        val = j + 1
        for i in range(r):
            h[i, j] = val & 1
            val >>= 1
    return h


def from_parity_check_matrix_binary(h: np.ndarray | Sequence[Sequence[int]]) -> LinearCode:
    """Construct a binary linear code from parity-check matrix H."""
    hh = normalize_binary_matrix(h)
    return LinearCode(_orthogonal_complement_basis(LinearCode(hh)))


def hamming_code_binary(r: int) -> LinearCode:
    """Construct binary Hamming code with redundancy ``r``."""
    return from_parity_check_matrix_binary(_hamming_parity_check_matrix_binary(r))


def simplex_code_binary(r: int) -> LinearCode:
    """Construct binary simplex code (dual of Hamming code)."""
    return LinearCode(_hamming_parity_check_matrix_binary(r))


def double_circulant_code_binary(first_row: Sequence[int]) -> LinearCode:
    """Construct deterministic binary double-circulant code G=[I|C]."""
    row = np.array(first_row, dtype=np.uint8).reshape(-1) % 2
    k = len(row)
    if k == 0:
        raise ValueError("first_row must be non-empty.")
    c = np.zeros((k, k), dtype=np.uint8)
    for i in range(k):
        c[i, :] = np.roll(row, i)
    g = np.hstack([np.eye(k, dtype=np.uint8), c])
    return LinearCode(g)


# ---------------------------------------------------------------------------
# Basic code transformations
# ---------------------------------------------------------------------------

def augment_code(code: LinearCode) -> LinearCode:
    """Add the all-ones vector if it is not already in the code."""
    ones = np.ones(code.n, dtype=np.uint8)
    if in_row_space_gf2(ones, code.G):
        return code.copy()
    return LinearCode(np.vstack([code.G, ones]))


def code_complement(code: LinearCode, subcode: LinearCode) -> LinearCode:
    """Return a complement C2 such that code = subcode + C2.

    This function assumes ``subcode`` is a subspace of ``code``.
    """
    _ensure_same_length(code, subcode)
    if not np.all([in_row_space_gf2(row, code.G) for row in subcode.G]):
        raise ValueError("subcode must be a subcode of code.")
    basis = subcode.G.copy()
    comp_rows: list[np.ndarray] = []
    for row in code.G:
        if in_row_space_gf2(row, basis):
            continue
        comp_rows.append(row)
        basis = row_basis_gf2(np.vstack([basis, row])) if basis.size else row.reshape(1, -1)
    if not comp_rows:
        return LinearCode(np.zeros((0, code.n), dtype=np.uint8))
    return LinearCode(np.array(comp_rows, dtype=np.uint8))


def direct_sum(c: LinearCode, d: LinearCode) -> LinearCode:
    """Construct the direct sum of two binary linear codes."""
    left = np.hstack([c.G, np.zeros((c.k, d.n), dtype=np.uint8)])
    right = np.hstack([np.zeros((d.k, c.n), dtype=np.uint8), d.G])
    return LinearCode(np.vstack([left, right]))


def direct_sum_many(codes: Sequence[LinearCode]) -> LinearCode:
    """Construct the direct sum of a sequence of codes."""
    if not codes:
        raise ValueError("At least one code is required.")
    result = codes[0]
    for code in codes[1:]:
        result = direct_sum(result, code)
    return result


def direct_product(c: LinearCode, d: LinearCode) -> LinearCode:
    """Construct the direct product code using the Kronecker generator matrix."""
    return LinearCode(kronecker_gf2(c.G, d.G))


def product_code(c: LinearCode, d: LinearCode) -> LinearCode:
    """Alias of :func:`direct_product`."""
    return direct_product(c, d)


def extend_code(code: LinearCode) -> LinearCode:
    """Extend code by one parity coordinate to force even total parity."""
    parity = (code.G.sum(axis=1, dtype=np.uint32) % 2).astype(np.uint8).reshape(-1, 1)
    return LinearCode(np.hstack([code.G, parity]))


def extend_code_n(code: LinearCode, n: int) -> LinearCode:
    """Apply extension repeatedly ``n`` times."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    out = code.copy()
    for _ in range(n):
        out = extend_code(out)
    return out


def pad_code(code: LinearCode, n: int) -> LinearCode:
    """Pad codewords with ``n`` trailing zeros."""
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return code.copy()
    pad = np.zeros((code.k, n), dtype=np.uint8)
    return LinearCode(np.hstack([code.G, pad]))


def expurgate_code_odd_weight(
    code: LinearCode, max_k_enum: int = DEFAULT_MAX_K_ENUM
) -> LinearCode:
    """Delete odd-weight codewords and return the linear span of remaining words."""
    words = code.enumerate_codewords(max_k=max_k_enum)
    even_words = words[(words.sum(axis=1, dtype=np.uint32) % 2) == 0]
    return _span_from_words(even_words, code.n)


def expurgate_code_list(
    code: LinearCode,
    words_to_delete: Iterable[Sequence[int]],
    max_k_enum: int = DEFAULT_MAX_K_ENUM,
) -> LinearCode:
    """Remove the subspace generated by the listed codewords.

    Returning the span of the remaining codewords is not stable for linear
    codes, because deleted codewords can be regenerated by linear
    combinations. In this GF(2)-only helper library, we instead interpret the
    deleted list as generators of a subcode and return a complementary subcode.
    """
    _ = max_k_enum
    delete_rows = []
    for word in words_to_delete:
        row = np.array(word, dtype=np.uint8).reshape(-1) % 2
        if len(row) != code.n:
            raise ValueError("Each word to delete must match code length.")
        if not in_row_space_gf2(row, code.G):
            raise ValueError("Each word to delete must belong to the code.")
        if np.any(row):
            delete_rows.append(row)

    if not delete_rows:
        return code.copy()

    deleted_subcode = LinearCode(np.array(delete_rows, dtype=np.uint8))
    return code_complement(code, deleted_subcode)


def expurgate_weight_code(
    code: LinearCode, weight: int, max_k_enum: int = DEFAULT_MAX_K_ENUM
) -> LinearCode:
    """Remove the 1-D subcode generated by a codeword of the given weight."""
    if weight < 0 or weight > code.n:
        raise ValueError("weight must be between 0 and code length.")
    words = code.enumerate_codewords(max_k=max_k_enum)
    target = None
    for word in words:
        if int(word.sum()) == weight and np.any(word):
            target = word
            break
    if target is None:
        raise ValueError(f"No non-zero codeword with weight {weight} found.")
    return code_complement(code, LinearCode(target.reshape(1, -1)))


def lengthen_code(code: LinearCode) -> LinearCode:
    """Lengthen a binary code by augmenting with ones and then extending."""
    return extend_code(augment_code(code))


def plotkin_sum(c1: LinearCode, c2: LinearCode) -> LinearCode:
    """Construct Plotkin sum of two codes: {u | u+v} with automatic zero padding."""
    n = max(c1.n, c2.n)
    c1p = _pad_to_length(c1, n)
    c2p = _pad_to_length(c2, n)
    top = np.hstack([c1p.G, c1p.G])
    bottom = np.hstack([np.zeros((c2p.k, n), dtype=np.uint8), c2p.G])
    return LinearCode(np.vstack([top, bottom]))


def plotkin_sum_three(c1: LinearCode, c2: LinearCode, c3: LinearCode, a: int = -1) -> LinearCode:
    """Construct three-code Plotkin sum: {u | u+a*v | u+v+w}.

    In GF(2), any non-zero ``a`` is equivalent to 1.
    """
    if a % 2 == 0:
        raise ValueError("In GF(2), parameter a must be odd (non-zero modulo 2).")
    n12 = max(c1.n, c2.n)
    n = max(n12, c3.n)
    c1p = _pad_to_length(c1, n)
    c2p = _pad_to_length(c2, n)
    c3p = _pad_to_length(c3, n)
    top = np.hstack([c1p.G, c1p.G, c1p.G])
    mid = np.hstack([np.zeros((c2p.k, n), dtype=np.uint8), c2p.G, c2p.G])
    bot = np.hstack([np.zeros((c3p.k, 2 * n), dtype=np.uint8), c3p.G])
    return LinearCode(np.vstack([top, mid, bot]))


def puncture_code(code: LinearCode, i: int) -> LinearCode:
    """Puncture code at one 0-based coordinate."""
    return puncture_code_set(code, [i])


def puncture_code_set(code: LinearCode, positions: Sequence[int]) -> LinearCode:
    """Puncture code at multiple distinct 0-based coordinates."""
    if len(positions) >= code.n:
        raise ValueError("Cannot puncture all coordinates.")
    return LinearCode(_remove_columns(code.G, positions))


def shorten_code(
    code: LinearCode, i: int, max_k_enum: int = DEFAULT_MAX_K_ENUM
) -> LinearCode:
    """Shorten code at one 0-based coordinate."""
    return shorten_code_set(code, [i], max_k_enum=max_k_enum)


def shorten_code_set(
    code: LinearCode, positions: Sequence[int], max_k_enum: int = DEFAULT_MAX_K_ENUM
) -> LinearCode:
    """Shorten code at multiple 0-based coordinates.

    Only codewords with zeros on all selected positions are kept, then these
    coordinates are deleted and the linear span is returned.
    """
    if not positions:
        return code.copy()
    pos = sorted(positions)
    if len(set(pos)) != len(pos):
        raise ValueError("Positions must be distinct.")
    if pos[0] < 0 or pos[-1] >= code.n:
        raise IndexError("Position out of range.")
    if len(pos) >= code.n:
        raise ValueError("Cannot shorten on all coordinates.")
    words = code.enumerate_codewords(max_k=max_k_enum)
    mask = np.all(words[:, pos] == 0, axis=1)
    kept = words[mask]
    punctured = _remove_columns(kept, pos)
    return _span_from_words(punctured, code.n - len(pos))


def intersection_code(c1: LinearCode, c2: LinearCode) -> LinearCode:
    """Return intersection c1 ∩ c2 over GF(2)."""
    return _intersection(c1, c2)


def concatenate_codes(c1: LinearCode, c2: LinearCode) -> LinearCode:
    """Concatenate two codes by pairwise row concatenation.

    If A, B are generator matrices, the returned generator rows are
    ``a | b`` for every row ``a`` in A and every row ``b`` in B.
    """
    if c1.k == 0 or c2.k == 0:
        return LinearCode(np.zeros((0, c1.n + c2.n), dtype=np.uint8))
    rows = []
    for r1 in c1.G:
        for r2 in c2.G:
            rows.append(np.hstack([r1, r2]))
    return LinearCode(np.array(rows, dtype=np.uint8))


def juxtaposition(c1: LinearCode, c2: LinearCode) -> LinearCode:
    """Horizontally join two codes with the same dimension."""
    if c1.k != c2.k:
        raise ValueError(f"Juxtaposition requires equal dimensions, got {c1.k} and {c2.k}.")
    return LinearCode(np.hstack([c1.G, c2.G]))


# ---------------------------------------------------------------------------
# Compound constructions
# ---------------------------------------------------------------------------

def concatenated_code(outer: LinearCode, inner: LinearCode) -> LinearCode:
    """Construct a concatenated code for binary outer and k=1 inner code.

    This GF(2)-only implementation supports the practical case where the inner
    code has dimension 1.
    """
    if inner.k != 1:
        raise NotImplementedError(
            "Binary concatenated_code currently supports only inner.k == 1."
        )
    return LinearCode(np.kron(outer.G, inner.G[0]) % 2)


def construction_x(c1: LinearCode, c2: LinearCode, c3: LinearCode) -> LinearCode:
    """Construct a GF(2) version of Construction X from c1, c2, c3."""
    _ensure_same_length(c1, c2)
    if not _is_subcode(c2, c1):
        raise ValueError("c2 must be a subcode of c1.")
    u = code_complement(c1, c2)
    if c3.k < u.k:
        raise ValueError("c3 dimension is too small for the quotient c1/c2.")
    tail = c3.G[: u.k]
    top = np.hstack([c2.G, np.zeros((c2.k, c3.n), dtype=np.uint8)])
    bottom = np.hstack([u.G, tail])
    return LinearCode(np.vstack([top, bottom]))


def construction_x_chain(codes: Sequence[LinearCode], c: LinearCode) -> list[LinearCode]:
    """Apply Construction X to a chain S[0] >= S[1] >= ... using suffix code c.

    Returns a sequence of lifted codes with the same order as ``codes``.
    """
    if len(codes) < 2:
        raise ValueError("At least two chain codes are required.")
    for prev, cur in zip(codes, codes[1:], strict=False):
        if not _is_subcode(cur, prev):
            raise ValueError("codes must form a descending chain: codes[i+1] <= codes[i].")
    c1, c2 = codes[0], codes[1]
    u = code_complement(c1, c2)
    if c.k < u.k:
        raise ValueError("Suffix code dimension is too small.")
    tail_basis = c.G[: u.k]
    lifted = []
    for s in codes:
        if not _is_subcode(s, c1):
            raise ValueError("Every chain code must be a subcode of codes[0].")
        s_in_c2 = intersection_code(s, c2)
        s_u = code_complement(s, s_in_c2)
        s_tail = _map_rows_to_tail(s_u.G, u.G, tail_basis) if s_u.k else np.zeros((0, c.n), dtype=np.uint8)
        part_a = np.hstack([s_in_c2.G, np.zeros((s_in_c2.k, c.n), dtype=np.uint8)])
        part_b = np.hstack([s_u.G, s_tail]) if s_u.k else np.zeros((0, c1.n + c.n), dtype=np.uint8)
        lifted.append(LinearCode(np.vstack([part_a, part_b])))
    return lifted


def construction_x3(c1: LinearCode, c2: LinearCode, c3: LinearCode, d1: LinearCode, d2: LinearCode) -> LinearCode:
    """Construct a GF(2) version of Construction X3."""
    _ensure_same_length(c1, c2)
    _ensure_same_length(c2, c3)
    if not _is_subcode(c3, c2) or not _is_subcode(c2, c1):
        raise ValueError("Codes must satisfy c3 <= c2 <= c1.")
    u1 = code_complement(c1, c2)  # k1-k2
    u2 = code_complement(c2, c3)  # k2-k3
    if d1.k < u1.k:
        raise ValueError("d1 dimension is too small (requires >= k1-k2).")
    if d2.k < u2.k:
        raise ValueError("d2 dimension is too small (requires >= k2-k3).")
    base = np.hstack([c3.G, np.zeros((c3.k, d1.n + d2.n), dtype=np.uint8)])
    part_u2 = np.hstack(
        [u2.G, np.zeros((u2.k, d1.n), dtype=np.uint8), d2.G[: u2.k]]
    )
    part_u1 = np.hstack(
        [u1.G, d1.G[: u1.k], np.zeros((u1.k, d2.n), dtype=np.uint8)]
    )
    return LinearCode(np.vstack([base, part_u2, part_u1]))


def construction_x3u(
    c1: LinearCode, c2: LinearCode, c3: LinearCode, d1: LinearCode, d2: LinearCode
) -> tuple[LinearCode, LinearCode]:
    """Return a pair of lifted codes following the Construction X3u pattern."""
    if not _is_subcode(c3, c1) or not _is_subcode(c3, c2):
        raise ValueError("c3 must be a subcode of both c1 and c2.")
    low = construction_x(c1, c3, d1)
    high = construction_x(c2, c3, d2)
    if low.n != high.n:
        return low, high
    if not _is_subcode(low, high):
        # Keep the pair usable even if user-provided chains are inconsistent.
        return low, _sum_codes(low, high)
    return low, high


def construction_xx(c1: LinearCode, c2: LinearCode, c3: LinearCode, d2: LinearCode, d3: LinearCode) -> LinearCode:
    """Construct a practical GF(2) variant of Construction XX.

    This implementation maps complements of c2 and c3 in c1 to tails from d2 and d3.
    """
    _ensure_same_length(c1, c2)
    _ensure_same_length(c1, c3)
    if not _is_subcode(c2, c1) or not _is_subcode(c3, c1):
        raise ValueError("c2 and c3 must be subcodes of c1.")
    u2 = code_complement(c1, c2)
    u3 = code_complement(c1, c3)
    if d2.k < u2.k:
        raise ValueError("d2 dimension is too small for c1/c2.")
    if d3.k < u3.k:
        raise ValueError("d3 dimension is too small for c1/c3.")
    core = intersection_code(c2, c3)
    c2_only = code_complement(c2, core)
    c3_only = code_complement(c3, core)
    rows = []
    for block in (core.G, c2_only.G, c3_only.G):
        for row in block:
            rows.append(np.hstack([row, np.zeros(d2.n + d3.n, dtype=np.uint8)]))
    for i, row in enumerate(u2.G):
        rows.append(np.hstack([row, d2.G[i], np.zeros(d3.n, dtype=np.uint8)]))
    for i, row in enumerate(u3.G):
        rows.append(np.hstack([row, np.zeros(d2.n, dtype=np.uint8), d3.G[i]]))
    if not rows:
        return LinearCode(np.zeros((0, c1.n + d2.n + d3.n), dtype=np.uint8))
    return LinearCode(np.array(rows, dtype=np.uint8))


def zinoviev_code(inner: Sequence[LinearCode], outer: Sequence[LinearCode]) -> LinearCode:
    """Construct a binary generalized concatenated code (simplified Zinoviev form).

    Requirements:
    - ``inner`` is a strictly increasing chain by inclusion and dimension:
      inner[0] <= inner[1] <= ... , all with equal length.
    - all ``outer`` codes are binary and have the same length N.
    """
    if len(inner) == 0 or len(inner) != len(outer):
        raise ValueError("inner and outer must have the same non-zero length.")
    n = inner[0].n
    for i in range(1, len(inner)):
        if inner[i].n != n or inner[i].k <= inner[i - 1].k or not _is_subcode(inner[i - 1], inner[i]):
            raise ValueError("inner must satisfy inner[i-1] <= inner[i], with strictly increasing dimension.")
    N = outer[0].n
    if any(o.n != N for o in outer):
        raise ValueError("All outer codes must have the same length.")
    rows: list[np.ndarray] = []
    prev = LinearCode(np.zeros((0, n), dtype=np.uint8))
    for idx, (ii, oo) in enumerate(zip(inner, outer)):
        ui = code_complement(ii, prev)
        ei = ui.k
        if ei == 0:
            prev = ii
            continue
        for orow in oo.G:
            for urow in ui.G:
                block = np.hstack([bit * urow for bit in orow]) % 2
                rows.append(block.astype(np.uint8))
        prev = ii
    if not rows:
        return LinearCode(np.zeros((0, n * N), dtype=np.uint8))
    return LinearCode(np.array(rows, dtype=np.uint8))


def dual_code(code: LinearCode) -> LinearCode:
    """Return the dual code C^perp."""
    return LinearCode(_orthogonal_complement_basis(code))


def construction_y1(code: LinearCode, weight: int | None = None, max_k_enum: int = DEFAULT_MAX_K_ENUM) -> LinearCode:
    """Apply Construction Y1 using a dual-code word support.

    If ``weight`` is omitted, a minimum non-zero dual weight is used.
    """
    d = dual_code(code)
    words = d.enumerate_codewords(max_k=max_k_enum)
    non_zero = words[np.any(words == 1, axis=1)]
    if non_zero.size == 0:
        raise ValueError("Dual code has no non-zero words.")
    weights = non_zero.sum(axis=1, dtype=np.uint32)
    if weight is None:
        target_weight = int(weights.min())
    else:
        target_weight = int(weight)
    idx = np.where(weights == target_weight)[0]
    if idx.size == 0:
        raise ValueError(f"No dual codeword of weight {target_weight} found.")
    support = np.where(non_zero[int(idx[0])] == 1)[0].tolist()
    return shorten_code_set(code, support, max_k_enum=max_k_enum)


FAMILY_REGISTRY: dict[str, Callable[..., LinearCode]] = {
    "cyclic_binary": cyclic_code_binary,
    "constacyclic_binary": constacyclic_code_binary,
    "hamming_binary": hamming_code_binary,
    "simplex_binary": simplex_code_binary,
    "double_circulant_binary": double_circulant_code_binary,
}


def build_family(family_name: str, **kwargs) -> LinearCode:
    """Build an initial code family from the registry."""
    if family_name not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family: {family_name}")
    return FAMILY_REGISTRY[family_name](**kwargs)


__all__ = [
    "DEFAULT_MAX_K_ENUM",
    "FAMILY_REGISTRY",
    "LinearCode",
    "augment_code",
    "build_family",
    "concatenate_codes",
    "concatenated_code",
    "constacyclic_code_binary",
    "code_complement",
    "construction_x",
    "construction_x_chain",
    "construction_x3",
    "construction_x3u",
    "construction_xx",
    "construction_y1",
    "cyclic_code_binary",
    "direct_product",
    "direct_sum",
    "direct_sum_many",
    "double_circulant_code_binary",
    "dual_code",
    "expurgate_code_list",
    "expurgate_code_odd_weight",
    "expurgate_weight_code",
    "extend_code",
    "extend_code_n",
    "from_generator_matrix",
    "from_parity_check_matrix_binary",
    "hamming_code_binary",
    "in_row_space_gf2",
    "intersection_code",
    "juxtaposition",
    "kronecker_gf2",
    "lengthen_code",
    "normalize_binary_matrix",
    "pad_code",
    "plotkin_sum",
    "plotkin_sum_three",
    "product_code",
    "puncture_code",
    "puncture_code_set",
    "rank_gf2",
    "rref_gf2",
    "row_basis_gf2",
    "shorten_code",
    "shorten_code_set",
    "simplex_code_binary",
    "zinoviev_code",
]
