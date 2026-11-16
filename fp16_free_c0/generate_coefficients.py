"""Constrained FP16 C1/C2 search with unrestricted C0 (optimal real midpoint).

For each d=1..11 and each of 128 segments, search (C1, C2) in the d-constrained
FP16 set and choose C0 as the exact continuous optimum (midpoint of residual
extrema). C0 is a free float64 value, not restricted to any FP16 candidate set.
"""
from pathlib import Path
import importlib.util
import csv
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
spec = importlib.util.spec_from_file_location('original_model', ROOT.parent / 'original/reproduce.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
H = 1 / 128


def allowed(d):
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[((bits & 0x7c00) != 0x7c00) & ((bits & ((1 << (11 - d)) - 1)) == 0)]
    return np.unique(bits.view(np.float16).astype(np.float64))


def half_range_error(a, c1, c2):
    """Exact continuous half-range error and optimal C0 for given (C1, C2).

    r(x) = 1/(a+x) - C1*x - C2*x^2 on [0, H]; optimal C0 = (max r + min r)/2
    and the error is (max r - min r)/2. Stationary points of r are found by
    vectorized bisection: r' has at most one turning point (r'' = 0), so it is
    monotone on each piece and each sign change brackets one root.
    """
    c1 = np.asarray(c1, dtype=np.float64)
    c2 = np.asarray(c2, dtype=np.float64)
    r = lambda x: 1 / (a + x) - c1 * x - c2 * x * x
    der = lambda x: -1 / (a + x) ** 2 - c1 - 2 * c2 * x
    turn = np.full(c2.shape, H)
    pos = c2 > 0
    turn[pos] = np.clip(np.cbrt(1 / c2[pos]) - a, 0.0, H)
    roots = []
    for lo, hi in [(np.zeros_like(c2), turn), (turn, np.full(c2.shape, H))]:
        active = der(lo) * der(hi) < 0
        lo, hi = lo.copy(), hi.copy()
        dlo = der(lo)
        for _ in range(28):
            mid = (lo + hi) / 2
            same = dlo * der(mid) > 0
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        roots.append(np.where(active, (lo + hi) / 2, np.nan))
    xs = [np.zeros_like(c2), np.full(c2.shape, H), roots[0], roots[1]]
    vals = np.stack([r(x) for x in xs])
    rmax = np.where(np.isnan(vals), -np.inf, vals).max(axis=0)
    rmin = np.where(np.isnan(vals), np.inf, vals).min(axis=0)
    return (rmax - rmin) / 2, (rmax + rmin) / 2


def minimax_bound(a):
    """Continuous minimax error (free C0, C1, C2) as a global lower bound."""
    b0, b1, b2 = original.initial_minimax(a)
    vals = original.residual_extrema(a, b1, b2)
    return b1, b2, float((max(vals) - min(vals)) / 2)


def solve_pruned(a, values, b1, b2):
    """Full search with sample-point pruning (necessary conditions, exact final check).

    If the continuous error is <= E then |r(x)-r(0)| <= 2E at every sample
    x, which for each C1 bounds C2 to an interval. Intersecting intervals
    over several samples prunes both variables; survivors are checked at
    every continuous extremum. Two passes: the second benefits from the
    tightened bound after the first.
    """
    n = len(values)
    v0 = float(values[np.argmin(abs(values - float(b1)))])
    u0 = float(values[np.argmin(abs(values - float(b2)))])
    err, c0 = half_range_error(a, np.array([v0]), np.array([u0]))
    best_err, best = float(err[0]), (float(c0[0]), v0, u0)
    samples = [H / 8, H / 4, H / 2, 3 * H / 4, H]
    for _ in range(2):
        E = best_err * (1 + 1e-9) + 1e-14
        for start in range(0, n, 256):
            c1 = values[start:start + 256]
            lo = np.full(len(c1), values[0])
            hi = np.full(len(c1), values[-1])
            for x in samples:
                center = (1 / (a + x) - 1 / a - c1 * x) / x**2
                rad = 2 * E / x**2
                lo = np.maximum(lo, center - rad)
                hi = np.minimum(hi, center + rad)
            left = np.searchsorted(values, lo, side='left')
            right = np.searchsorted(values, hi, side='right')
            counts = np.maximum(right - left, 0)
            if not np.any(counts):
                continue
            u = np.repeat(c1, counts)
            idx = np.concatenate([np.arange(l, r) for l, r in zip(left, right) if r > l])
            err, c0 = half_range_error(a, u, values[idx])
            j = int(np.argmin(err))
            if err[j] < best_err:
                best_err = float(err[j])
                best = (float(c0[j]), float(u[j]), float(values[idx][j]))
    return best, best_err


def solve_segment(a, values, b1, b2):
    n = len(values)
    if n > 991:
        return solve_pruned(a, values, b1, b2)
    best_err, best = np.inf, None
    for start in range(0, n, 256):
        c1, c2 = np.meshgrid(values[start:start + 256], values, indexing='ij')
        err, c0 = half_range_error(a, c1, c2)
        j = np.unravel_index(int(np.argmin(err)), err.shape)
        if err[j] < best_err:
            best_err = float(err[j])
            best = (float(c0[j]), float(c1[j]), float(c2[j]))
    return best, best_err


def main():
    OUT.mkdir(exist_ok=True)
    for d in range(1, 12):
        values = allowed(d)
        rows = []
        for i in range(128):
            a = 1 + mp.mpf(i) / 128
            b1, b2, bound = minimax_bound(a)
            (c0, c1, c2), err = solve_segment(float(a), values, b1, b2)
            rows.append((i, c0, c1, c2, err, bound))
        coeff = np.array([r[2:4] for r in rows])
        bits = coeff.astype(np.float16).view(np.uint16)
        assert np.all((bits & ((1 << (11 - d)) - 1)) == 0)
        assert np.array_equal(coeff, coeff.astype(np.float16).astype(np.float64))
        with (OUT / f'coefficients_d{d}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['segment', 'C0', 'C1', 'C2', 'continuous_max_abs_error',
                        'minimax_error_bound', 'C1_fp16_hex', 'C2_fp16_hex'])
            for row, b in zip(rows, bits):
                w.writerow([*row, *(f'{int(v):04x}' for v in b)])
        print(f'Generated d={d}: 128 segments', flush=True)


if __name__ == '__main__':
    main()
