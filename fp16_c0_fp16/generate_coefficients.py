"""C0 from the full FP16 set (no d mask); C1/C2 from the d-constrained set.

Same search as ../fp16_constrained (endpoint/sample necessary-condition
pruning, exact final check at all continuous extrema), but the C0 candidate
set is the full FP16 set while C1/C2 keep the d mask.
"""
from pathlib import Path
import csv
import numpy as np

OUT = Path(__file__).resolve().parent / "results"
H = 1 / 128


def allowed(d):
    bits = np.arange(65536, dtype=np.uint16)
    bits = bits[((bits & 0x7c00) != 0x7c00) & ((bits & ((1 << (11 - d)) - 1)) == 0)]
    return np.unique(bits.view(np.float16).astype(np.float64))


def extrema_error(a, c0, c1, c2):
    """Vectorized continuous maximum: endpoints plus all stationary points.

    Residual derivative has at most one turning point, so bisection on
    the two monotone pieces finds every sign-changing stationary point.
    """
    def err(x):
        return c0 + c1*x + c2*x*x - 1/(a+x)

    def der(x):
        return c1 + 2*c2*x + 1/(a+x)**2
    turn = np.full_like(c2, H)
    pos = c2 > 0
    turn[pos] = np.clip(np.cbrt(1/c2[pos]) - a, 0, H)
    worst = np.maximum(abs(err(0)), abs(err(H)))
    for left, right in [(np.zeros_like(c2), turn), (turn, np.full_like(c2, H))]:
        active = der(left)*der(right) < 0
        lo, hi = left.copy(), right.copy()
        dl = der(lo)
        for _ in range(45):
            mid = (lo+hi)/2
            same = dl*der(mid) > 0
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
        worst = np.maximum(worst, np.where(active, abs(err((lo+hi)/2)), 0))
    return worst


def solve_segment(a, values0, values):
    if len(values) > 495:
        return solve_large(a, values0, values)
    # A feasible constant polynomial provides an upper bound. At x=0,
    # any better solution must have |C0-1/a| <= this bound.
    c0 = values0[np.argmin(abs(values0 - (1/a + 1/(a+H))/2))]
    best = max(abs(c0 - 1/a), abs(c0 - 1/(a+H)))
    result = (float(c0), 0., 0.)
    c1, c2 = np.meshgrid(values, values, indexing='ij')
    c1, c2 = c1.ravel(), c2.ravel()
    tol = 2e-14
    for v0 in values0[abs(values0 - 1/a) <= best + tol]:
        # Necessary conditions at endpoints and midpoint; never discard a
        # potentially better polynomial (apart from float64 tolerance).
        lower = np.maximum(abs(v0 - 1/a), abs(v0 + c1*H + c2*H*H - 1/(a+H)))
        lower = np.maximum(lower, abs(v0 + c1*(H/2) + c2*(H/2)**2 - 1/(a+H/2)))
        keep = lower <= best + tol
        u, v = c1[keep], c2[keep]
        if not len(u):
            continue
        errors = extrema_error(a, v0, u, v)
        j = int(np.argmin(errors))
        if errors[j] < best:
            best = float(errors[j])
            result = (float(v0), float(u[j]), float(v[j]))
    return result, best


def solve_large(a, values0, values):
    """Warm start, endpoint lower bound, then bounded-memory full search.

    Sample-point inequalities bound C2 for each C1. They are necessary,
    not sufficient: survivors are checked at every continuous extremum.
    """
    tol = 2e-14
    v0 = float(values0[np.argmin(abs(values0 - 1/a))])
    bound = abs(v0 - 1/a)
    delta = v0 - 1/a

    def neighbors(target):
        j = int(np.searchsorted(values, target))
        return values[max(0, j - 3):min(len(values), j + 4)]
    u = np.unique(np.concatenate([neighbors(-1/a**2 - k*delta/H) for k in [0, 1, 2, 3]]))
    v = np.unique(np.concatenate([neighbors(1/a**3 + k*delta/H**2) for k in [0, 1, 2]]))
    u, v = np.meshgrid(u, v, indexing='ij')
    errors = extrema_error(a, v0, u.ravel(), v.ravel())
    j = int(np.argmin(errors))
    best = float(errors[j])
    result = (v0, float(u.ravel()[j]), float(v.ravel()[j]))
    if best <= bound + tol:
        return result, best
    for c0 in values0[abs(values0 - 1/a) <= best + tol]:
        # Work in batches so even d=11 never builds a 63487 squared array.
        for start in range(0, len(values), 256):
            c1 = values[start:start + 256]
            low = np.full(len(c1), values[0])
            high = np.full(len(c1), values[-1])
            for x in [H/4, H/2, 3*H/4, H]:
                center = (1/(a+x) - c0 - c1*x)/x**2
                low = np.maximum(low, center - (best + tol)/x**2)
                high = np.minimum(high, center + (best + tol)/x**2)
            left = np.searchsorted(values, low, side='left')
            right = np.searchsorted(values, high, side='right')
            counts = np.maximum(right - left, 0)
            if not np.any(counts):
                continue
            u = np.repeat(c1, counts)
            indices = np.concatenate([np.arange(l, r) for l, r in zip(left, right) if r > l])
            v = values[indices]
            errors = extrema_error(a, c0, u, v)
            j = int(np.argmin(errors))
            if errors[j] < best:
                best = float(errors[j])
                result = (float(c0), float(u[j]), float(v[j]))
            if best <= bound + tol:
                return result, best
    return result, best


def main():
    OUT.mkdir(exist_ok=True)
    values0 = allowed(11)  # C0: full FP16 set
    for d in range(1, 12):
        values = allowed(d)  # C1/C2: d-constrained
        rows = []
        for i in range(128):
            coeff, error = solve_segment(1 + i/128, values0, values)
            rows.append((i, *coeff, error))
        coeff = np.array([r[1:4] for r in rows])
        bits = coeff.astype(np.float16).view(np.uint16)
        assert np.all((bits[:, 1:] & ((1 << (11 - d)) - 1)) == 0)
        assert np.all((bits[:, 0] & 0x7c00) != 0x7c00)
        assert np.array_equal(coeff, coeff.astype(np.float16).astype(np.float64))
        with (OUT / f'coefficients_d{d}.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['segment', 'C0', 'C1', 'C2', 'continuous_max_abs_error',
                        'C0_fp16_hex', 'C1_fp16_hex', 'C2_fp16_hex'])
            for row, b in zip(rows, bits):
                w.writerow([*row, *(f'{int(v):04x}' for v in b)])
        print(f"Generated d={d}: 128 segments", flush=True)


if __name__ == "__main__":
    main()
