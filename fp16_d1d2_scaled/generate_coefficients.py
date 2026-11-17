"""d1/d2 version with range-scaled C1/C2 integers.

Same model as ../fp16_d1d2 (2**d1 segments, polynomial input t = the low
10-d1 FP16 fraction bits, C0 any FP16 value), but C1/C2 are stored as d2-bit
unsigned integers m mapped onto their ACTUAL value ranges instead of a bare
2**-d2 grid:

    C1 = lo1 + m1 * step1,  m1 in [0, 2**d2 - 1]   (C1 range is negative)
    C2 = lo2 + m2 * step2,  m2 in [0, 2**d2 - 1]   (C2 range is positive)

Ranges come from the Taylor expansion of 1/x around the segment start X0
with delta = t * 2**-10 (t integer):

    f'(x)  = -1/x^2  in [-2^0, -2^-2]   -> C1 = f' * 2^-10    in [-2^-10, -2^-12]
    f''(x) =  2/x^3, Taylor quadratic coefficient f''/2 = 1/x^3 in [2^-3, 2^0]
                                          -> C2 = (f''/2) * 2^-20 in [2^-23, 2^-20]

Each range is widened by 1/16 of its span on both sides (boundary safety),
then covered by exactly 2**d2 levels: step = span_widened / (2**d2 - 1).
Sign is carried by the offset (C1 always negative, C2 always positive).

Search is the same complete enumeration as ../fp16_d1d2 (warm start, then
all (m1, m2) that can beat it via necessary conditions), just over the
scaled integer grids.

Run all sweep configs:            python generate_coefficients.py
Run one config:                   python generate_coefficients.py 5 8
"""
from pathlib import Path
import csv
import sys
import numpy as np

OUT = Path(__file__).resolve().parent / 'results'
U = 1 / 1024.0

RANGE1 = (-2.0**-10, -2.0**-12)   # C1 actual range (negative)
RANGE2 = (2.0**-23, 2.0**-20)     # C2 actual range (positive)

SWEEP = ([(3, d2) for d2 in (2, 3, 4, 5, 6, 8, 10, 12)]
         + [(5, d2) for d2 in (2, 3, 4, 5, 6, 8, 10, 12)]
         + [(7, d2) for d2 in (2, 3, 4, 5, 6, 8)]
         + [(10, 2)])


def make_grid(rng, d2, margin=1 / 16):
    lo, hi = rng
    span = hi - lo
    lo_e = lo - span * margin
    hi_e = hi + span * margin
    step = (hi_e - lo_e) / (2**d2 - 1)
    return lo_e, step


def fp16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def pair_error(c1s, c2, t, f):
    """Exact discrete error of each C1 with fixed C2 and best FP16 C0."""
    r = c1s[:, None] * t[None, :] + (c2 * t * t - f)[None, :]
    rmin = r.min(axis=1)
    rmax = r.max(axis=1)
    half = (rmax - rmin) / 2
    m = (rmax + rmin) / 2
    c0 = fp16(-m)
    return half + np.abs(c0 + m), c0


def interpolation_point(f):
    t1, t2 = 1.0, float(len(f) - 1)
    det = t1 * t2 * (t2 - t1)
    d1f = f[1] - f[0]
    d2f = f[-1] - f[0]
    return (d1f * t2**2 - d2f * t1**2) / det, (t1 * d2f - t2 * d1f) / det


def to_m(value, lo, step, levels):
    return int(np.clip(round((value - lo) / step), 0, levels - 1))


def solve_segment(a, n, d2):
    lo1, s1 = make_grid(RANGE1, d2)
    lo2, s2 = make_grid(RANGE2, d2)
    levels = 2**d2
    t = np.arange(n, dtype=np.float64)
    f = 1.0 / (a + t * U)
    if n == 1:
        c0 = float(fp16(f[0]))
        return abs(c0 - float(f[0])), c0, 0, 0
    c1_ip, c2_ip = interpolation_point(f)
    starts = [(to_m(c1_ip, lo1, s1, levels), to_m(c2_ip, lo2, s2, levels))]
    A = np.polyfit(t, f, 2)
    starts.append((to_m(float(A[1]), lo1, s1, levels), to_m(float(A[0]), lo2, s2, levels)))
    best = None
    for m1, m2 in starts:
        err, c0 = pair_error(np.array([lo1 + m1 * s1]), lo2 + m2 * s2, t, f)
        cand = (float(err[0]), float(c0[0]), m1, m2)
        if best is None or cand[0] < best[0]:
            best = cand
    E = best[0]
    # Necessary conditions: |C1*(t2-t1) + C2*(t2^2-t1^2) - df| <= 2E.
    # The (1, T-1) pair bounds C2 around the interpolation point.
    t1, t2 = 1.0, float(n - 1)
    det = t1 * t2 * (t2 - t1)
    b2 = 2 * E * (t1 + t2) / det
    eps = 1e-9
    m2a = max(0, int(np.ceil((c2_ip - b2 - lo2) / s2 - eps)))
    m2b = min(levels - 1, int(np.floor((c2_ip + b2 - lo2) / s2 + eps)))
    tt = t[1:]
    df = f[1:] - f[0]
    for m2 in range(m2a, m2b + 1):
        c2 = lo2 + m2 * s2
        center = (df - c2 * tt * tt) / tt
        lo = float(np.max(center - 2 * E / tt))
        hi = float(np.min(center + 2 * E / tt))
        m1a = max(0, int(np.ceil((lo - lo1) / s1 - eps)))
        m1b = min(levels - 1, int(np.floor((hi - lo1) / s1 + eps)))
        if m1a > m1b:
            continue
        c1s = lo1 + np.arange(m1a, m1b + 1, dtype=np.float64) * s1
        err, c0 = pair_error(c1s, c2, t, f)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(c0[j]), m1a + j, m2)
    return best


def run_config(d1, d2):
    OUT.mkdir(exist_ok=True)
    lo1, s1 = make_grid(RANGE1, d2)
    lo2, s2 = make_grid(RANGE2, d2)
    rows = []
    for i in range(2**d1):
        a = 1 + i / 2**d1
        n = 2**(10 - d1)
        err, c0, m1, m2 = solve_segment(a, n, d2)
        rows.append((i, c0, m1, m2, lo1 + m1 * s1, lo2 + m2 * s2, err))
    coeff = np.array([r[1:6] for r in rows])
    bits = coeff[:, 0].astype(np.float16).view(np.uint16)
    assert np.all((bits & 0x7c00) != 0x7c00), 'C0 must be finite FP16'
    assert np.all(coeff[:, 3] < 0) and np.all(coeff[:, 4] > 0), 'C1 negative, C2 positive'
    path = OUT / f'coefficients_d1_{d1}_d2_{d2}.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['segment', 'C0', 'm1', 'm2', 'C1', 'C2', 'error', 'C0_fp16_hex'])
        for (i, c0, m1, m2, c1, c2, err), b in zip(rows, bits):
            w.writerow([i, c0, m1, m2, c1, c2, err, f'{int(b):04x}'])
    print(f'Generated d1={d1} d2={d2}: {2**d1} segments '
          f'(step1={s1:.3e}, step2={s2:.3e})', flush=True)


def main():
    if len(sys.argv) == 3:
        configs = [(int(sys.argv[1]), int(sys.argv[2]))]
    else:
        configs = SWEEP
    for d1, d2 in configs:
        if 2**(10 - d1) == 2:
            raise SystemExit('d1=9 (2 points per segment) is not supported')
        run_config(d1, d2)


if __name__ == '__main__':
    main()
