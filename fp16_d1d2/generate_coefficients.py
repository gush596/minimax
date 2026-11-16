"""d1/d2 version: 2^d1 segments, FP16-grid input, C0 FP16, C1/C2 d2-bit fractions.

- Segments: 2**d1 over [1,2) (d1 default 5 -> 32 segments).
- The polynomial input is t = floor((x - segment_start) * 1024): the low
  (10 - d1) fraction bits of the FP16 input x, i.e. the integer count of
  FP16 ULPs inside the segment. P(t) = C0 + C1*t + C2*t**2.
- C0: any finite FP16 value (no d mask).
- C1, C2: multiples of 2**-d2 (binary fraction of at most d2 bits), any sign.

Search per segment: warm start (least-squares fit and 2-point interpolation,
rounded to the grid), then enumerate every (C1, C2) grid pair that can still
beat it. Necessary conditions (residual differences <= 2E) bound the C2 box
and prune C1 per C2; final errors use the exact discrete extrema over the
segment's FP16 grid points. Grid completeness is guaranteed relative to the
warm-start error.

Run all sweep configs:            python generate_coefficients.py
Run one config:                   python generate_coefficients.py 5 16
"""
from pathlib import Path
import csv
import sys
import numpy as np

OUT = Path(__file__).resolve().parent / 'results'
U = 1 / 1024.0  # one input FP16 ULP in [1,2)

SWEEP = [(d1, d2) for d1 in (3, 5, 7, 10) for d2 in (8, 12, 16, 20, 24)] + [(3, 28)]


def fp16(x):
    return np.asarray(x, dtype=np.float64).astype(np.float16).astype(np.float64)


def round_grid(v, step):
    return round(v / step) * step


def pair_error(c1s, c2, t, f):
    """Exact discrete error of each C1 with fixed C2 and best FP16 C0.

    Error = half-range of the residual + distance of its midpoint to the
    nearest FP16 value (the min-max optimal C0 is the FP16 value closest to
    the midpoint of the residual range).
    """
    r = c1s[:, None] * t[None, :] + (c2 * t * t - f)[None, :]
    rmin = r.min(axis=1)
    rmax = r.max(axis=1)
    half = (rmax - rmin) / 2
    m = (rmax + rmin) / 2
    # error = C0 + r, so the min-max C0 is the FP16 value nearest to -m.
    c0 = fp16(-m)
    return half + np.abs(c0 + m), c0


def interpolation_point(f):
    """Exact (C1, C2) through residual differences at t=1 and t=T-1."""
    t1, t2 = 1.0, float(len(f) - 1)
    det = t1 * t2 * (t2 - t1)
    d1f = f[1] - f[0]
    d2f = f[-1] - f[0]
    return (d1f * t2**2 - d2f * t1**2) / det, (t1 * d2f - t2 * d1f) / det


def solve_segment(a, n, d2):
    step = 2.0 ** -d2
    t = np.arange(n, dtype=np.float64)
    f = 1.0 / (a + t * U)
    if n == 1:
        c0 = float(fp16(f[0]))
        return abs(c0 - float(f[0])), c0, 0.0, 0.0
    c1_ip, c2_ip = interpolation_point(f)
    starts = [(round_grid(c1_ip, step), round_grid(c2_ip, step))]
    A = np.polyfit(t, f, 2)
    starts.append((round_grid(float(A[1]), step), round_grid(float(A[0]), step)))
    best = None
    for c1, c2 in starts:
        err, c0 = pair_error(np.array([c1]), c2, t, f)
        cand = (float(err[0]), float(c0[0]), float(c1), float(c2))
        if best is None or cand[0] < best[0]:
            best = cand
    E = best[0]
    # Necessary conditions: |C1*(t2-t1) + C2*(t2^2-t1^2) - df| <= 2E for any
    # pair. Using (1, T-1) bounds C2 around the interpolation point.
    t1, t2 = 1.0, float(n - 1)
    det = t1 * t2 * (t2 - t1)
    b2 = 2 * E * (t1 + t2) / det
    m2a = int(np.ceil((c2_ip - b2 - step) / step))
    m2b = int(np.floor((c2_ip + b2 + step) / step))
    tt = t[1:]
    df = f[1:] - f[0]
    for m2 in range(m2a, m2b + 1):
        c2 = m2 * step
        # For this C2, every t gives a strip for C1; intersect them.
        center = (df - c2 * tt * tt) / tt
        lo = float(np.max(center - 2 * E / tt))
        hi = float(np.min(center + 2 * E / tt))
        m1a = int(np.ceil((lo - 0.5 * step) / step))
        m1b = int(np.floor((hi + 0.5 * step) / step))
        if m1a > m1b:
            continue
        c1s = np.arange(m1a, m1b + 1, dtype=np.float64) * step
        err, c0 = pair_error(c1s, c2, t, f)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(c0[j]), float(c1s[j]), c2)
    return best


def run_config(d1, d2):
    OUT.mkdir(exist_ok=True)
    step = 2.0 ** -d2
    rows = []
    for i in range(2**d1):
        a = 1 + i / 2**d1
        n = 2**(10 - d1)
        err, c0, c1, c2 = solve_segment(a, n, d2)
        rows.append((i, c0, c1, c2, err))
    coeff = np.array([r[1:4] for r in rows])
    bits = coeff[:, 0].astype(np.float16).view(np.uint16)
    assert np.all((bits & 0x7c00) != 0x7c00), 'C0 must be finite FP16'
    for col in (1, 2):
        m = coeff[:, col] / step
        assert np.all(np.abs(m - np.round(m)) < 1e-6), 'C1/C2 must be multiples of 2^-d2'
    path = OUT / f'coefficients_d1_{d1}_d2_{d2}.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['segment', 'C0', 'C1', 'C2', 'error', 'C0_fp16_hex',
                    'C1_num', 'C2_num'])
        for (i, c0, c1, c2, err), b in zip(rows, bits):
            w.writerow([i, c0, c1, c2, err, f'{int(b):04x}',
                        int(round(c1 / step)), int(round(c2 / step))])
    print(f'Generated d1={d1} d2={d2}: {2**d1} segments', flush=True)


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
