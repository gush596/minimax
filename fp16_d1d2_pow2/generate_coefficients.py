"""d1/d2 version with pure power-of-2 scaling for C1/C2.

Simplification of ../fp16_d1d2_scaled/: instead of the affine map
C = lo + m*step (offset + non-power-of-2 step), each coefficient range is
scaled by 2**n (n integer) so its maximum becomes 2**d2, and the stored
integer m is an unsigned d2-bit value in [0, 2**d2 - 1]:

    C1 = -(m1 * 2**-(d2+10)),  m1 in [0, 2**d2 - 1]   (C1 always <= 0)
    C2 = +(m2 * 2**-(d2+20)),  m2 in [0, 2**d2 - 1]   (C2 always >= 0)

Scaling exponents come from the Taylor ranges (see ../fp16_d1d2_scaled):
|C1|max = 2**-10, C2max = 2**-20. Scaling by 2**n maps max to 2**d2, so
n1 = d2+10, n2 = d2+20. The hardware scaling is a pure bit shift; the sign
is carried by the coefficient itself (C1 negative, C2 positive).

Model is unchanged: 2**d1 segments, polynomial input t = low (10-d1) FP16
fraction bits (segment ULP count), P(t) = C0 + C1*t + C2*t**2, C0 FP16.

Search: same complete enumeration as ../fp16_d1d2_scaled (warm start, then
all (m1, m2) that can beat it via necessary conditions), over the full
integer domain [0, 2**d2 - 1].

Run all sweep configs:            python generate_coefficients.py
Run one config:                   python generate_coefficients.py 5 6
"""
from pathlib import Path
import csv
import sys
import numpy as np

OUT = Path(__file__).resolve().parent / 'results'
U = 1 / 1024.0

SWEEP = ([(3, d2) for d2 in (2, 3, 4, 5, 6, 8, 10)]
         + [(5, d2) for d2 in (2, 3, 4, 5, 6, 8, 10)]
         + [(7, d2) for d2 in (2, 3, 4, 5, 6, 8)]
         + [(10, 2)])


def steps(d2):
    return 2.0**-(d2 + 10), 2.0**-(d2 + 20)


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


def to_m1(c1_value, s1, levels):
    return int(np.clip(round(-c1_value / s1), 0, levels - 1))


def to_m2(c2_value, s2, levels):
    return int(np.clip(round(c2_value / s2), 0, levels - 1))


def solve_segment(a, n, d2):
    s1, s2 = steps(d2)
    levels = 2**d2
    t = np.arange(n, dtype=np.float64)
    f = 1.0 / (a + t * U)
    if n == 1:
        c0 = float(fp16(f[0]))
        return abs(c0 - float(f[0])), c0, 0, 0
    c1_ip, c2_ip = interpolation_point(f)
    starts = [(to_m1(c1_ip, s1, levels), to_m2(c2_ip, s2, levels))]
    A = np.polyfit(t, f, 2)
    starts.append((to_m1(float(A[1]), s1, levels), to_m2(float(A[0]), s2, levels)))
    best = None
    for m1, m2 in starts:
        err, c0 = pair_error(np.array([-m1 * s1]), m2 * s2, t, f)
        cand = (float(err[0]), float(c0[0]), m1, m2)
        if best is None or cand[0] < best[0]:
            best = cand
    E = best[0]
    # Necessary conditions |C1*(t2-t1) + C2*(t2^2-t1^2) - df| <= 2E bound
    # C2 around the interpolation point ((1, T-1) pair).
    t1, t2 = 1.0, float(n - 1)
    det = t1 * t2 * (t2 - t1)
    b2 = 2 * E * (t1 + t2) / det
    eps = 1e-9
    m2a = max(0, int(np.ceil((c2_ip - b2) / s2 - eps)))
    m2b = min(levels - 1, int(np.floor((c2_ip + b2) / s2 + eps)))
    tt = t[1:]
    df = f[1:] - f[0]
    for m2 in range(m2a, m2b + 1):
        c2 = m2 * s2
        # For this C2 every t gives a strip for C1; intersect, then map to
        # m1 = -C1/s1 (C1 is non-positive).
        center = (df - c2 * tt * tt) / tt
        lo = float(np.max(center - 2 * E / tt))
        hi = float(np.min(center + 2 * E / tt))
        m1a = max(0, int(np.ceil(-hi / s1 - eps)))
        m1b = min(levels - 1, int(np.floor(-lo / s1 + eps)))
        if m1a > m1b:
            continue
        c1s = -(np.arange(m1a, m1b + 1, dtype=np.float64) * s1)
        err, c0 = pair_error(c1s, c2, t, f)
        j = int(np.argmin(err))
        if err[j] < best[0]:
            best = (float(err[j]), float(c0[j]), m1a + j, m2)
    return best


def run_config(d1, d2):
    OUT.mkdir(exist_ok=True)
    s1, s2 = steps(d2)
    rows = []
    for i in range(2**d1):
        a = 1 + i / 2**d1
        n = 2**(10 - d1)
        err, c0, m1, m2 = solve_segment(a, n, d2)
        rows.append((i, c0, m1, m2, -m1 * s1, m2 * s2, err))
    coeff = np.array([r[1:6] for r in rows])
    bits = coeff[:, 0].astype(np.float16).view(np.uint16)
    assert np.all((bits & 0x7c00) != 0x7c00), 'C0 must be finite FP16'
    assert np.all(coeff[:, 3] <= 0) and np.all(coeff[:, 4] >= 0), 'C1 <= 0, C2 >= 0'
    assert np.all((coeff[:, 1] >= 0) & (coeff[:, 1] <= 2**d2 - 1)), 'm1 in range'
    assert np.all((coeff[:, 2] >= 0) & (coeff[:, 2] <= 2**d2 - 1)), 'm2 in range'
    path = OUT / f'coefficients_d1_{d1}_d2_{d2}.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['segment', 'C0', 'm1', 'm2', 'C1', 'C2', 'error', 'C0_fp16_hex'])
        for (i, c0, m1, m2, c1, c2, err), b in zip(rows, bits):
            w.writerow([i, c0, m1, m2, c1, c2, err, f'{int(b):04x}'])
    print(f'Generated d1={d1} d2={d2}: {2**d1} segments '
          f'(step1=2^-{d2 + 10}={s1:.3e}, step2=2^-{d2 + 20}={s2:.3e})', flush=True)


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
