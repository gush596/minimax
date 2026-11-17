#!/usr/bin/env python3
"""Bit-accurate model of fp16_recip.v integer datapath + verification.

No Verilog simulator is required: this script executes exactly the same
integer operations as the RTL (same widths, same rounding) and checks,
for all 1024 FP16 inputs in [1,2):

  1. inner = m1*1024 - m2*t >= 0  (RTL unsigned-ness assumption)
  2. y_q11 = round((m0<<SHIFT) + t*inner) >> SHIFT stays in [1024, 2048]
  3. the integer result matches the float64 polynomial evaluation used in
     ../evaluate_accuracy.py after output rounding
  4. error vs 1/x: max abs error, FP16-ulp, correctly-rounded mismatches

Usage: python3 datapath_model.py [d1 d2]   (default 3 7)
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    d1, d2 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (3, 7)
    shift = d2 + 9
    rows = list(csv.DictReader((ROOT.parent / 'results' /
                                f'coefficients_d1_{d1}_d2_{d2}.csv').open()))
    rom = []
    for r in rows:
        m0 = round(float(r['C0']) * 2**11)
        rom.append((m0, int(r['m1']), int(r['m2'])))

    def f16(bits):  # pack fixed-point y*2^11 in [1024,2048] as FP16
        return 0x3c00 if bits >= 2048 else (0x3800 | (bits & 0x3ff))

    def to_f16(val):  # float -> FP16 bits (round to nearest even)
        import struct
        return struct.unpack('<H', struct.pack('<e', val))[0]

    worst = 0.0
    worst_frac = -1
    mismatches = 0
    inner_bad = 0
    range_bad = 0
    mismatch_vs_float64 = 0
    for frac in range(1024):
        seg = frac >> (10 - d1)
        t = frac & ((1 << (10 - d1)) - 1)
        m0, m1, m2 = rom[seg]
        p = m2 * t                      # stage 2 multiply
        inner = p - m1 * 1024           # subtract: m2*t - 1024*m1 (<= 0)
        if inner > 0:
            inner_bad += 1
        core = t * inner                # stage 3 multiply (<= 0)
        acc = (m0 << shift) + core      # add
        y_q11 = (acc + (1 << (shift - 1))) >> shift   # rounding shift
        if not (1024 <= y_q11 <= 2048):
            range_bad += 1
        # float64 polynomial reference (same as evaluate_accuracy.py),
        # rounded the same half-up way as the RTL rounding shift
        c0, c1, c2 = float(rows[seg]['C0']), float(rows[seg]['C1']), float(rows[seg]['C2'])
        y64 = c0 + c1 * t + c2 * t * t
        if y_q11 != int(y64 * 2048 + 0.5):
            mismatch_vs_float64 += 1
        # error vs exact reciprocal
        x = 1 + frac / 1024
        err = y_q11 / 2048 - 1 / x
        if abs(err) > worst:
            worst, worst_frac = abs(err), frac
        if f16(y_q11) != to_f16(1 / x):
            mismatches += 1

    print(f'config d1={d1} d2={d2}: 1024 inputs')
    print(f'  inner > 0 occurrences : {inner_bad}')
    print(f'  y_q11 out of range    : {range_bad}')
    print(f'  differs from float64 poly (same half-up rounding): {mismatch_vs_float64}')
    print(f'  max |y_hw - 1/x|      : {worst:.6e} = {worst / 2**-11:.4f} FP16 ulp '
          f'(frac={worst_frac})')
    print(f'  FP16 correct-rounding mismatches: {mismatches}/1024')
    ok = (inner_bad == 0 and range_bad == 0 and mismatch_vs_float64 == 0
          and worst <= 1.2 * 2**-11)
    print('DATAPATH MODEL: ' + ('PASS' if ok else 'FAIL'))


if __name__ == '__main__':
    main()
