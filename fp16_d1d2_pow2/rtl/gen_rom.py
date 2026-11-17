#!/usr/bin/env python3
"""Generate the ROM init file for the fp16 reciprocal datapath from a CSV.

Usage: python3 gen_rom.py [d1 d2]     (default 3 7)

Reads ../results/coefficients_d1_<d1>_d2_<d2>.csv and writes rom_d1_<d1>_d2_<d2>.memh.
Each entry packs:  { m2[D2-1:0], m1[D2-1:0], m0[11:0] }
with m0 = C0 * 2^11 (12-bit integer 1024..2048; C0 is FP16 in [0.5, 1],
exponent field always -1), m1 = -C1 * 2^(d2+10), m2 = C2 * 2^(d2+20)
(unsigned integers in [0, 2^d2-1]).
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    d1, d2 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (3, 7)
    src = ROOT.parent / 'results' / f'coefficients_d1_{d1}_d2_{d2}.csv'
    rows = list(csv.DictReader(src.open()))
    assert len(rows) == 2**d1, f'expected {2**d1} segments'
    out = ROOT / f'rom_d1_{d1}_d2_{d2}.memh'
    lines = []
    for r in rows:
        m0 = round(float(r['C0']) * 2**11)
        m1, m2 = int(r['m1']), int(r['m2'])
        assert 1024 <= m0 <= 2048 and 0 <= m1 < 2**d2 and 0 <= m2 < 2**d2
        entry = m0 | (m1 << 12) | (m2 << (12 + d2))
        lines.append(f'{entry:0{((12 + 2 * d2) + 3) // 4}x}')
    out.write_text('\n'.join(lines) + '\n')
    print(f'wrote {out.name}: {len(lines)} entries, {12 + 2 * d2} bits each')


if __name__ == '__main__':
    main()
