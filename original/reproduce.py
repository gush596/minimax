"""Reciprocal-only numerical reproduction of Pineiro et al., TC 2005.

Three-pass coefficients follow section 3.4.1; mpmath replaces Maple.
The truncated-square model is NOT the paper's partial-product squarer.
Run: python reproduce.py
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
import mpmath as mp

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
M, T, P, Q = 7, 26, 16, 10
mp.mp.dps = 70


def quantize(x, bits):
    """Round to nearest, ties to even, at a specified binary fractional bit."""
    z = x * 2**bits
    lo = mp.floor(z)
    f = z - lo
    return int(lo + (f > mp.mpf('.5') or (f == mp.mpf('.5') and int(lo) % 2)))


def initial_minimax(a):
    """Remez exchange on u in [0,1], with high precision extrema.

    Reciprocal has exactly two interior error extrema for its quadratic
    minimax approximation. Solve alternation, then relocate these extrema.
    """
    h = mp.mpf(1) / 2**M
    points = [mp.mpf(0), mp.mpf('.25'), mp.mpf('.75'), mp.mpf(1)]
    for _ in range(30):
        mat = mp.matrix([[1, u, u*u, (-1)**j] for j, u in enumerate(points)])
        b0, b1, b2, err = mp.lu_solve(mat, mp.matrix([1/(a+h*u) for u in points]))
        der = lambda u: -h/(a+h*u)**2 - b1 - 2*b2*u
        left = mp.findroot(der, (mp.mpf('.1'), mp.mpf('.4')))
        right = mp.findroot(der, (mp.mpf('.6'), mp.mpf('.9')))
        assert 0 < left < right < 1
        movement = max(abs(left-points[1]), abs(right-points[2]))
        points = [mp.mpf(0), left, right, mp.mpf(1)]
        if movement < mp.mpf('1e-55'):
            break
    else:
        raise RuntimeError('Remez did not converge')
    return b0, b1/h, b2/h**2


def residual_extrema(a, c1, c2):
    """Find every stationary point of 1/(a+x)-c1*x-c2*x*x.

    Its derivative is concave, so split at its sole possible turning point;
    on each resulting interval a sign change contains exactly one root.
    """
    h = mp.mpf(1)/2**M
    d = lambda x: -1/(a+x)**2-c1-2*c2*x
    cuts = [mp.mpf(0), h]
    turn = c2**(-mp.mpf(1)/3)-a
    if 0 < turn < h:
        cuts.insert(1, turn)
    candidates = [mp.mpf(0), h]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        if d(lo) == 0:
            candidates.append(lo)
        if d(lo)*d(hi) < 0:
            for _ in range(210):
                mid = (lo+hi)/2
                if d(lo)*d(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            candidates.append((lo+hi)/2)
    return [1/(a+x)-c1*x-c2*x*x for x in candidates]


def generate():
    rows = []
    for i in range(2**M):
        a = 1+mp.mpf(i)/2**M
        _, a1, a2 = initial_minimax(a)
        i1 = quantize(a1, P)
        c1 = mp.mpf(i1)/2**P
        # Paper equation (10): analytical second pass.
        i2 = quantize(a2+(a1-c1)*2**M, Q)
        c2 = mp.mpf(i2)/2**Q
        vals = residual_extrema(a, c1, c2)
        # Best constant approximation is the midpoint of min and max.
        i0 = quantize((min(vals)+max(vals))/2, T)
        rows.append((i0, i1, i2))
    return np.array(rows, dtype=np.int64)


def accum(coeff, segment, mode):
    """Exact fixed-point accumulator, scale 2^-56, no product rounding."""
    t = np.arange(2**16, dtype=np.int64)
    c0, c1, c2 = map(int, coeff[segment])
    base = (c0 << 30) + ((c1*t) << 17)
    square = t*t
    if mode == 'exact_square':
        return base + c2*square
    if mode == 'truncated_square_q28':
        return base + ((c2*(square >> 18)) << 18)
    raise ValueError(mode)


def evaluate(coeff, mode, bias):
    """Exhaustive comparison to exact rational reciprocal of every FP32 X.

    X = mantissa/2^23; output Y/2^24. Signed error in output ulps is
    (Y*mantissa - 2^47)/mantissa. This numerator is exact int64.
    """
    worst = (-1, None)
    failures = faithful_fail = correct_diff = 0
    count = 0
    absolute_sum_ulp = 0.
    min_signed, max_signed = math.inf, -math.inf
    for i in range(128):
        mant = 2**23 + i*2**16 + np.arange(2**16, dtype=np.int64)
        raw = accum(coeff, i, mode) + bias
        # Round accumulator Q56 to Q24, ties to even.
        y = raw >> 32
        rem = raw & (2**32-1)
        y = y + ((rem > 2**31) | ((rem == 2**31) & ((y & 1) != 0)))
        numerator = y*mant - 2**47
        signed = numerator.astype(np.float64)/mant
        absolute_sum_ulp += float(np.sum(np.abs(signed), dtype=np.float64))
        j = int(np.argmax(np.abs(signed)))
        if abs(signed[j]) > worst[0]:
            worst = (float(abs(signed[j])), {
                'input_bits_hex': f'{0x3f800000+i*65536+j:08x}',
                'x': float(mant[j]/2**23), 'output': float(y[j]/2**24),
                'signed_error_ulp': float(signed[j]),
                'exact_error_ulp_fraction': f'{int(numerator[j])}/{int(mant[j])}'})
        failures += int(np.count_nonzero(np.abs(numerator) >= mant))
        floor = 2**47 // mant
        residue = 2**47 % mant
        ceil = floor + (residue != 0)
        faithful_fail += int(np.count_nonzero((y != floor) & (y != ceil)))
        nearest = floor + ((2*residue > mant) | ((2*residue == mant) & ((floor & 1) != 0)))
        correct_diff += int(np.count_nonzero(y != nearest))
        min_signed = min(min_signed, float(signed.min()))
        max_signed = max(max_signed, float(signed.max()))
        count += len(mant)
    return dict(mode=mode, bias_q56=int(bias), bias_real=bias/2**56,
                tested_inputs=count, max_absolute_error_ulp=worst[0],
                max_absolute_error=worst[0]/2**24,
                mean_absolute_error=absolute_sum_ulp/count/2**24,
                min_signed_error_ulp=min_signed, max_signed_error_ulp=max_signed,
                violations_strict_1ulp=failures, faithful_rounding_violations=faithful_fail,
                differs_from_correct_rounding=correct_diff, worst_case=worst[1])


def choose_bias(coeff, mode):
    """Our reproducible centering heuristic, not the author's original bias."""
    low, high = math.inf, -math.inf
    for i in range(128):
        mant = 2**23 + i*2**16 + np.arange(2**16, dtype=np.int64)
        err = accum(coeff, i, mode).astype(np.float64)/2**56 - 2**23/mant
        low, high = min(low, float(err.min())), max(high, float(err.max()))
    return round(-(low+high)*2**55)


def pack_rom(coeff):
    """Own documented ROM format: 25 C0 bits, 16 C1 bits, 10 C2 bits.

    C0 has implicit bit 2^-1, C1 has implicit negative sign extension.
    The packed table excludes the independently stored rounding bias.
    """
    packed = 0
    for c0, c1, c2 in coeff:
        c0, c1, c2 = int(c0), int(c1), int(c2)
        assert 2**25 <= c0 < 2**26 and -2**16 <= c1 < 0 and 0 <= c2 < 2**10
        word = ((c0-2**25) << 26) | ((c1 & 0xffff) << 10) | c2
        packed = (packed << 51) | word
    return packed.to_bytes(816, 'big')


def unpack_rom(data):
    if len(data) != 816:
        raise ValueError('Expected exactly 816 bytes')
    bits = int.from_bytes(data, 'big')
    rows = []
    for i in range(128):
        word = (bits >> (51*(127-i))) & (2**51-1)
        rows.append(((word >> 26)+2**25, ((word >> 10) & 0xffff)-2**16, word & 1023))
    return np.array(rows, dtype=np.int64)


def reciprocal_bits(input_bits, coeff, bias, mode='exact_square'):
    """Scalar integer reference; accepts ONLY binary32 encodings in [1,2)."""
    if not 0x3f800000 <= input_bits < 0x40000000:
        raise ValueError('Only positive normalized binary32 inputs in [1,2) are supported')
    frac = input_bits-0x3f800000
    segment, t = divmod(frac, 2**16)
    c0, c1, c2 = map(int, coeff[segment])
    sq = t*t
    if mode == 'exact_square':
        term = c2*sq
    elif mode == 'truncated_square_q28':
        term = (c2*(sq >> 18)) << 18
    else:
        raise ValueError(mode)
    raw = (c0 << 30)+(c1*t << 17)+term+int(bias)
    y, rem = divmod(raw, 2**32)
    y += rem > 2**31 or (rem == 2**31 and y % 2 != 0)
    if not 2**23 <= y <= 2**24:
        raise ArithmeticError('Output outside the supported [0.5,1] range')
    return 0x3f000000 + y-2**23


def main():
    OUT.mkdir(exist_ok=True)
    coeff = generate()
    with (OUT/'coefficients.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['index','left','right','C0_q26','C1_q16','C2_q10','C0','C1','C2'])
        for i, (c0,c1,c2) in enumerate(coeff):
            w.writerow([i,1+i/128,1+(i+1)/128,int(c0),int(c1),int(c2),c0/2**26,c1/2**16,c2/2**10])
    # Full signed 32-bit words, intentionally not claiming the author's packed ROM.
    for j, name in enumerate(['c0_q26','c1_q16','c2_q10']):
        (OUT/(name+'.mem')).write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in coeff[:,j]), encoding='ascii')
    rom = pack_rom(coeff)
    assert np.array_equal(coeff, unpack_rom(rom))
    (OUT/'coefficients_packed.bin').write_bytes(rom)
    reports = []
    for mode in ['exact_square','truncated_square_q28']:
        for bias in [0, choose_bias(coeff, mode)]:
            reports.append(evaluate(coeff, mode, bias))
    payload = dict(scope='positive normalized binary32 significands [1,2)',
                   paper_config=dict(m=M,C0_fraction=T,C1_fraction=P,C2_fraction=Q),
                   original_author_table=False, original_partial_product_squarer=False,
                   models=reports)
    (OUT/'validation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    lines = ['# 本次运行的穷举结果', '',
             '范围：[1,2) 全部 8,388,608 个 binary32 输入。偏置由本实现生成。', '',
             '| 模型 | 偏置 Q56 整数 | 最大误差 / ulp | ≥1 ulp 数量 | 非忠实舍入数量 |',
             '|---|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['mode']} | {r['bias_q56']} | {r['max_absolute_error_ulp']:.10f} | {r['violations_strict_1ulp']} | {r['faithful_rounding_violations']} |")
    lines += ['', '平均误差采用 MAE，即全部输入绝对误差的算术平均。以下包含最终 Q24 舍入。', '',
              '| 模型 | 偏置 Q56 整数 | 最大绝对误差 | 平均绝对误差 MAE |',
              '|---|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['mode']} | {r['bias_q56']} | {r['max_absolute_error']:.10g} | {r['mean_absolute_error']:.10g} |")
    lines += ['', '注意：exact_square 是精确平方/累加参考模型；truncated_square_q28 仅在完整平方后截断。',
              '两者均不是论文专用部分积平方器的逐位复现；未验证负数、零、特殊值和范围恢复。',
              '1 ulp 不等于正确舍入；与正确舍入不同的输入数量详见 validation.json。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    main()
