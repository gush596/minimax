"""Read generated CSV coefficients and evaluate every FP16 input in [1,2)."""
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
U = 1 / 1024.0
AFFINE = ROOT.parent / 'fp16_d1d2_scaled' / 'results' / 'validation.json'


def evaluate(d1, d2, rows):
    worst_abs = worst_rel = 0.
    worst = None
    absolute_sum = squared = 0.
    mismatches = 0
    count = 0
    for i in range(2**d1):
        a = 1 + i / 2**d1
        n = 2**(10 - d1)
        t = np.arange(n, dtype=np.float64)
        x = a + t * U
        ref = 1.0 / x
        c0, c1, c2 = rows[i][0], rows[i][1], rows[i][2]
        y = c0 + c1 * t + c2 * t * t
        err = y - ref
        j = int(np.argmax(np.abs(err)))
        if abs(err[j]) > worst_abs:
            worst_abs = float(abs(err[j]))
            worst = dict(x=float(x[j]), output=float(y[j]), reference=float(ref[j]),
                         x_fp16_hex=f'{int(np.float16(x[j]).view(np.uint16)):04x}',
                         segment=i, t=int(j))
        worst_rel = max(worst_rel, float(np.max(np.abs(err) * x)))
        absolute_sum += float(np.sum(np.abs(err)))
        squared += float(np.dot(err, err))
        mismatches += int(np.count_nonzero(np.float16(y) != np.float16(ref)))
        count += len(t)
    return dict(d1=d1, d2=d2, segments=2**d1, tested_inputs=count,
                max_absolute_error=worst_abs,
                mean_absolute_error=absolute_sum / count,
                max_relative_error=worst_rel,
                max_error_fp16_output_ulp=worst_abs / 2**-11,
                rms_absolute_error=float(np.sqrt(squared / count)),
                fp16_correct_rounding_mismatches=mismatches,
                worst_case=worst)


def main():
    files = sorted(OUT.glob('coefficients_d1_*_d2_*.csv'),
                   key=lambda p: tuple(map(int, re.findall(r'd1_(\d+)_d2_(\d+)', p.stem)[0])))
    if not files:
        raise SystemExit('No coefficient files; run generate_coefficients.py first.')
    affine = {}
    if AFFINE.exists():
        affine = {(r['d1'], r['d2']): r for r in json.loads(AFFINE.read_text())}
    reports = []
    for path in files:
        d1, d2 = map(int, re.findall(r'd1_(\d+)_d2_(\d+)', path.stem)[0])
        data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        if not np.array_equal(data['segment'], np.arange(2**d1)):
            raise ValueError(f'{path.name}: expected {2**d1} segments')
        rows = np.column_stack([data[n] for n in ['C0', 'C1', 'C2']])
        stored = rows[:, 0].astype(np.float16)
        if not np.array_equal(stored.astype(np.float64), rows[:, 0]):
            raise ValueError(f'{path.name}: C0 must be FP16-representable')
        s1, s2 = 2.0**-(d2 + 10), 2.0**-(d2 + 20)
        m1, m2 = data['m1'].astype(np.int64), data['m2'].astype(np.int64)
        if np.any((m1 < 0) | (m1 > 2**d2 - 1)) or np.any((m2 < 0) | (m2 > 2**d2 - 1)):
            raise ValueError(f'{path.name}: m out of [0, 2^d2-1]')
        if not np.allclose(rows[:, 1], -m1 * s1, rtol=0, atol=s1 * 1e-6) or \
           not np.allclose(rows[:, 2], m2 * s2, rtol=0, atol=s2 * 1e-6):
            raise ValueError(f'{path.name}: C1/C2 violate the power-of-2 scaling')
        r = evaluate(d1, d2, rows)
        r['step1'], r['step2'] = s1, s2
        r['affine_scheme_max_error'] = affine.get((d1, d2), {}).get('max_absolute_error')
        reports.append(r)
        print(json.dumps(r), flush=True)
    (OUT / 'validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')

    lines = ['# d1/d2 纯 2^n 缩放版：C1/C2 范围最大值对齐 2^d2，无符号 d2 位整数', '',
             '模型同 fp16_d1d2：2^d1 段，t=段内 FP16 ULP 计数，P(t)=C0+C1*t+C2*t²，C0 为 FP16。',
             'C1/C2 表示简化为纯移位缩放：范围最大值缩放到 2^d2，整数 m ∈ [0, 2^d2-1]，',
             'C1 = -m1·2^-(d2+10)（恒负），C2 = +m2·2^-(d2+20)（恒正）。缩放因子是 2^n，',
             '无偏移、无非 2 幂乘法；符号由系数承担。搜索域为完整 [0, 2^d2-1]，',
             '搜索同 fp16_d1d2（必要条件剪枝完备）。评估穷举 1024 个 FP16 输入。', '',
             '| d1 | d2 | step1 | step2 | 最大绝对误差 | MAE | 最大相对误差 | 最大误差(输出ulp) | FP16正确舍入不符 | 仿射版同d2最大误差 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        a = r['affine_scheme_max_error']
        a = f'{a:.10g}' if a else '-'
        lines.append(f"| {r['d1']} | {r['d2']} | {r['step1']:.3e} | {r['step2']:.3e} "
                     f"| {r['max_absolute_error']:.10g} | {r['mean_absolute_error']:.10g} "
                     f"| {r['max_relative_error']:.6%} | {r['max_error_fp16_output_ulp']:.6f} "
                     f"| {r['fp16_correct_rounding_mismatches']} | {a} |")
    lines += ['', '仿射版列为 fp16_d1d2_scaled（C=lo+m·step 偏移缩放）同参数的最大绝对误差。',
              '运行：先 `python generate_coefficients.py`（可带 d1 d2 只跑一组），再 `python evaluate_accuracy.py`。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
