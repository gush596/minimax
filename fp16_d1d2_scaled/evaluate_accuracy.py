"""Read generated CSV coefficients and evaluate every FP16 input in [1,2)."""
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
U = 1 / 1024.0
OLD = ROOT.parent / 'fp16_d1d2' / 'results' / 'validation.json'
RANGE1 = (-2.0**-10, -2.0**-12)
RANGE2 = (2.0**-23, 2.0**-20)


def make_grid(rng, d2, margin=1 / 16):
    lo, hi = rng
    span = hi - lo
    lo_e = lo - span * margin
    hi_e = hi + span * margin
    return lo_e, (hi_e - lo_e) / (2**d2 - 1)


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
    old = {}
    if OLD.exists():
        old = {(r['d1'], r['d2']): r for r in json.loads(OLD.read_text())}
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
        lo1, s1 = make_grid(RANGE1, d2)
        lo2, s2 = make_grid(RANGE2, d2)
        for col, lo, s in ((1, lo1, s1), (2, lo2, s2)):
            m = (rows[:, col] - lo) / s
            if not np.all(np.abs(m - np.round(m)) < 1e-6) or \
               not np.all((np.round(m) >= 0) & (np.round(m) <= 2**d2 - 1)):
                raise ValueError(f'{path.name}: coefficients violate the scaled d2={d2} grid')
        r = evaluate(d1, d2, rows)
        r['step1'], r['step2'] = s1, s2
        r['old_scheme_max_error'] = old.get((d1, d2), {}).get('max_absolute_error')
        reports.append(r)
        print(json.dumps(r), flush=True)
    (OUT / 'validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')

    lines = ['# d1/d2 缩放版：C1/C2 按实际范围映射 2^d2 个整数档位', '',
             '模型同 fp16_d1d2：2^d1 段，t=段内 FP16 ULP 计数，P(t)=C0+C1*t+C2*t²，C0 为 FP16。',
             '区别：C1、C2 不再用裸 2^-d2 格点，而是各自的实际范围（泰勒展开导出）映射到 2^d2 档：',
             'C1 实际范围 [-2^-10, -2^-12]（负），C2 实际范围 [2^-23, 2^-20]（正）；',
             '范围两端各放宽 1/16 跨度后均分为 2^d2 档，C = lo + m*step，m 为无符号 d2 位整数，',
             '符号由偏移承担。C1 恒负、C2 恒正，不再出现 0。搜索同 fp16_d1d2（必要条件剪枝完备）。',
             '评估穷举 [1,2) 全部 1024 个 FP16 输入。FP16 输出 ulp = 2^-11。', '',
             '| d1 | d2 | step1 | step2 | 最大绝对误差 | MAE | 最大误差(输出ulp) | FP16正确舍入不符 | 旧方案同d2最大误差 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        o = r['old_scheme_max_error']
        o = f'{o:.10g}' if o else '-'
        lines.append(f"| {r['d1']} | {r['d2']} | {r['step1']:.3e} | {r['step2']:.3e} "
                     f"| {r['max_absolute_error']:.10g} | {r['mean_absolute_error']:.10g} "
                     f"| {r['max_error_fp16_output_ulp']:.6f} | {r['fp16_correct_rounding_mismatches']} | {o} |")
    lines += ['', '旧方案列为 fp16_d1d2（C1/C2 用裸 2^-d2 格点）同参数的最大绝对误差。',
              '运行：先 `python generate_coefficients.py`（可带 d1 d2 只跑一组），再 `python evaluate_accuracy.py`。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
