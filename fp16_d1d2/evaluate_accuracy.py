"""Read generated CSV coefficients and evaluate every FP16 input in [1,2)."""
from pathlib import Path
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
U = 1 / 1024.0


def evaluate(d1, d2, rows):
    step = 2.0 ** -d2
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
    return dict(d1=d1, d2=d2, segments=2**d1, points_per_segment=2**(10 - d1),
                tested_inputs=count, max_absolute_error=worst_abs,
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
    reports = []
    for path in files:
        d1, d2 = map(int, re.findall(r'd1_(\d+)_d2_(\d+)', path.stem)[0])
        data = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        rows = np.column_stack([data[n] for n in ['C0', 'C1', 'C2']])
        if rows.shape != (2**d1, 3) or not np.array_equal(data['segment'], np.arange(2**d1)):
            raise ValueError(f'{path.name}: expected {2**d1} segments')
        stored = rows[:, 0].astype(np.float16)
        if not np.array_equal(stored.astype(np.float64), rows[:, 0]):
            raise ValueError(f'{path.name}: C0 must be FP16-representable')
        step = 2.0 ** -d2
        for col in (1, 2):
            m = rows[:, col] / step
            if not np.all(np.abs(m - np.round(m)) < 1e-6):
                raise ValueError(f'{path.name}: C1/C2 violate the d2={d2} fraction grid')
        r = evaluate(d1, d2, rows)
        reports.append(r)
        print(json.dumps(r), flush=True)
    (OUT / 'validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')

    lines = ['# d1/d2 版本：FP16 输入网格 + C0 FP16 + C1/C2 d2 位小数', '',
             '2^d1 段覆盖 [1,2)；多项式输入 t = (x-段起点)×1024 的整数部分，即 FP16 输入尾数的',
             '低 10-d1 位（段内 ULP 计数）。P(t)=C0+C1*t+C2*t²。C0 为任意 FP16 值；C1、C2 为',
             '2^-d2 的整数倍（小数最多 d2 位）。搜索为必要条件剪枝全枚举（相对热启动误差完备）。',
             '评估穷举 [1,2) 全部 1024 个 FP16 输入；误差按 float64 多项式计算，不做输出舍入',
             '（FP16 正确舍入不符数为把多项式输出与 1/x 各自舍入到 FP16 后的差异数，1024 个输入中）。', '',
             'FP16 输出 ulp = 2^-11。', '',
             '| d1 | d2 | 段数 | 最大绝对误差 | MAE | 最大相对误差 | RMS | 最大误差(输出ulp) | FP16正确舍入不符 |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['d1']} | {r['d2']} | {r['segments']} | {r['max_absolute_error']:.10g} "
                     f"| {r['mean_absolute_error']:.10g} | {r['max_relative_error']:.6%} "
                     f"| {r['rms_absolute_error']:.10g} | {r['max_error_fp16_output_ulp']:.6f} "
                     f"| {r['fp16_correct_rounding_mismatches']} |")
    lines += ['', '运行：先 `python generate_coefficients.py`（可带 d1 d2 只跑一组），',
              '再 `python evaluate_accuracy.py`。']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
