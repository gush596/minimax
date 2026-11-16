"""Read generated CSV coefficients and exhaustively evaluate FP32 inputs."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
BASELINE = ROOT.parent / 'fp16_constrained' / 'results' / 'validation.json'
FREE = ROOT.parent / 'fp16_free_c0' / 'results' / 'validation.json'


def validate(coeff):
    worst_abs = worst_rel = 0.
    worst_input = None
    squared = 0.
    absolute_sum = 0.
    count = 0
    for i, (c0, c1, c2) in enumerate(coeff):
        t = np.arange(65536, dtype=np.float64)
        x = t / 2**23
        z = 1 + i/128 + x
        y = c0 + c1*x + c2*x*x
        err = y - 1/z
        j = int(np.argmax(abs(err)))
        if abs(err[j]) > worst_abs:
            worst_abs = float(abs(err[j]))
            worst_input = dict(x=float(z[j]), output=float(y[j]), reference=float(1/z[j]),
                               input_bits_hex=f'{0x3f800000+i*65536+j:08x}')
        worst_rel = max(worst_rel, float(np.max(abs(err)*z)))
        squared += float(np.dot(err, err))
        absolute_sum += float(np.sum(np.abs(err), dtype=np.float64))
        count += len(t)
    return dict(tested_inputs=count, max_absolute_error=worst_abs,
                mean_absolute_error=absolute_sum/count,
                max_relative_error=worst_rel, max_error_q24_ulp=worst_abs*2**24,
                rms_absolute_error=float(np.sqrt(squared/count)), worst_case=worst_input)


def main():
    reports = []
    for d in range(1, 12):
        path = OUT/f'coefficients_d{d}.csv'
        if not path.exists():
            raise SystemExit(f'Missing {path.name}; run generate_coefficients.py first.')
        rows = np.genfromtxt(path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        coeff = np.column_stack([rows[name] for name in ['C0', 'C1', 'C2']])
        if coeff.shape != (128, 3) or not np.array_equal(rows['segment'], np.arange(128)):
            raise ValueError(f'{path.name}: expected segments 0..127')
        stored = coeff.astype(np.float16)
        if not np.all(np.isfinite(coeff)) or not np.array_equal(stored.astype(np.float64), coeff):
            raise ValueError(f'{path.name}: coefficients must be finite FP16 values')
        if np.any(stored.view(np.uint16)[:, 1:] & ((1 << (11-d))-1)):
            raise ValueError(f'{path.name}: C1/C2 fraction bits violate d={d}')
        r = dict(d=d, candidate_values_c0=63487, candidate_values_c1c2=62*2**(d-1)-1,
                 continuous_max_absolute_error=float(max(rows['continuous_max_abs_error'])),
                 **validate(coeff))
        reports.append(r)
        print(json.dumps(r), flush=True)
    (OUT/'validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')

    baseline, free = {}, {}
    if BASELINE.exists():
        baseline = {r['d']: r for r in json.loads(BASELINE.read_text())}
    if FREE.exists():
        free = {r['d']: r for r in json.loads(FREE.read_text())}
    lines = ['# C0 全 FP16 + C1/C2 受限 FP16 实验', '',
             '128 分段，P(x)=C0+C1*x+C2*x²，x∈[0,1/128)。C0 取全 FP16 集合（无 d 掩码）；',
             'C1/C2 约束同 fp16_constrained：d=1～11，低 11-d 位尾数为零的合法 FP16 值。',
             '搜索方法同 fp16_constrained：d≤4 筛选 C0 后枚举 C1、C2；d≥5 先试邻域再分批搜索，',
             '采样点剪枝 + 端点/驻点精确检查。float64 容差 2e-14；不是形式化全局最优证明。',
             '无偏置；系数 FP16 存储，float64 乘加与误差评估，不模拟 FP16 运算或输出舍入。',
             '每组系数穷举 [1,2) 全部 8,388,608 个 FP32 输入。Q24 ulp=2^-24，仅为误差单位。', '',
             '平均误差采用 MAE：所有输入的绝对误差之和除以输入数量。', '',
             '| d | 最大绝对误差 | MAE | 最大相对误差 | RMS | 基线(全部受限)最大误差 | C0自由 最大误差 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        b = baseline.get(r['d'])
        f = free.get(r['d'])
        bmax = f"{b['max_absolute_error']:.10g}" if b else '-'
        fmax = f"{f['max_absolute_error']:.10g}" if f else '-'
        lines.append(f"| {r['d']} | {r['max_absolute_error']:.10g} | {r['mean_absolute_error']:.10g} "
                     f"| {r['max_relative_error']:.6%} | {r['rms_absolute_error']:.10g} "
                     f"| {bmax} | {fmax} |")
    lines += ['', '基线列为 fp16_constrained（C0 同样 d 受限）的最大绝对误差；',
              'C0自由列为 fp16_free_c0（C0 最优实数）的最大绝对误差。',
              '运行：先 `python generate_coefficients.py`，再 `python evaluate_accuracy.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
