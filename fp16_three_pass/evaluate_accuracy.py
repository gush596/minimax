"""Read generated CSV coefficients and exhaustively evaluate FP32 inputs."""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent / "results"


def validate(coeff):
    worst_abs = worst_rel = 0.
    worst_input = None
    squared = 0.
    absolute_sum = 0.
    count = 0
    for i, (c0, c1, c2) in enumerate(coeff):
        t = np.arange(65536, dtype=np.float64)
        x = t/2**23
        z = 1+i/128+x
        y = c0+c1*x+c2*x*x
        err = y-1/z
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
        if np.any(stored.view(np.uint16) & ((1 << (11-d))-1)):
            raise ValueError(f'{path.name}: coefficient fraction bits violate d={d}')
        r = dict(d=d, candidate_values=62*2**(d-1)-1,
                 continuous_max_absolute_error=float(max(rows['continuous_max_abs_error'])),
                 **validate(coeff))
        reports.append(r)
        print(json.dumps(r), flush=True)
    (OUT/'validation.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    lines = ['# 受限 FP16 系数实验', '',
             '128 分段，P(x)=C0+C1*x+C2*x²，x∈[0,1/128)。d 包含隐含首位；低 11-d 位尾数为零。',
             '无偏置；系数 FP16 存储，float64 乘加与误差评估，不模拟 FP16 运算或输出舍入。',
             '保留三步流程：Remez求初始系数；量化C1；按a2+(a1-C1)*128修正并量化C2；残差极值中点量化为C0。',
             '三次量化均直接舍入到受限FP16集合，最近舍入、平局取偶；不是先转FP16再截断。',
             '每组系数穷举 [1,2) 全部 8,388,608 个 FP32 输入。Q24 ulp=2^-24，仅为误差单位。', '',
             '平均误差采用 MAE：所有输入的绝对误差之和除以输入数量，不是带符号误差的平均值。', '',
             '| d | 最大绝对误差 | 平均绝对误差 MAE | 最大相对误差 | Q24 ulp | RMS 绝对误差 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in reports:
        lines.append(f"| {r['d']} | {r['max_absolute_error']:.10g} | {r['mean_absolute_error']:.10g} | {r['max_relative_error']:.6%} | {r['max_error_q24_ulp']:.6f} | {r['rms_absolute_error']:.10g} |")
    lines += ['', '运行：先 `python generate_coefficients.py`，再 `python evaluate_accuracy.py`。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')

if __name__ == "__main__":
    main()
