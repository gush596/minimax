# d1/d2 缩放版：C1/C2 按实际范围映射 2^d2 个整数档位

修复 `../fp16_d1d2/` 的问题（裸 2^-d2 格点与 C1/C2 的实际量级不匹配，小 d2 时大量系数为 0）。
模型不变：2^d1 段、t = 段内 FP16 ULP 计数、P(t) = C0 + C1·t + C2·t²、C0 为 FP16。

**系数表示**：C1/C2 从 1/x 在段起点的泰勒展开导出实际范围，各自映射到 2^d2 档：

| 参数 | 推导 | 实际范围 | 符号 |
|---|---|---|---|
| C1 | f′=−1/x²∈[−2⁰,−2⁻²]，×2⁻¹⁰ | [−2⁻¹⁰, −2⁻¹²] | 恒负 |
| C2 | f″/2=1/x³∈[2⁻³,2⁰]，×2⁻²⁰ | [2⁻²³, 2⁻²⁰] | 恒正 |

范围两端各放宽 1/16 跨度（防最优解压边界）后均分 2^d2 档：C = lo + m·step，
m 为无符号 d2 位整数，符号由偏移 lo 承担。C1/C2 永不为 0。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py            # 全部扫描组合
python -B generate_coefficients.py 5 8        # 只跑 d1=5, d2=8
python -B evaluate_accuracy.py
```

搜索与评估同 `../fp16_d1d2/`（必要条件剪枝完备搜索；穷举 1024 个 FP16 输入）。
`results/REPORT.md` 含与旧方案同参数的对比列。

结论：d1=5 配 d2=5~6 即达 C0 FP16 底板（~0.47 输出 ulp）；旧方案同档需要 d2=20。
