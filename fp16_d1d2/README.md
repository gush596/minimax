# d1/d2 版本：FP16 输入网格 + C0 FP16 + C1/C2 d2 位小数

- **分段**：2^d1 段覆盖 [1,2)，d1 默认 5（32 段）。
- **多项式输入**：t = (x − 段起点) × 1024 的整数部分，即把 x 当成 FP16 后尾数的低 10−d1 位
  （段内 FP16 ULP 计数）。P(t) = C0 + C1·t + C2·t²。
- **C0**：任意 FP16 值（无掩码）。
- **C1、C2**：2^-d2 的整数倍（二进制小数最多 d2 位，任意符号）。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py            # 全部扫描组合
python -B generate_coefficients.py 5 16       # 只跑 d1=5, d2=16
python -B evaluate_accuracy.py
```

- 搜索：热启动（最小二乘/二点插值 + 格点舍入）后，用必要条件（残差差分 ≤ 2E）限定 C2 框、
  按 C2 剪枝 C1，全枚举可改进热启动的格点对；最终误差在段内离散 FP16 网格上精确计算。
  相对热启动误差是完备搜索（不漏更优解）。
- 评估：穷举 [1,2) 全部 1024 个 FP16 输入；不做输出舍入；另统计多项式输出与 1/x 各自
  舍入到 FP16 后的正确舍入不符数。
- `results/coefficients_d1_*_d2_*.csv` 保存系数（C0 FP16 hex、C1/C2 的定点整数分子）、
  `results/REPORT.md` 与 `validation.json` 为全部指标。

重新运行会覆盖本目录结果。
