# C0 无约束 + C1/C2 受限 FP16 实验

与 `../fp16_constrained/` 相同的 C1/C2 约束（d=1～11，低 11−d 位尾数为零的合法 FP16 值），但
**C0 不再限制为 FP16 候选值**，取每段连续最优实数（残差极值中点），考察 C0 量化对误差的贡献。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py
python -B evaluate_accuracy.py
```

- 128 分段；d≤5 对 (C1,C2) 全枚举；d≥6 用采样点必要条件（|r(x)−r(0)|≤2E）剪枝后全搜索，
  并在 d=6 抽样段上与全枚举结果一致。
- 每段记录 `minimax_error_bound`：C0/C1/C2 全自由时的连续 minimax 误差，是可达误差的全局下界。
- 无偏置；C1/C2 FP16 存储，float64 乘加与误差评估，不模拟 FP16 运算或输出舍入。
- `results/coefficients_d*.csv` 保存系数（C0 为 float64）；`results/REPORT.md` 与基线对比。

两个脚本只读写本目录下的结果；重新运行会覆盖对应结果。需要保留 `../original/`（Remez 求解）与
`../fp16_constrained/results/validation.json`（基线对比列）的相对位置。
