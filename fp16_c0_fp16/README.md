# C0 全 FP16 + C1/C2 受限 FP16 实验

与 `../fp16_constrained/` 的区别只在 C0 的候选集：**C0 取全 FP16 集合（无 d 掩码）**，
C1/C2 仍为 d=1～11 的受限 FP16 值。搜索方法与 fp16_constrained 相同（采样点剪枝 + 端点/驻点精确检查）。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py
python -B evaluate_accuracy.py
```

- d=11 时三系数同为全 FP16，与 fp16_constrained 的 d=11 结果逐位一致（用作交叉验证）。
- 无偏置；系数 FP16 存储，float64 乘加与误差评估，不模拟 FP16 运算或输出舍入。
- `results/coefficients_d*.csv` 保存系数与 FP16 hex；`results/REPORT.md` 含基线与 C0自由 两列对比。

重新运行会覆盖本目录结果。基线对比读取 `../fp16_constrained/results/validation.json`，
C0自由对比读取 `../fp16_free_c0/results/validation.json`（缺失时对应列显示 `-`）。
