# FP16 受限系数组合搜索

与 `../fp16_three_pass/` 的固定三步量化不同，本方法从合法 FP16 数集合中重新选择 C0、C1、C2 的组合。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py
python -B evaluate_accuracy.py
```

- 128分段；d=1～11，低11−d位尾数为零。候选含所有满足位掩码的有限FP16数，包括正负数、零、次正规数。
- 每段最小化连续区间上的最大绝对误差。d≤4从常数近似开始筛选C0并枚举C1、C2；d≥5先尝试邻域组合，未达到端点误差下界时继续分批搜索。
- 端点和采样点的必要条件用于剪枝；最后检查端点与全部内部驻点。float64数值容差2e-14，不是形式化全局最优证明。
- 无偏置；系数FP16存储，float64乘加，不模拟FP16运算或输出舍入。未优化并列解的系数大小、MAE或RMS。
- `results/coefficients_d*.csv` 保存系数；`results/REPORT.md` 和 `validation.json` 保存全部8388608个FP32输入的最大绝对误差、平均绝对误差MAE等指标。

两个脚本只读写本目录下的结果；重新运行会覆盖对应结果。
