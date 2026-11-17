# d1/d2 纯 2^n 缩放版

`../fp16_d1d2_scaled/` 的简化：C1/C2 不再用"偏移 + 非 2 幂步长"的仿射映射，而是把各自的
实际范围用 2^n（n 整数）**缩放到最大值为 2^d2**，直接存 [0, 2^d2-1] 的无符号整数：

```
C1 = -m1 · 2^-(d2+10)     m1 ∈ [0, 2^d2-1]   （C1 恒负）
C2 = +m2 · 2^-(d2+20)     m2 ∈ [0, 2^d2-1]   （C2 恒正）
```

缩放指数来自泰勒范围最大值 |C1|max = 2^-10、C2max = 2^-20（见 fp16_d1d2_scaled/README.md）。
硬件实现即纯移位：无偏移加法、无非 2 幂乘法；符号由系数承担。

模型、搜索、评估与 `../fp16_d1d2/` 相同（2^d1 段、t=段内 FP16 ULP 计数、C0 为 FP16、
必要条件剪枝完备搜索、穷举 1024 个 FP16 输入）。

```powershell
python -m pip install -r requirements.txt
python -B generate_coefficients.py            # 全部扫描组合
python -B generate_coefficients.py 5 6        # 只跑 d1=5, d2=6
python -B evaluate_accuracy.py
```

结论：饱和点与仿射版几乎相同（d1=5 需 d2=6、d1=7 需 d2=5、d1=3 需 d2=8，即达 C0 FP16
底板 ~2.2-2.4e-4）；仅在极低 d2（2~4）比仿射版差 1.3~2×（2 幂步长覆盖范围不紧凑）。
甜点配置 d1=5, d2=6：C1=-m1·2^-16, C2=m2·2^-26，每段 ROM = C0(16b)+m1(6b)+m2(6b)=28 bit。
