# RTL：FP16 倒数单元数据通路

主推配置 **d1=3, d2=7**（8 段、t 7 bit、m1/m2 各 7 bit）：

```
C1 = -m1 · 2^-17,  C2 = m2 · 2^-27,  C0 = m0 · 2^-11 (m0 1024..2048)
y·2^11 = round( m0·2^16 + t·(m2·t − 1024·m1) )
```

运行时只有：ROM 读取、2 个小整数乘法（m2·t、t·inner）、纯连线移位、一次减法一次加法、
最后一次舍入移位。**无任何浮点运算**；输出 y·2^11 直接就是 FP16 尾数（指数域恒为 −1）。

## 文件

| 文件 | 说明 |
|---|---|
| `fp16_recip.v` | 参数化 Verilog（D1/D2/ROMFILE 可配，默认 3/7），4 级流水，顺序输出 |
| `tb_fp16_recip.v` | 测试平台：穷举 1024 个 FP16 输入，对照 1/x 实数参考，容差 2 LSB |
| `gen_rom.py` | 从 `../results/coefficients_d1_*_d2_*.csv` 生成 ROM 初始化文件 |
| `rom_d1_3_d2_7.memh` | 主配置 ROM：8 条 × 26 bit = {m2[6:0], m1[6:0], m0[11:0]} |
| `datapath_model.py` | 位精确 Python 模型（逐位复现整数流水线），无需仿真器即可验证 |

## 数据通路（4 级流水）

```
S0: 拆分 frac → seg[2:0] = frac[9:7], t[6:0] = frac[6:0]
S1: ROM 读取 (m0 12b, m1 7b, m2 7b)
S2: p = m2·t;  inner = p − (m1<<10)          // 一次乘法 + 移位 + 减法
S3: core = t·inner;  acc = (m0<<16) + core;  y = (acc + 2^15) >> 16   // 乘法 + 加法 + 舍入
    fp16_out = y==2048 ? 0x3c00 : {5'b01110, y[9:0]}
```

## 验证

```powershell
# 位精确模型验证（无需 Verilog 仿真器）
python3 datapath_model.py 3 7

# 有 iverilog 时：
python3 gen_rom.py 3 7
iverilog -o tb fp16_recip.v tb_fp16_recip.v
vvp tb
```

datapath_model.py 实测（d1=3, d2=7，1024 输入）：
- inner = m2·t − 1024·m1 全程 ≤ 0（无符号假设成立）、y·2^11 全程 ∈ [1024, 2048]
- 整数流水线结果与 float64 多项式**逐位一致**（同为 half-up 舍入）
- 对 1/x 最大误差 4.33e-4 = 0.89 ulp（多项式 0.45 ulp + 输出舍入 0.5 ulp）
- FP16 正确舍入不符 142/1024

注意：`gen_rom.py` 的 m0 为 12 bit（C0=1.0 时 m0=2048=2^11，11 bit 会溢出）；
整数核心符号为 m2·t − 1024·m1（C1 项是减法）。这两处曾出错，由 datapath_model 抓出。
