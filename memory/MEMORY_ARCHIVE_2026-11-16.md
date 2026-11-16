# 记忆存档 — 2026-11-16

（本文件为 `memory/2026-11-16.md` + `MEMORY.md` 当前内容的存档快照。）

## 今日工作记录

- 克隆并分析了 https://github.com/waterkfc/minimax.git（倒数近似实验：论文 TC2005 minimax
  二次插值复现 + FP16 受限系数组合搜索）。三套原始实验：`original/`（论文定点复现）、
  `fp16_three_pass/`（三遍法+受限FP16量化）、`fp16_constrained/`（受限FP16系数组合搜索）。
- 扩展实验 1：`minimax/fp16_free_c0/` —— C0 无约束（连续最优实数）、C1/C2 保持受限 FP16。
  - 结果：相对 fp16_constrained 基线全 d 改善约 750~1050×；d=11 最大误差 2.36e-7（基线 2.42e-4）。
  - 教训：最初 d≥6 用"连续 minimax 解附近窗口搜索"漏掉远点补偿解（最优 C2 可远离连续解，
    如 -0.125），d=6 误差反超 d=5；改为采样点必要条件剪枝全搜索后修复，与全枚举逐位一致。
  - 误差主导因素：基线中 C0 量化（≈FP16 半 ulp）占主导；放开 C0 后由 C1 量化(δ·H/2)
    和 minimax 下界 1.47e-8 决定。
- 扩展实验 2：`minimax/fp16_c0_fp16/` —— C0 全 FP16（无 d 掩码）、C1/C2 d 约束。
  - d=11 与 fp16_constrained 逐位一致（2.419008028e-4，交叉验证通过）。
  - d≥3 误差饱和在 2.419e-4（C0 的 FP16 底板，由某段 1/a 到最近 FP16 的距离决定），
    C1/C2 再加位数无收益。
- 三档对比结论：若 C0 存 FP16，精度上限 ~2.4e-4（≈4058 Q24 ulp），C1/C2 给 d=3 即够；
  要突破需 C0 升精度或加偏置修正。

## 长期记忆（MEMORY.md 摘要）

- 发布仓库：https://github.com/gush596/minimax.git；**每次修改 minimax 代码后必须 commit + push**。
- 精度归因链：误差 = C0 量化底板 + C1/C2 量化(δ·H/2 量级) + minimax 逼近极限(1.47e-8)。
- 教训：格点最优解可远离连续解，窗口搜索不可靠，必要条件剪枝全搜索才稳；关键档位做逐位交叉验证。
- 待办：GitHub 推送凭据待配置（本机无 gh CLI / credential helper / token）。
