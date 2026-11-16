# MEMORY.md — 长期记忆

## 项目：minimax 倒数近似实验

- 仓库：本地 `minimax/`；来源 https://github.com/waterkfc/minimax.git（origin）；
  **发布仓库 https://github.com/gush596/minimax.git（remote 名 `publish`）**。
- 内容：复现论文 *High-Speed Function Approximation Using a Minimax Quadratic Interpolator*
  (Pineiro et al., IEEE TC 2005) 的倒数 1/x 二次 minimax 近似 + FP16 受限系数组合搜索。
- 我做的扩展实验（2026-11-16）：
  - `fp16_free_c0/`：C0 无约束（连续最优实数），C1/C2 受限 FP16。比基线好 750~1050×，
    d=11 最大误差 2.36e-7；误差由 C1 量化(δ·H/2) 和 minimax 下界 1.47e-8 决定。
  - `fp16_c0_fp16/`：C0 全 FP16（无 d 掩码），C1/C2 受限 FP16。d≥3 误差饱和在
    2.419e-4（C0 的 FP16 底板）；d=11 与 fp16_constrained 基线逐位一致（交叉验证）。
  - `fp16_d1d2/`（用户新版本）：2^d1 段（默认 d1=5），x=段内 FP16 ULP 整数计数，
    C0 限 FP16，C1/C2 限小数最多 d2 位。误差饱和于 C0 底板；达标所需 d2：
    d1=3→24，d1=5→20，d1=7→16，d1=10→无关。
- 精度归因链：误差 = C0 量化底板 + C1/C2 量化 + minimax 逼近极限(1.47e-8)。

## 教训

- 搜索最优系数组合时不要假设最优点在连续解附近：凸误差函数在格点上可以有远离连续解的
  补偿解（如最优 C2 = -0.125 vs 连续解 ≈ +1）。窗口搜索会漏掉，必须用必要条件剪枝全搜索。
- 交叉验证习惯：d=11 三系数同为全 FP16 时，不同实验必须逐位一致。

## 规则 / 待办

- **每次修改 minimax 代码或记忆文档后，都要 commit 并 push 到 https://github.com/gush596/minimax.git（remote `publish`）**。
- 推送认证：SSH deploy key（ed25519），已配置在仓库 `core.sshCommand`；
  私钥存 `.openclaw/tmp/github_gush596_ed25519`（chmod 600，禁止回显/外发/复制到其他路径）。
- 记忆文档双份：工作区 `MEMORY.md` + `memory/` 为活文档；每次推送时同步一份到仓库 `memory/` 目录（用户要求）。
- 注意：私钥曾通过聊天/CDN 链接传输，如在意可提醒用户轮换。
