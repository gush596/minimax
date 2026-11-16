# 倒数近似实验

包含原论文定点复现、仅替换量化规则的 FP16 三步法，以及 FP16 受限系数组合搜索。

论文：*High-Speed Function Approximation Using a Minimax Quadratic Interpolator*，IEEE Transactions on Computers，2005，DOI：[10.1109/TC.2005.52](https://doi.org/10.1109/TC.2005.52)。

本项目复现三遍系数生成流程，未逐位复现论文专用截断平方器、融合累加树和舍入偏置。原始方法中的偏置与截断平方为本项目参考模型，不能当作作者原始硬件配置。FP16 三步法不加偏置，使用 float64 乘加评估。

使用 Python 3.13 验证。先克隆仓库：

```powershell
git clone https://github.com/waterkfc/minimax.git
cd minimax
```

| 目录 | 内容 | 入口 |
|---|---|---|
| [original/](original/) | 原论文方法的定点系数数值复现，含论文、测试和原始结果 | `reproduce.py` |
| [fp16_three_pass/](fp16_three_pass/) | 保留原三步流程，仅替换为受限 FP16 量化，d=1～11 | `generate_coefficients.py`、`evaluate_accuracy.py` |
| [fp16_constrained/](fp16_constrained/) | 合法 FP16 系数组合搜索，128分段，d=1～11 | `generate_coefficients.py`、`evaluate_accuracy.py` |

## 原始复现

```powershell
cd original
python -m pip install -r requirements.txt
python -B reproduce.py
python -B -m unittest -v
cd ..
```

结果：[original/results/REPORT.md](original/results/REPORT.md)。

## FP16 三步法

```powershell
cd fp16_three_pass
python -m pip install -r requirements.txt
python -B generate_coefficients.py
python -B evaluate_accuracy.py
cd ..
```

结果：[fp16_three_pass/results/REPORT.md](fp16_three_pass/results/REPORT.md)。

三步法复用 `original/reproduce.py` 中的高精度 Remez 求解，因此请保留两个目录的相对位置。

各套实验分别读写各自的 `results/`，重新运行会覆盖对应结果。

## FP16 组合搜索

从仓库根目录运行：

```powershell
cd fp16_constrained
python -m pip install -r requirements.txt
python -B generate_coefficients.py
python -B evaluate_accuracy.py
cd ..
```

结果：[fp16_constrained/results/REPORT.md](fp16_constrained/results/REPORT.md)。该方法使用剪枝和误差下界提前终止，不改变另外两套实验。
