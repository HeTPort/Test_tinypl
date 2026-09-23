# AVS NPU 确定性输入集

本目录提供面向手机 NPU AVS/DVFS 测试的合成输入。它用于稳定地产生计算负载和检查降压后的输出一致性，不是 MLPerf 准确率数据集，也不附带语义标签。

## 输入规模

| 负载 | Smoke | Stress | 数据格式 |
| --- | ---: | ---: | --- |
| `ic01` | 8 个 | 32 个 | `32×32×3`，HWC，`uint8` |
| `kws01` | 8 个 | 32 个 | `49×10×1`，`int8` |
| `ad01` | 8 个 | 32 个 | `5×128`，小端 `float32` |
| `sww01` | 8 个窗口 | 32 个窗口 + 256 帧特征流 | 窗口 `30×40`、帧 `40`，`int8` |

每类输入都覆盖以下模式：

- 低激活或低动态范围；
- 密集伪随机；
- 梯度、条纹、棋盘或频带等结构化数据；
- 最小值、最大值、脉冲和交替边界值。

## 目录

```text
input/
├── generate_inputs.py
├── manifest.json
├── ic01/
│   ├── manifest.json
│   ├── smoke/
│   └── stress/
├── kws01/
├── ad01/
└── sww01/
    ├── smoke/
    └── stress/
        ├── windows/
        └── stream_features_256x40_int8.bin
```

二进制文件不包含文件头。shape、dtype、生成模式、seed、字节数和 SHA-256 均记录在相应的 `manifest.json` 中。

## 重新生成

仅依赖 Python 3 标准库：

```bash
python input/generate_inputs.py
python input/generate_inputs.py --verify
```

生成器使用代码内实现的 PCG32，避免依赖 Python、C++ 标准库各自的随机分布实现。相同生成器版本、seed 和参数应产生完全相同的文件。

## 在 benchmark 中使用

1. 程序初始化阶段读取或生成整个输入库。
2. 在 warmup 和正式测试期间按 `batch_index % sample_count` 轮换输入。
3. 输入准备应位于 inference-only 计时区间之外。
4. 正常电压下为每个输入建立模型输出 Golden。
5. AVS 测试中持续比较输出；INT8 可逐元素或 CRC 校验，FP16 应使用数值容差。

`ad01` 当前使用 `float32`，与仓库中 5×128 特征设置一致。若最终模型输入为量化 INT8，应在模型确定后新增独立 profile，不要在不记录版本的情况下直接覆盖本输入集。

`sww01` 的 256 帧特征流由四个连续 64 帧区段组成：低激活、频带结构、周期脉冲和密集随机。32 个 stress 窗口从该特征流中均匀抽取，每个窗口包含 30 帧。
