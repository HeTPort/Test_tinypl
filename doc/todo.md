# 手机 NPU 低功耗与 AVS Benchmark TODO

## 目标

将当前 MLPerf Tiny 派生负载改造成可在 UDP 单板上运行的手机 NPU benchmark，覆盖量产手机软件环境中的鸿蒙单框架，以及安卓、鸿蒙双框架。测试结果应能区分模型计算、运行时开销、系统功耗和 AVS/DVFS 策略影响。

## 总体验收条件

- 同一模型、输入和预处理可以在参考 CPU、安卓 NPU 后端、鸿蒙 NPU 后端上运行。
- 每个后端能够证明实际执行设备和算子分区情况，不能静默回退后仍标记为 NPU 结果。
- 每个负载具有模型哈希、输入哈希、真值、参考输出和精度阈值。
- 冷启动、热启动、持续运行和周期唤醒具有明确计时边界。
- 每个 AVS/DVFS 测试点记录电压、频率、温度、功率、延迟、吞吐和正确性。
- 同一实验可根据配置文件在另一块同型号单板上复现。

---

## P0：恢复模型、输入、真值和预处理闭环

> 这是当前最高优先级。先恢复预处理后的推理测试数据；完整训练数据和训练环境可后续恢复。

### 0.1 确认上游版本

- [ ] 将当前文件与 [MLCommons Tiny](https://github.com/mlcommons/tiny) 的固定 commit 做 SHA 对照。
- [ ] 记录上游仓库、commit、许可证和本仓库的裁剪说明。
- [ ] 优先检查当前代码对应的上游 commit `4addd0fa08d216e20637637874e084895f289da4`。
- [ ] 建立 `provenance/upstream_files.csv`，记录本地文件、上游文件和 SHA-256。

### 0.2 恢复运行配置

- [ ] 恢复 `runner/tests_accuracy.yaml`。
- [ ] 恢复 `runner/tests_performance.yaml`。
- [ ] 恢复 `runner/tests_energy.yaml`。
- [ ] 将循环次数、预热次数和最短运行时间改成显式配置项。
- [ ] 保留原始配置作为 `legacy_mlperf_tiny` profile，手机场景使用新的 profile。

上游参考：

- [tests_accuracy.yaml](https://github.com/mlcommons/tiny/blob/master/benchmark/runner/tests_accuracy.yaml)
- [tests_performance.yaml](https://github.com/mlcommons/tiny/blob/master/benchmark/runner/tests_performance.yaml)
- [tests_energy.yaml](https://github.com/mlcommons/tiny/blob/master/benchmark/runner/tests_energy.yaml)

### 0.3 恢复四类负载资产

#### 图像分类 IC01

- [ ] 恢复量化 ResNet/TFLite 模型及模型 SHA-256。
- [ ] 恢复 CIFAR-10 测试输入和 `y_labels.csv`。
- [ ] 明确输入为 `32×32×3`、布局、RGB/BGR、数据类型、scale 和 zero point。
- [ ] 检查当前 `ic_inputs.cc` 中近似全零样本是否只是占位数据。
- [ ] 生成至少 20 个样本的冒烟集和完整准确率集。

#### 关键词识别 KWS01

- [ ] 恢复 DS-CNN 量化模型及模型 SHA-256。
- [ ] 恢复预定义测试输入、类别标签和 `y_labels.csv`。
- [ ] 明确 `49×10×1` 特征张量的时间轴、频率轴和内存布局。
- [ ] 恢复 MFCC 或 LFBE 生成参数、量化参数和生成脚本。
- [ ] 先生成至少 12 类各一个样本的冒烟集，再恢复官方准确率集。

上游参考：[KWS 训练与测试输入生成说明](https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/README.md)

#### 异常检测 AD01

- [ ] 恢复 ToyADMOS/ToyCar 模型和量化版本。
- [ ] 恢复测试特征文件、正常/异常标签和 `y_labels.csv`。
- [ ] 恢复 `bytes_to_send`、`stride` 与滑窗生成规则。
- [ ] 明确 5×128 输入与原始 200 个频谱切片之间的转换关系。
- [ ] 生成固定阈值、ROC-AUC 和 PR-AUC 的参考结果。

上游参考：[异常检测参考模型](https://github.com/mlcommons/tiny/tree/master/benchmark/training/anomaly_detection)

#### 流式唤醒词 SWW01

- [ ] 恢复 1D DS-CNN 模型和 `sww_model_data.h`。
- [ ] 恢复 `fixed_data.h` 及 Mel/filterbank、窗函数等固定参数。
- [ ] 恢复测试 WAV 和 `sww_long_test.json` 检测时间窗。
- [ ] 恢复并核对 `th_process_chunk_and_cont_streaming()` 的参考实现。
- [ ] 明确 16 kHz、1024 样本窗、512 样本步长、40 个 Mel 特征和 1200 元素模型输入的关系。
- [ ] 核对检测阈值 115、去抖动和误报抑制窗口的语义。

### 0.4 建立统一资产清单

- [ ] 为每个负载创建 `manifest.json`。
- [ ] 记录模型、输入和标签文件的 SHA-256。
- [ ] 记录输入输出名称、shape、layout、dtype、scale、zero point。
- [ ] 记录预处理版本、类别顺序和后处理方法。
- [ ] 保存每个冒烟样本的参考输出或参考输出哈希。
- [ ] 大型或受许可证限制的数据只提供下载与生成脚本，不直接提交仓库。

建议结构：

```text
assets/
├── manifests/
│   ├── ad01.json
│   ├── ic01.json
│   ├── kws01.json
│   └── sww01.json
├── scripts/
│   ├── download.py
│   ├── prepare.py
│   └── verify.py
├── smoke/
└── checksums.json
```

### P0 完成条件

- [ ] 四个负载都能在 PC 参考运行时完成推理。
- [ ] 重复生成的数据文件哈希一致。
- [ ] 输出满足参考精度或逐元素容差。
- [ ] 任意成员可按 README 从空目录重建冒烟数据。

---

## P1：建立正确性基线

- [ ] 为每个负载选择 10～20 个固定冒烟样本。
- [ ] 保存 FP32 或原始框架输出作为一级参考。
- [ ] 保存量化 CPU 输出作为二级参考。
- [ ] 定义逐元素绝对误差、相对误差和分类一致性阈值。
- [ ] 检查输入布局、字节序、signed/unsigned INT8 和量化参数。
- [ ] 失败时转储输入、输出、后端、分区和运行时版本。
- [ ] 修正分类输出简单除以总和的问题；logits 只用于 argmax，概率计算使用明确 softmax。
- [ ] 修正异常检测指标，将 ROC-AUC、PR-AUC、F1 和固定阈值指标分开。
- [ ] 核对 SWW 检测时间窗是否重复加入误报抑制延迟。

### P1 完成条件

- [ ] PC 参考后端与单板参考 CPU 后端通过冒烟集。
- [ ] 准确率集达到已记录的参考指标。
- [ ] 同一输入连续执行 100 次输出稳定。

---

## P2：安卓、鸿蒙和 NPU 后端适配

### 2.1 统一后端接口

- [ ] 定义 `prepare(model)`。
- [ ] 定义 `set_input(tensor)`。
- [ ] 定义 `invoke()`。
- [ ] 定义 `get_output()`。
- [ ] 定义 `get_backend_info()`。
- [ ] 定义 `get_partition_info()`。
- [ ] 定义 `get_profiling_info()`。
- [ ] 定义 `destroy()` 并验证反复初始化无泄漏。

### 2.2 后端矩阵

- [ ] 实现参考 CPU 后端。
- [ ] 实现安卓框架 NPU 后端。
- [ ] 实现鸿蒙单框架 NPU 后端。
- [ ] 实现安卓、鸿蒙双框架量产环境中的对应调用路径。
- [ ] 对每个后端记录运行时、编译器、驱动和固件版本。
- [ ] 记录模型实际分区、NPU 节点数及 CPU/GPU 回退节点数。
- [ ] 后端发生回退时将本次结果标为 invalid 或 mixed，不计入纯 NPU 结果。

### 2.3 输入传输

- [ ] 将原 UART `db load` 语义替换为适合单板的文件、共享内存或 RPC 输入路径。
- [ ] 推理计时前完成模型加载和输入准备。
- [ ] 增加只测 NPU 执行以及端到端执行两个计时范围。
- [ ] 检查零拷贝、缓存同步和内存对齐是否真实生效。

### P2 完成条件

- [ ] 四个负载在所有目标框架上通过冒烟集。
- [ ] 能够证明执行设备和算子分区。
- [ ] 双框架使用完全相同的模型、输入和输出判定。

---

## P3：测量与实验环境校准

- [ ] 定义统一单调时钟和时间单位。
- [ ] 分离模型加载、编译、首次推理、热推理、输入复制和后处理时间。
- [ ] 建立外部整机功耗测量接口。
- [ ] 若 BSP 提供 NPU/SoC 电源轨数据，接入内部轨级测量。
- [ ] 校准外部仪器时间与软件事件时间。
- [ ] 记录板卡、SoC、内存、固件、系统镜像和电源配置。
- [ ] 固定后台服务、网络、显示、日志等级和调试接口状态。
- [ ] 记录初始温度、结束温度和完整 thermal status 时间序列。
- [ ] 测量 idle 基线，并同时报告总系统能量和扣除基线后的增量能量。
- [ ] 每个配置至少重复多轮，报告中位数、P90、P99 和离散程度。

### P3 完成条件

- [ ] 空载功率在规定窗口内稳定。
- [ ] 相同测试重复执行的能量与延迟波动满足预设阈值。
- [ ] 软件计时和外部仪器事件可对齐。

---

## P4：AVS/DVFS 实验

- [ ] 列出 NPU 支持的 OPP、频率和可控电压范围。
- [ ] 确认电压/频率控制来自固件、驱动、Power HAL 还是测试接口。
- [ ] 区分固定 OPP、量产动态策略和实验 AVS 策略。
- [ ] 每个测试点记录请求值与实际观测值。
- [ ] 扫描过程中持续验证输出正确性，不能只测速度和功率。
- [ ] 记录超时、驱动复位、ECC/校验错误和输出偏差。
- [ ] 采用随机化或往返顺序扫描 OPP，降低温度随时间上升造成的偏差。
- [ ] 冷启动测试之间恢复到规定温度。
- [ ] 持续测试单独报告温控前与稳态区间。

建议输出指标：

- Joule/inference。
- Joule/有效检测。
- 平均功率和峰值功率。
- P50/P90/P99 延迟。
- 吞吐量。
- NPU 利用率、频率和电压时间序列。
- 首次推理与热推理延迟。
- 准确率、FP/FN 或输出误差。
- 温度、thermal status 和发生降频的时间点。

### P4 完成条件

- [ ] 至少完成一个负载的完整 OPP/AVS 扫描。
- [ ] 能区分 NPU 本体能量、运行时开销和整机能量。
- [ ] 结果中不存在未识别的 CPU/GPU 回退。
- [ ] 找到性能、能量和稳定性之间的 Pareto 点。

---

## P5：报告、回归和发布

- [ ] 定义机器可解析的 JSON 结果格式。
- [ ] 结果包含配置、版本、哈希、环境、原始样本和汇总指标。
- [ ] 保存原始功率和温度时间序列，不只保存汇总数字。
- [ ] 自动生成安卓、鸿蒙和双框架对比表。
- [ ] 自动生成电压—频率—延迟—能量曲线。
- [ ] 每次修改模型、编译器、驱动或固件后运行冒烟回归。
- [ ] 标记 incompatible 的模型或测试版本，避免跨版本误比较。
- [ ] 文档中明确区分 inference-only、runtime 和 system-level 结果。

---

# 附录 A：建议研究场景

## A.1 NPU 启动与调度开销

目的：确定小模型使用 NPU 是否比 CPU 更节能。

- 使用当前 32×32 图像分类和 KWS 小模型。
- 测量模型编译、首次推理、热推理和输入复制。
- 对比 CPU、NPU 以及可用的其他硬件后端。
- 分别测试单次、连续 10 次、100 次和 1000 次。
- 输出 NPU 相对 CPU 的延迟与能量交叉点。

## A.2 Race to Idle

目的：比较高频快速完成与低频长时间执行。

- 选择 KWS 和 AD 小负载。
- 固定每秒请求数量。
- 扫描不同 OPP。
- 记录活跃时间、空闲时间和周期总能量。
- 比较最低频点、最高频点和量产动态策略。

## A.3 中型持续推理

目的：建立稳定占用 NPU 时的 AVS 能效曲线。

- 在 IC01 之外增加 96×96 或 224×224 的 MobileNet 类模型。
- 连续运行 10～30 分钟。
- 输出温控前、温控转换期和稳态三个阶段。
- 比较固定 OPP 与量产 AVS/DVFS 策略。

## A.4 周期实时负载

目的：模拟摄像头、传感器或语音帧周期触发。

- 设置 10 ms、20 ms、32 ms、100 ms 和 1 s 周期。
- 请求超期时记录 deadline miss。
- 统计周期能量、唤醒延迟和可用空闲时间。
- 研究性能提示或框架 QoS 对 AVS 策略的影响。

## A.5 流式唤醒词系统场景

目的：评估持续监听的整机低功耗能力。

- 使用真实 WAV 播放或稳定的数字音频注入。
- 统计 FP、FN、检测延迟、占空比和平均功率。
- 分别测 inference-only 与完整音频链路。
- 完整链路结果包含音频输入、特征提取、CPU/DSP、NPU 和内存功耗。
- 比较屏幕关闭、不同系统框架和不同音频路由。

## A.6 冷启动与热启动

目的：分析模型缓存、编译缓存和 NPU 电源门控。

- 冷启动前清除运行时缓存并确保 NPU 回到空闲状态。
- 热启动复用已编译模型和已分配张量。
- 分别报告 prepare、first invoke 和 steady invoke。
- 检查安卓与鸿蒙是否采用不同缓存策略。

## A.7 安卓、鸿蒙单框架和双框架对比

目的：分离框架开销和 NPU 硬件能力。

- 使用同一模型文件、同一输入顺序和同一结果判定。
- 固定固件、驱动、OPP 和温度条件。
- 记录各框架的模型转换、内存复制、调度和回退情况。
- 双框架环境增加同时运行、先后运行和资源竞争测试。
- 结果按“框架路径”分组，避免仅按系统名称归类。

## A.8 多模型和并发场景

目的：研究量产手机同时使用多个 AI 功能时的调度与功耗。

- IC 与 KWS 交替执行。
- KWS 持续运行时突发执行 IC。
- 两个框架分别持有模型并竞争 NPU。
- 测量上下文切换、队列等待、峰值功率和尾延迟。
- 检查高优先级实时任务是否出现 deadline miss。

## A.9 数据相关功耗

目的：避免用单个全零输入低估功耗。

- 比较全零、随机、真实低激活和真实高激活输入。
- 固定模型和 OPP。
- 观察激活稀疏度、内存访问和功耗差异。
- 正式 AVS 测试使用固定、多样且有哈希记录的样本集合。

## A.10 稳定性与电压裕量

目的：在允许控制电压的工程环境中验证低压可靠性。

- 每个电压点运行长时间正确性测试。
- 持续比较输出与 golden 结果。
- 记录偶发错误、超时、驱动复位和系统重启。
- 不把一次成功推理当作电压点稳定的依据。
- 最终报告安全工作区和保护裕量。

---

# 附录 B：实验结果最小元数据

每次运行至少保存：

```json
{
  "board": "",
  "soc": "",
  "firmware": "",
  "os_framework": "android|harmony|dual",
  "runtime": "",
  "driver": "",
  "npu_firmware": "",
  "workload": "",
  "model_sha256": "",
  "dataset_sha256": "",
  "backend": "",
  "partition_info": {},
  "precision": "int8|int16|fp16",
  "scenario": "cold|warm|sustained|periodic|streaming",
  "requested_voltage": null,
  "requested_frequency": null,
  "observed_voltage": null,
  "observed_frequency": null,
  "initial_temperature": null,
  "ambient_temperature": null,
  "idle_power": null,
  "result_file": "",
  "raw_trace_file": ""
}
```

# 附录 C：近期优先执行的最小闭环

1. [ ] 从固定上游版本恢复三份测试 YAML。
2. [ ] 先恢复 KWS 的模型、12 类冒烟输入和 golden 输出。
3. [ ] 在 PC 参考后端验证 KWS。
4. [ ] 在 UDP 单板参考 CPU 后端验证 KWS。
5. [ ] 接入鸿蒙 NPU 后端并确认无回退。
6. [ ] 接入安卓 NPU 后端并确认无回退。
7. [ ] 完成 KWS 的 cold、warm、periodic 和 Race-to-Idle 测试。
8. [ ] 验证测量方法后，再依次恢复 IC、AD 和 SWW。

选择 KWS 作为首个闭环，是因为数据规模较小、类别和输入尺寸明确，同时能覆盖量化、周期唤醒和低功耗场景。图像分类可作为第二个负载，用来补充更高 NPU 利用率的持续运行测试。
