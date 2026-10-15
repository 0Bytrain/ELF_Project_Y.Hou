# 极低频跨冰通信：Polar + 四变量张量 DQN

本目录是独立重写版。决策变量是频点、调制、Polar 码率、符号率；依据变化的环境决策，没有通信模式，也没有 HC-full、SA-full、Beam 或 imnet。入口为 `run.py`。

## 运行

Python 3.10 或更高版本（3.9–3.13 均可运行；本机使用 Miniconda Python 3.13 验证）。依赖见 `requirements.txt`；NumPy 1.24 及以上均可（1.x/2.x 兼容，见 `elfcomm/channel.py`）。CPU 即可，训练设备可在配置中设为 `cuda`。

```powershell
cd <run.py 所在目录>
python -m pip install -r requirements.txt
python run.py demo
```

现在可以用一条命令完成“读取占用频段 → 建立/读取 BER 表 → DQN 训练 → 四变量迭代曲线 → 最终决策”：

```powershell
python run.py run --scenario configs/scenario_occupied.json
```

运行前只需修改 `configs/scenario_occupied.json` 中的 `occupied_bands`。结果保存在 `outputs/`，其中 `decision_variables_vs_episode.png` 显示频点、调制、Polar 码率和符号率随训练轮数的贪心决策变化，`training_history.csv` 保存对应数据。快速测试可加 `--quick`，但只能用于检查程序流程。

`demo` 完整运行小规模误码表、DQN 训练、决策及独立误码验证，使用 **N=64、L=2、极少样本**，仅验证程序可运行，不能作为论文可靠性证据。产物在 `outputs/quick/`。正式配置是 **N=256、L=8**：

```powershell
python run.py build-table
python run.py train
python run.py decide
python run.py validate
python run.py sensitivity
```

误码表首次构建需要逐帧真实译码；进度逐点打印，中断后重新运行可从已完成的点继续。DQN 每一步查询表，不重复运行 Polar 译码。训练每个评估周期保存历史；正式默认 500 个 episode，每个 24 步。`sensitivity` 在参考 SNR 为 5/10/15/20 dB 时评估同一策略，不重新训练。

如果 `python` 不在 PATH（Windows 常见），用解释器完整路径替换上面的 `python`。例如本机使用 Miniconda：

```powershell
& 'D:\Miniconda\Miniconda3\python.exe' run.py demo
```

如果改变码长、列表大小、码率集合、Es/N0 网格、仿真样本预算或编译码源码，需要另建误码表。程序用配置与源码哈希检查兼容性，不会静默读取旧表：

```powershell
python run.py build-table --max-frames 1024 --table outputs/table_1024.json
python run.py train --config outputs/table_config.json
```

`table_config.json` 保存建表时的全部配置。`run_config.json` 保存训练时的全部配置。改变动作顺序、Polar 参数、状态归一化范围后不能直接加载原模型，需要重新训练。

## 人工修改环境的位置

所有默认参数集中在 `configs/default.json`，不需编辑算法源代码。

| 配置路径 | 用途 |
|---|---|
| `link.reference_snr_db` | 相对标定基准，默认 20 dB |
| `link.receiver_noise_rms_pt` | **接收机**在参考带宽内的 RMS 噪声，默认 0.1 pT |
| `link.reference_noise_bandwidth_hz` | 上述噪声对应的参考等效带宽，暂定 1 Hz |
| `link.reference_environment` | 标定环境及默认手动决策环境 |
| `environment.ranges` | 训练时环境随机采样范围 |
| `environment.occupied_bands` | 人工指定的占用频带列表 |
| `environment.random_occupancy` | 训练时是否额外随机生成占用 |
| `link.guard_band_hz` | 硬占用区间两侧保护带，默认各 0.5 Hz |
| `objective.target_ber` | 目标 BER，默认 1e-4 |
| `objective.target_goodput_bps` | 有效吞吐量优化目标，默认 1 bit/s |

`configs/scenario_occupied.json` 给出可直接修改的独立场景。频段用区间表示：

```json
{"low_hz": 4.0, "high_hz": 8.0, "mode": "hard"}
```

该区间加上保护带后，只要与待发射波形占用带宽重叠，整个动作就被屏蔽；不是只检查载频中心。软占用格式：

```json
{"low_hz": 16.0, "high_hz": 20.0, "mode": "soft", "interference_asd_pt_rt_hz": 1.0}
```

软占用按实际重叠带宽叠加不相关干扰功率谱。保护带之外没有“距离占用频点越远奖励越高”的人为项。所有频点不可用时返回停发，BER 为 `null`，并统计 outage；不能以停发制造 BER=0。

```powershell
python run.py decide --scenario configs/scenario_occupied.json
python run.py decide --noise-asd 0.3 --depth 60 --distance 230
python run.py decide --ice-thickness 1.8 --ice-temperature -15 --water-conductivity 4.2
python run.py decide --quick --scenario configs/scenario_occupied.json
```

`--noise-asd` 是接收位置处的**环境磁场噪声幅度谱密度**，单位 pT/sqrt(Hz)，与固定的接收机 0.1 pT RMS 分开。默认环境范围 1–8 pT/sqrt(Hz) 是用于策略敏感性训练的假设扫描范围（2026-10 按 42 dB 参考锚重新标定；仍非现场测量统计，待实测/ITU 数据锚定）。

## 约定的系统参数

- 频点：5、7、…、29 Hz，共 13 个；500 kW 作为固定发射工况元数据，不参与动作或绝对场强换算。
- 调制：BPSK、Gray QPSK、MSK；Polar 名义码率 1/3、1/2、2/3、3/4。
- Polar：N=256，CRC-11，CA-SCL 列表 8，PW 冻结位构造。`K=round(N*rate)` 包含 CRC，载荷长度为 `K-11`，因此实际码率可能略偏离名义值；这是独立 Polar 实现，不声称 5G 标准兼容。
- 符号率：0.25、0.5、2/3 sym/s。动作张量形状为 13×3×4×3，共 **468 个动作**。
- 标定环境：水平 100 km、冰水界面下 50 m、水深 100 m、冰厚 1.5 m、冰温 -10°C、海水电导率 4 S/m。
- 训练范围：冰厚 1–2 m，冰温 -25 至 -2°C，海水电导率 2.5–5 S/m，深度 30–70 m，水平距离 80–120 km。每个 episode 重采样，之后作有界缓慢随机变化。步长按决策步建模，不是实测环境时间过程。
- 波形占用带宽上限 1 Hz。PSK 用 `(1+0.35)*Rs`；MSK 用首零点主瓣宽度 `1.5*Rs`。MSK 有旁瓣，保护带规则是简化的频谱接入约束，不等于严格的全频带零泄漏保证。

**1 bit/s 目前是优化目标，不是可保证的硬约束。** N256、名义码率 3/4、QPSK、Rs=2/3 时，载荷为 181 bit；加入 16 个前导符号后，无误帧的最大有效吞吐量为 `181/216 ≈ 0.838 bit/s`。程序保留低于目标的结果，不篡改符号率或忽略开销来满足指标。

## 相对磁场链路及噪声

参考接收机噪声 PSD 为 `0.1^2/1 = 0.01 pT²/Hz`。参考环境环境噪声设为 0，42 dB 基准对应假设的接收信号 RMS 约为 12.6 pT（由 ITU 场强与海水衰减折算的模型假设值，对应 30 Hz、100 km 参考点）。**口径说明（2026-10）：42 dB 仅作为信号电平的编码口径；项目链路信噪比一律按"含环境噪声"口径表述——实际工作信噪比取决于环境噪声取值，约 5–15 dB 量级。**

```text
B_rx(f,e) = B_rx_ref * |H_B(f,e) / H_B(30 Hz,e_ref)|
noise_power = effective_noise_PSD * occupied_bandwidth
SNR = signal_mean_square / noise_power
Es/N0 = signal_mean_square / (effective_noise_PSD * symbol_rate)
```

只有初始参考场标定一次，不对每个新环境重新归一化。传感器 PSD、环境 PSD、软占用 PSD 按不相关功率相加。环境 PSD 暂按 `(f/5Hz)^(-1)` 变化，可配置；这是合成噪声模型，不声称复现 ITU 噪声表或海洋实测谱。

`H_B` 采用空气—有限冰层—海水的法向入射稳定多反射公式，然后在海水中传播至接收深度。先求电场传输，再按两侧波阻抗转换为**磁场**传输；不会直接把电场系数或 dBm 与 pT 相减。海床默认不参与反射。水平距离通过 `(d_ref/r)^p` 修正（`d_ref` 为参考点距离 100 km），默认 `p=1`；这是显式的相对几何假设，不是发射天线的近场解。

该模块参考 `E:/LFCommSystem/PolarCrossIceELF` 的 Debye、传播常数、层间反射约定重新用 Python 实现，无 MATLAB 运行依赖。没有采用 ELF_PINN 原来的长距离波导参数来代替本项目 200 m 场景。

冰参数沿用低盐/纯冰近似：静态介电常数 91、高频极限 3.2、DC 电导率 1e-5 S/m。温度通过文献 Arrhenius 弛豫时间 `tau=5.3e-16*exp(0.571/(k_B*T))` 进入 Debye 模型。在本 ELF 范围、当前电导率下温度效应非常弱。未加入虚构温度惩罚；含盐海冰温盐耦合需要实测材料参数以后替换。

来源：

- [Arrhenius/Debye 冰介电模型及参数，公式 3、6](https://doi.org/10.1051/0004-6361/201628304)
- 本地参考模块：`E:/LFCommSystem/PolarCrossIceELF/README.md`、`icePermittivity.m`、`layeredMediumTMM.m`。

这种版本研究的是**指定相对标定和模型假设下的策略性能**。源频谱暂定平坦，没有完成 500 kW 到初始场强的天线换算。不能将 1 pT 或最终 BER 写成真实发射台的绝对预测。

## BER、性能表与验证

每个误码仿真点运行真实的随机载荷 → CRC → Polar 编码 → 调制 → 噪声 → 软解调 → CA-SCL → 载荷逐位比较。BPSK/QPSK 使用理想相干符号级等效；MSK 使用连续相位波形及相应软判决。默认假设理想同步和已知平坦等效信道。信道的频带内起伏会输出诊断值，但本版没有完整的 RF 同步、导频信道估计和频率选择性均衡模块。

原始表保存 `bit_errors/total_bits` 和 `frame_errors/total_frames`。训练使用半计数平滑、单调拟合与对数插值的 BER/FER **估计值**，零误码样本仍有非零估计和抽样不确定性。插值值不是新生成的误码样本；超表范围会明确标记，并采用保守端点/低 SNR 回退，不做无依据的低 BER 外推。

Polar 译码错误在同一码字内相关，因此不能把全部比特简单当成独立伯努利试验来声称严格的置信区间。`validate` 使用独立固定帧数验证，采用独立帧的 FER 单侧 95% 上界作为保守 BER 上界（每帧误码比例 ≤ 该帧是否出错）。默认 5000 帧即使零误帧，保守上界也约为 6e-4，仍不能认证 1e-4；增加样本预算：

```powershell
python run.py validate --max-frames 40000
```

零误帧约需 29956 个独立帧，FER 单侧 95% 上界才可降至 1e-4。这只是固定仿真工况下的统计结论，不是实测工程保证。独立验证采用未参与建表的随机种子。

## 张量方法与 DQN

这里借鉴张量表示思想，使用状态相关 CP 分解：

```text
Q(s,f,m,c,r) = V(s) + sum_j w_j(s) F[f,j] M[m,j] C[c,j] R[r,j]
```

四个因子轴对应四个离散决策，状态网络产生每个低秩分量的权重。采用 masked Double DQN、经验回放、目标网络软更新、梯度裁剪。非法动作在探索、贪心选择和目标 Q 计算中全部屏蔽。

这不是把论文的信号张量直接当作决策张量，也不是复现论文的 ALS 半盲接收机；不将该自定义 CP-DQN 实现称为经过文献验证的 TEME 算法。状态包含环境量、各频点信噪比、频带占用、各频率/调制/符号率的合法性与干扰，以及上一动作。

训练奖励由有效吞吐量、BER 超标程度、速率不足及切换开销组成，权重是公开的任务偏好，不作为传播物理系数。可靠性在训练中是罚项；推荐结果会明确标记是否估计达标、是否独立验证。不会因奖励高就宣称 BER 已得到保证。

## 输出与文件位置

| 文件 | 内容 |
|---|---|
| `elfcomm/channel.py` | 相对磁场、噪声及频谱占用 |
| `elfcomm/polar.py` | Polar、CRC-11、CA-SCL |
| `elfcomm/modem.py` | 调制和软解调 |
| `elfcomm/ber.py` | 逐位误码统计与置信度 |
| `elfcomm/performance.py` | 建表、插值、吞吐量和奖励 |
| `elfcomm/environment.py` | 动态环境和动作掩码 |
| `elfcomm/dqn.py` | 四维 CP 张量 Q 网络及 Double DQN |
| `elfcomm/training.py` | 训练、固定验证环境、模型保存 |
| `outputs/training_convergence.png` | DQN 奖励、损失、吞吐量及目标满足比例 |
| `outputs/ber_vs_episode.png` | DQN BER 估计随训练轮数变化及停发比例 |
| `outputs/training_history.csv` | 含探索的每轮原始指标 |
| `outputs/validation_history.csv` | 固定种子环境下当前贪心策略指标 |
| `outputs/best_model.pt` / `last_model.pt` | 最佳验证策略及最后策略 |
| `outputs/decision.json` | 人工环境下四变量推荐与达标状态 |
| `outputs/validation_report.json` | 独立逐比特验证及可靠性状态 |

EMA 是指数移动平均，仅用于画图；原始 CSV 保留。图中的 greedy validation 指不探索的策略评估，与训练探索曲线不同，但都属于 DQN；没有加入其他算法比较。最佳 checkpoint 选择不会把训练曲线替换成历史最大值。

运行测试：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest
```

禁用自动插件是为兼容本机已有 Python 环境，不影响项目自带测试。
