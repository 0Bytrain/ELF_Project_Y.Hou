# 跨海冰低频通信：原始 DQN 学习曲线与部署效果

本轮只修改代码和配置，没有运行建表、训练或测试。

运行 `run_shift_experiment.py`，默认依次完成三个控制变量实验；每次完整训练 1600 轮，
第 1000 轮后只改变对应的一项环境参数，其他条件固定。当前不再提前冻结或提前停止训练。

| 配置 | 唯一变化 |
|---|---|
| `configs/shift_noise.json` | 环境噪声幅度谱密度 0.30 → 0.45 pT/√Hz |
| `configs/shift_distance.json` | 水平距离 1950 → 3150 km，配置使用 m |
| `configs/shift_depth.json` | 接收深度 60 → 95 m |

电导率在三个实验中固定为 4.2 S/m；电导率变化实验已删除。
训练轮内随机漂移和随机占用关闭，频点硬占用、发射带宽等合法性约束始终有效。
`configs/shift_experiment.json` 是噪声实验的兼容配置。

## 训练与部署分别处理

最终目标仍是 BER 估计值 ≤ 10⁻⁵，在达标配置中优先降低功率。
`training.ber_feasibility_filter=false`：训练及原始贪心评估允许尝试 BER 超限动作，
通过奖励惩罚学习避开；原始评估不使用切换控制，也不会先把超限动作筛掉。
`training.convergence.enabled=false`：诊断稳定性仍会计算，但不会停止模型更新或提前结束。
探索率在每阶段逐步降到 0，后段继续更新网络并观察纯策略行为。

`decision.ber_feasibility_filter=true`：实际部署单独筛除估计 BER 超限配置，并保留切换控制；
新配置不可行时停发，不能用部署结果冒充原始 DQN 收敛。
原始策略评估会额外计算模型最低可行功率作为比较，但该比较不参与原始动作筛选。
因此训练 BER 因探索而超限，并不等于最终部署策略未达标；应分别查看两个结果。

## 直接使用已有误码表

现在直接读取 `outputs/plan_ber/ber_table_cc.json`，不再进行 99 个点的细化。
`ber_table.existing_only=true` 禁止训练入口自动新建、续建或细化误码表；
若文件缺失、不完整或参数不兼容则明确报错，不自动启动长时间计算。
只放行已识别的卷积码奇偶位计数等价版本，其他编译码、调制、采样参数变化仍会被拦截。
原始计数文件和已经完成的细化进度均保留，不修改或删除。

运行入口仍为 `run_shift_experiment.py`。如果旧进程还在计算细化点，请先停止后重新运行，
已启动的进程不会因为配置文件变更自动改为读取旧表。
进入训练前会打印旧表路径与采样点数，随后显示 1/1600 等训练轮数。

最终 BER 目标、原始探索曲线、部署保护和三个控制变量实验保持不变。
旧表中零误码是有限样本观察，不代表真实 BER=0；直接复用不会提高其在 10⁻⁵ 附近的分辨能力。
空心三角的 1/N 位置仅用于显示，不参与奖励、平滑或达标判断，可靠性认证状态仍保留。

## 新输出目录和图

新结果集中保存到 `outputs/ber1e5_learning/`；之前的平直曲线和其他旧结果保留。
根目录 `noise.png`、`distance.png`、`depth.png` 均改为对应实验的原始训练 BER、EMA 和原始策略功率。
每个同名子目录还有：

| 文件 | 含义 |
|---|---|
| `training_ber_curve.png` / `ber_vs_episode.png` | 训练 BER—轮数曲线（含最终稳定 BER/SNR 标注） / 原始训练探索 BER + EMA 及策略功率 |
| `policy_evaluation.png` | 固定环境、关闭探索、未做 BER 筛选的原始策略评估 |
| `deployment_performance.png` | 部署筛选和切换控制后的 BER 与执行功率 |
| `decision_variables_vs_episode.png` | 原始 DQN 的频点、波形、功率和编码配置迭代 |
| `raw_decision_variables_vs_episode.png` | 上一项的兼容文件名 |
| `deployment_decision_variables_vs_episode.png` | 带切换控制的执行配置迭代 |
| `training_history.csv` / `training_smoothed.csv` | 原始训练数据及仅供显示的 EMA |
| `validation_history.csv` / `validation_smoothed.csv` | 原始策略评估及显示 EMA |
| `deployment_history.csv` | 独立部署结果、原始候选 BER、筛选是否改变候选 |

平滑在环境变化前后分别计算，原始数据完整保留。离散配置不求移动平均，不添加随机扰动来制造波动。
曲线是否下降、是否振荡以及恢复需要多久取决于实际学习过程；不能承诺一定出现某种形状。

以下保留历史实现说明；其中旧目标、旧目录和旧实验数值以本节及当前JSON配置为准。

本目录是独立重写版。决策变量是**频点、发射功率、波形模式（调制 + 符号率）、编码模式（编码方式 + 编码率）**；依据变化的环境决策，没有通信模式，也没有 HC-full、SA-full、Beam 或 imnet。入口为 `run.py`。

当前已实现 **Polar、CC（卷积码）、BCH（扩展 BCH）** 三种编码方式；LDPC/Turbo/RS 的分阶段方案见 `学习笔记/18_编码方式维度扩展_2026-10-23.md`，尚未进入动作空间，不能把它们的 BER 说成已实现。

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

运行前只需修改 `configs/scenario_occupied.json` 中的 `occupied_bands`。结果保存在 `outputs/`，其中 `decision_variables_vs_episode.png` 显示频点、波形模式、功率和有效载荷率随训练轮数的贪心决策变化，`training_history.csv` 保存对应数据。快速测试可加 `--quick`，但只能用于检查程序流程。

`demo` 完整运行小规模误码表、DQN 训练、决策及独立误码验证，使用 **N=64、L=2、极少样本** 和 **10×2×9×7** 的 quick 动作网格（state 221，quick 只启用 Polar/CC；BCH 需要 255 位基码，quick 不启用）；quick 的 Polar payload 为 10/64、21/64、32/64、37/64，CC payload 为 15/64、26/64、31/64。它仅验证程序可运行，不能作为论文可靠性证据。产物在 `outputs/plan_demo/`（可用 `--output` 改目录）。正式配置是 **N=256、L=8**、**54×14×9×11** 动作网格（Polar 4 + CC 3 + BCH 4）：

```powershell
python run.py build-table
python run.py train
python run.py decide
python run.py validate
python run.py sensitivity
```

误码表首次构建需要逐帧真实译码；进度逐点打印，中断后重新运行可从已完成的点继续。DQN 每一步查询表，不重复运行编码/译码。训练每个评估周期保存历史；正式默认 1600 个 episode，每个 24 步。`sensitivity` 对参考信号整体缩放 `−6/−3/0/+3/+6 dB`（同时缩放锚点，保持锚点自检一致），评估同一策略，不重新训练。

建表日志里的 `871/5120 bit errors` 是**蒙特卡洛实测的误码数/总比特数**，不是程序报错；`0/40960 bit errors` 表示 40,960 bit 中零误码。低 SNR 出现大量 bit errors 是正常统计结果，程序会继续建下一个点。

中途信道切换实验会先检查新编码轴的 BER 表；表缺失或未建完时自动新建/续建，然后训练至多 1600 个 episode：前 1000 轮使用 `configs/shift_experiment.json` 的 `fixed_environment`，第 1000 轮之后（episode 1001 起）切到 `channel_shift.environment`，随后继续训练并画出恢复曲线：

```powershell
python run_shift_experiment.py
```

也可以先单独建表，便于观察建表进度：

```powershell
python run.py build-table --config configs/shift_experiment.json
```

产物写入 `outputs/ber1e5_learning/<实验名>/`，包括 `training_history.csv` 和 `training_ber_curve.png`。运行时看 `tqdm` 进度条；不要同时启动两个训练进程写同一目录。当前动作空间为 **54×14×9×11 = 74,844**（Polar 4 + CC 3 + BCH 4）。

2026-10-24 的 800 轮诊断见 `学习笔记/20_信道切换800轮结果诊断与修改报告_2026-10-24.md`；旧的 500 轮结果目录已删除。当前代码已按报告 P0/P1 修改（固定验证的 varied occupancy/CI、target_ber 奖励、replay reset、对称判据等），但 pre 工作点仍落在 BER 表 floor，严格 ±15% 结论必须等新的可测工作点和直接仿真验证。

旧的 `outputs/plan_ber/ber_table.json` 已删除；当前代码统一使用 `outputs/plan_ber/ber_table_cc.json`。`outputs/plan_demo/` 仍用于 quick 自检，不作为正式可靠性证据。

如果 `python` 不在 PATH（Windows 常见），用解释器完整路径替换上面的 `python`。例如本机使用 Miniconda：

```powershell
& 'D:\Miniconda\Miniconda3\python.exe' run.py demo
```

如果改变码长、列表大小、码率集合、Es/N0 网格、仿真样本预算或编译码源码，需要另建误码表。程序用配置与源码哈希检查兼容性，不会静默读取旧表：

```powershell
python run.py build-table --max-frames 1024 --table outputs/table_1024.json
python run.py train --config outputs/table_config.json
```

`table_config.json` 保存建表时的全部配置。`run_config.json` 保存训练时的全部配置。改变动作顺序、编码模式/编码参数、Polar 参数或状态归一化范围后不能直接加载原模型，需要重新训练。

## 人工修改环境的位置

所有默认参数集中在 `configs/default.json`，不需编辑算法源代码。

| 配置路径 | 用途 |
|---|---|
| `link.field_anchor_distance_m` / `link.field_anchor_e_db_uv_m` | 绝对锚点：1950 km 处 57.00 dB μV/m；必须与参考点距离一致 |
| `link.reference_signal_rms_pt` | 由锚点链派生的参考信号 RMS，默认 1.5906 pT；不是独立可调 SNR |
| `link.source_model` | `anchor`（默认）或 `current_moment`（可选电流矩源模型） |
| `link.antenna.current_moment_a_m` | 500 kW 时电流矩；刘勇论文：东西向 3e7 A·m、南北向 2e7 A·m |
| `link.antenna.input_resistance_ohm` / `radiation_resistance_ohm` | 陆院士书第 41 页 100 km 天线的 `R_in(f)`/`R_r(f)` 表；代码 log-log 插值 |
| `link.antenna.unit_convention` | `engineering`（默认，长度 km，公式结果乘 1/sqrt(1000) 转 μV/m）或 `si`（长度 m，先得 V/m 再转 μV/m）；两者已统一到同一物理场 |
| `link.receiver_noise_rms_pt` | **接收机**在参考带宽内的 RMS 噪声，默认 0.1 pT |
| `link.reference_noise_bandwidth_hz` | 上述噪声对应的参考等效带宽，暂定 1 Hz |
| `link.reference_environment` | 标定环境及默认手动决策环境（1950 km、噪声 0.3162 pT/√Hz） |
| `link.propagation_model` | `lu_attenuation`（默认，陆院士公式）或 `lwpc_table`（30 Hz 表对照） |
| `link.ionosphere_mode` | `day`（默认）或 `night`；决定 β(f) 用哪组参数 |
| `link.beta_scale` | 公式衰减率缩放，默认 1.0；仅用于显式校准，不是拟合旋钮 |
| `link.propagation_min_km` | 公式最小距离，默认 1200 km；近场不把公式外推 |
| `actions.frequency_hz` / `tx_power_kw` / `waveform_mode` / `coding_mode` | 四轴候选集合；`coding_mode=[编码方式, 编码率]`，轴顺序决定动作索引，改变后旧模型不能加载 |
| `environment.ranges` | 训练时环境随机采样范围（水平距离 1200–2950 km，噪声 0.1–1 pT/√Hz） |
| `environment.occupied_bands` | 人工指定的占用频带列表 |
| `environment.random_occupancy` | 训练时是否额外随机生成占用 |
| `environment.random_occupancy_min_hz` | 随机占用的最低频点，默认 60 Hz；低于它的频点不会被随机占用 |
| `environment.random_band_count_max` / `random_band_width_hz` | 随机占用 1–3 个中心、每个宽 1.2 Hz（硬占用） |
| `link.max_occupied_bandwidth_hz` | 单个动作允许的最大占用带宽，默认 1.2 Hz |
| `link.guard_band_hz` | 硬占用区间两侧保护带，默认各 0.5 Hz |
| `objective.target_ber` | 目标 BER，当前为 1e-5 |
| `objective.target_goodput_bps` | 有效吞吐量优化目标，默认 0.6 bit/s |

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
python run.py decide --noise-asd 0.3 --depth 60 --distance 1950000
python run.py decide --ice-thickness 1.8 --ice-temperature -15 --water-conductivity 4.2
python run.py decide --quick --scenario configs/scenario_occupied.json
```

`--noise-asd` 是接收位置处的**环境磁场噪声幅度谱密度**，单位 pT/sqrt(Hz)，与固定的接收机 0.1 pT RMS 分开。默认环境范围 **0.1–1 pT/sqrt(Hz)**，参考环境取 `0.316227766`（PSD=0.1）；这是用于策略敏感性训练的假设扫描范围，仍非现场测量统计，待实测/ITU 数据锚定。`--distance` 的单位是 **m**（例如 1950000 表示 1950 km）；公式模型只接受 ≥1200 km。

## 约定的系统参数（2026-10-23 四轴定稿）

- 频点：**54 个** = `17, 25, 33, 41, 55` Hz（低频段稀疏，避开 7.8/14/20 Hz 舒曼谐振及 50 Hz 附近）+ `60, 65, …, 300` Hz（5 Hz 步进）。相邻频点至少 5 Hz，大于 1.2 Hz 波形带宽加保护带的需求。
- 功率：5/10/15/20/25/30/40/50、62.5/100/150/200/250、500 kW，共 14 档（以 500 kW 为参考）。
- 波形模式：3 调制（BPSK、Gray QPSK、MSK）× 3 符号率（0.25、0.5、2/3 sym/s）= **9 个** `(modulation, symbol_rate)` 组合。
- 编码模式：`coding_mode = [编码方式, 编码率]`，当前 **11 档**：
  - Polar：N=256，CRC-11，CA-SCL list 8；名义码率 1/3、1/2、2/3、3/4；payload 74/117/160/181 bit。
  - CC：K=7、生成多项式 (171,133)_8、零尾、软判决 Viterbi；名义码率 1/2、2/3、3/4；payload 111/154/175 bit；2/3 和 3/4 使用确定性的均匀删除图案（保留尾部比特），不是 CCSDS 标准图案。
  - BCH：基码 (255,k)，k=91/131/171/191，扩展 1 个总校验位到 256；payload 80/120/160/180 bit；硬判决 Berlekamp–Massey。k=91 接近 1/3，但没有把 (255,87) 虚构成已实现。
- 统一 CRC-11：三种码都在信息位后带 11 bit CRC；FER 统计包含 CRC 拒绝。实际码率、payload bits、coded bits 由编码器报告，**名义码率不等于实际码率**。
- 动作张量形状：**54×14×9×11 = 74,844**。信道指标只依赖频点、功率、调制、符号率，所以 `evaluate_many` 实际只算 54×14×3×3 = **6,804** 个唯一物理组合；编码方式/编码率只影响 BER 表和有效吞吐。
- 编码模式顺序：Polar 四档 → CC 三档 → BCH 四档；`payload_rate` 是派生量，不再是独立轴。
- 状态维度：6 环境量 + 3×54 频率特征 + 54×9 合法性 + 54×9 噪声 + 5 上一动作 = **1,145**。
- 随机占用：每个 episode 从 ≥60 Hz 的候选频点中随机抽 1–3 个中心，生成 1.2 Hz 硬占用带；动作波形与占用带（含 0.5 Hz 保护带）重叠即被 mask，训练过程中按 `occupancy_change_probability` 变化。低频谱点不随机占用。
- 极化已从动作空间移除；`data/polarization/` 只保留为历史核对资料，不参与当前计算。
- 标定环境：水平 **1950 km**、冰水界面下 50 m、水深 100 m、冰厚 1.5 m、冰温 -10°C、海水电导率 4 S/m。
- 训练范围：冰厚 1–2 m，冰温 -25 至 -2°C，海水电导率 2.5–5 S/m，深度 30–70 m，水平距离 **1200–2950 km**。每个 episode 重采样，之后作有界缓慢随机变化。步长按决策步建模，不是实测环境时间过程。
- 波形占用带宽上限 1.2 Hz。PSK 用 `(1+0.35)*Rs`（最大 0.9 Hz）；MSK 用首零点主瓣宽度 `1.5*Rs`（最大 1.0 Hz）。MSK 有旁瓣，保护带规则是简化的频谱接入约束，不等于严格的全频带零泄漏保证。

**1 bit/s 目前是优化目标，不是可保证的硬约束。** Polar N256、名义码率 3/4、QPSK、Rs=2/3 时，载荷为 181 bit；加入 16 个前导符号后，无误帧的最大有效吞吐量为 `181/216 ≈ 0.838 bit/s`。CC 同码率下 payload 为 175 bit，BCH 为 180 bit，goodput 会相应更低；程序保留低于目标的结果，不篡改符号率或忽略开销来满足指标。

## 相对磁场链路及噪声

参考接收机噪声 PSD 为 `0.1^2/1 = 0.01 pT²/Hz`；参考环境噪声 `0.316227766 pT/√Hz`，PSD `0.1 pT²/Hz`，合计 `0.11 pT²/Hz`。参考点 1950 km/30 Hz/500 kW 的锚点电场 57.00 dB μV/m → `B_air=2.361 pT` → 乘局部 `H_local=0.67358` → `B_rx=1.5906 pT`，因此参考 SNR=`10log10(1.5906²/(0.11×1))=13.62 dB`。**不再有独立的 `reference_snr_db` 旋钮**；参考信号由锚点链派生。

```text
G_dB(f,d) = -20 log10(d/d_ref) - beta(f)*(d-d_ref)/1e6      # d,d_ref in m
B_rx = B_air_anchor * |H_local(f,env)| * 10^(G_dB/20) * sqrt(P/500)
noise_power = effective_noise_PSD * occupied_bandwidth
SNR = B_rx² / noise_power
Es/N0 = B_rx² / (effective_noise_PSD * symbol_rate)
```

`beta(f)` 默认用陆院士第 26 页闭式：白天 `β=8.6858e6*k0*(π/4)*sqrt(h1/h0)*(σ0/h0+σ1/h1)`，其中 `h0=H−σ0*ln(2.5e5/(2πf))`、`h1=h0+2σ0*ln(2.39e4/(f*σ0))`；夜间 `β=0.1429*f*sqrt(hE/h0)*(σ0/h0+1/(π*k0*|nE|*hE))`，`nE=sqrt(σE/(ω*ε0))`；单位 dB/Mm。默认白天 H=70 km、σ0=σ1=3.33 km；夜间 H=90 km、σ0=2.5 km、hE=90 km、σE=8e-6 S/m。`beta_scale=1.0` 表示不做曲线拟合；LWPC 表仅作 `lwpc_table` 可选模式和 30 Hz/100 km 交叉校验。

只有初始参考场标定一次，不对每个新环境重新归一化。传感器 PSD、环境 PSD、软占用 PSD 按不相关功率相加。环境 PSD 暂按 `(f/30Hz)^(-1)` 变化，可配置；这是合成噪声模型，不声称复现 ITU 噪声表或海洋实测谱。

`H_local` 采用空气—有限冰层—海水的法向入射稳定多反射公式，然后在海水中传播至接收深度。先求电场传输，再按两侧波阻抗转换为**磁场**传输；不会直接把电场系数或 dBm 与 pT 相减。海床默认不参与反射。距离不进入 `H_local`，由 `G_dB(f,d)` 单独负责；100 km 近场不套用远场公式，只用 LWPC 表核对（81.07 dB→25.41 pT）。

### 可选电流矩源模型（`source_model="current_moment"`）

默认仍是上面的 57 dB 锚点；可选模型按刘勇论文的 500 kW 电流矩 `M=3×10^7 A·m` 和陆院士书式 `2.1.75/2.1.79/2.1.80` 计算：

```text
I_a(f,P) = sqrt(P / R_in(f))
M(f,P)   = I_a(f,P) * L_eff
E_z(f,d) = sqrt(60)*pi*M / (h*lambda*sqrt(sigma*a_e*sin(d/a_e)))
           * cos(phi) * exp(-beta*d)
beta     = 1/(2*sqrt(2)*h) * (sqrt(omega*eps0/sigma_e) + sqrt(omega*eps0/sigma_i))
```

`R_in(f)` 和 `R_r(f)` 用陆院士书第 41 页 100 km 天线表格 log-log 插值；`eta=R_r/R_in`，`P_rad=eta*P`。`L_eff` 由 `M_ref` 和参考频点的 `I_a` 反推（默认约 175.9 km，是“电流矩等效长度”，不是垂直天线高度）。功率仍严格按 `I_a ∝ sqrt(P)`，所以固定频点下接收功率仍 ∝ `P/500`。

式 `2.1.80` 的绝对长度单位在原始资料里有两种口径：

```text
si:          h, lambda, a_e 用 m；M 用 A·m；先得 V/m 再转 μV/m
engineering: h, lambda, a_e 用 km；M 用 A·m；公式结果乘 1/sqrt(1000) 转 μV/m
```

两个分支现在返回同一个物理场（不再相差 `sqrt(1000)`/30 dB）。**2026-10-23 已确定长度用 km，默认 `unit_convention="engineering"`**，但**不要把它当成已验证的绝对标定**：在 1950 km/30 Hz/500 kW 下，锚点为 57.00 dB μV/m、1.5906 pT、13.62 dB；电流矩口径为 36.08 dB μV/m、0.1431 pT、−7.30 dB。默认源模型仍是 `anchor`；启用 `current_moment` 时两者都走同一场值。先运行对照：

```powershell
python run.py antenna-check --output outputs/antenna_check
python run.py demo --quick --source-model current_moment --antenna-unit engineering --output outputs/antenna_eng_smoke
```

`antenna_check.json` 同时输出两种口径的参考场、频率扫描、距离扫描、`R_in`、`R_r`、效率和辐射功率。只有在确认单位和方向因子后再把 `source_model` 默认切换；当前默认仍为 `anchor`，不会破坏已有基线。

该模块参考 `E:/LFCommSystem/PolarCrossIceELF` 的 Debye、传播常数、层间反射约定重新用 Python 实现，无 MATLAB 运行依赖。没有采用 ELF_PINN 原来的长距离波导参数来代替本项目 200 m 场景。

冰参数沿用低盐/纯冰近似：静态介电常数 91、高频极限 3.2、DC 电导率 1e-5 S/m。温度通过文献 Arrhenius 弛豫时间 `tau=5.3e-16*exp(0.571/(k_B*T))` 进入 Debye 模型。在本 ELF 范围、当前电导率下温度效应非常弱。未加入虚构温度惩罚；含盐海冰温盐耦合需要实测材料参数以后替换。

来源：

- [Arrhenius/Debye 冰介电模型及参数，公式 3、6](https://doi.org/10.1051/0004-6361/201628304)
- 本地参考模块：`E:/LFCommSystem/PolarCrossIceELF/README.md`、`icePermittivity.m`、`layeredMediumTMM.m`。

这种版本研究的是**指定锚点/公式和模型假设下的策略性能**。默认锚点把 500 kW 映射到 1950 km 的 57.00 dB μV/m，源频谱仍暂定平坦；可选 `current_moment` 模型提供电流矩/有效高度/辐射电阻换算，但尚未完成绝对标定，默认不启用。不能将 1 pT 或最终 BER 写成真实发射台的绝对预测。功率轴按线性波导精确换算 `S ∝ P/500`；若实际功放效率随功率/频率变化，需要另加硬件效率模型。

## BER、性能表与验证

每个误码仿真点运行真实的随机载荷 → CRC-11 → 编码（Polar/CC/BCH）→ 调制 → 噪声 → 软解调 → 译码 → 载荷逐位比较。Polar 用 CA-SCL，CC 用软判决 Viterbi，BCH 用硬判决 BM/Chien；BPSK/QPSK 使用理想相干符号级等效，MSK 使用连续相位波形及相应软判决。默认假设理想同步和已知平坦等效信道。信道的频带内起伏会输出诊断值，但本版没有完整的 RF 同步、导频信道估计和频率选择性均衡模块。

原始表保存 `bit_errors/total_bits` 和 `frame_errors/total_frames`。训练使用半计数平滑、单调拟合与对数插值的 BER/FER **估计值**，零误码样本仍有非零估计和抽样不确定性。插值值不是新生成的误码样本；超表范围会明确标记，并采用保守端点/低 SNR 回退，不做无依据的低 BER 外推。

编码译码错误在同一码字内相关，因此不能把全部比特简单当成独立伯努利试验来声称严格的置信区间。`validate` 使用独立固定帧数验证，采用独立帧的 FER 单侧 95% 上界作为保守 BER 上界（每帧误码比例 ≤ 该帧是否出错）。默认 5000 帧即使零误帧，保守上界也约为 6e-4，仍不能认证 1e-4；增加样本预算：

```powershell
python run.py validate --max-frames 40000
```

零误帧约需 29956 个独立帧，FER 单侧 95% 上界才可降至 1e-4。这只是固定仿真工况下的统计结论，不是实测工程保证。独立验证采用未参与建表的随机种子。

## 张量方法与 DQN

这里借鉴张量表示思想，使用状态相关 CP 分解：

```text
Q(s,f,p,w,c) = V(s) + sum_j w_j(s) F[f,j] T[p,j] W[w,j] C[c,j]
```

四个因子轴对应**频点、发射功率、波形模式、编码模式**四个离散决策，状态网络产生每个低秩分量的权重。采用 masked Double DQN、经验回放、目标网络软更新、梯度裁剪。非法动作在探索、贪心选择和目标 Q 计算中全部屏蔽。

这不是把论文的信号张量直接当作决策张量，也不是复现论文的 ALS 半盲接收机；不将该自定义 CP-DQN 实现称为经过文献验证的 TEME 算法。状态包含环境量、各频点特征、各频率/波形模式的合法性与噪声，以及上一动作。

训练奖励由有效吞吐量、BER 超标程度、速率不足及切换开销组成，权重是公开的任务偏好，不作为传播物理系数。可靠性在训练中是罚项；推荐结果会明确标记是否估计达标、是否独立验证。不会因奖励高就宣称 BER 已得到保证。

## 输出与文件位置

| 文件 | 内容 |
|---|---|
| `elfcomm/antenna.py` | 可选电流矩源模型：`R_in/R_r` 插值、`I_a/M`、式 2.1.80 两种单位口径 |
| `elfcomm/propagation.py` | 陆院士衰减率闭式 `β(f)`、距离增益、参数校验 |
| `elfcomm/range_model.py` | LWPC 30 Hz 表（可选模式/交叉校验），参考点归一化 |
| `elfcomm/channel.py` | 相对磁场、传播增益、噪声及频谱占用；不含极化轴 |
| `elfcomm/polar.py` | Polar、CRC-11、CA-SCL |
| `elfcomm/codecs.py` | 编码方式注册表、统一 CRC-11、CC 编码器/软 Viterbi |
| `elfcomm/modem.py` | 调制和软解调 |
| `elfcomm/ber.py` | 按编码方式分派的逐位误码统计与置信度 |
| `elfcomm/performance.py` | 建表、插值、吞吐量和奖励 |
| `elfcomm/environment.py` | 动态环境和动作掩码 |
| `elfcomm/dqn.py` | 四维 CP 张量 Q 网络及 Double DQN |
| `elfcomm/training.py` | 训练、固定验证环境、模型保存 |
| `outputs/<run>/training_convergence.png` | DQN 奖励、损失、吞吐量及目标满足比例 |
| `outputs/<run>/ber_vs_episode.png` | DQN BER 估计随训练轮数变化及停发比例 |
| `outputs/<run>/training_history.csv` | 含探索的每轮原始指标 |
| `outputs/<run>/validation_history.csv` | 固定种子环境下当前贪心策略指标 |
| `outputs/<run>/best_model.pt` / `last_model.pt` | 最佳验证策略及最后策略 |
| `outputs/<run>/decision.json` | 人工环境下四轴推荐与达标状态（含 `waveform_mode`/`coding_mode`） |
| `outputs/<run>/validation_report.json` | 独立逐比特验证及可靠性状态 |
| `outputs/<run>/training_ber_curve.png` | 训练 BER—轮数曲线（含最终稳定 BER/SNR 标注） |
| `outputs/<run>/shift_summary.json` | 切换前后稳定 BER、SNR/Es/N0 及尾窗统计 |

EMA 是指数移动平均，仅用于画图；原始 CSV 保留。图中的 greedy validation 指不探索的策略评估，与训练探索曲线不同，但都属于 DQN；没有加入其他算法比较。最佳 checkpoint 选择不会把训练曲线替换成历史最大值。

运行测试：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
python -m pytest
```

禁用自动插件是为兼容本机已有 Python 环境，不影响项目自带测试。当前 Polar/CC/BCH 四轴版本在 2026-10-24 通过 **104 项测试**（含 CC 标准抽头/终止状态、BCH 扩展校验、表键、固定验证、随机占用和场景范围回归）。
