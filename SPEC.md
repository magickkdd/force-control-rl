# SPEC.md — 变柔顺控制销孔装配项目技术规格

> 基于论文 *Variable Compliance Control for Robotic Peg-in-Hole Assembly: A Deep-Reinforcement-Learning Approach* (Applied Sciences, 2020)
>
> 项目阶段：Phase 1a — MuJoCo 仿真力控基础（当前）→ Phase 1b 真机力控 → Phase 2 RL仿真 → Phase 3 完整复现+Sim2Real → Phase 4 扩展+论文
>
> 硬件平台：UR20 机械臂 + 六维力传感器（型号待定）
>
> 更新日期：2026-07-11

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 仿真环境规格](#3-仿真环境规格)
- [4. 力控算法规格](#4-力控算法规格)
- [5. RL 算法规格 (Phase 2+)](#5-rl-算法规格-phase-2)
- [6. 网络架构规格 (Phase 3)](#6-网络架构规格-phase-3)
- [7. Sim2Real 规格 (Phase 3)](#7-sim2real-规格-phase-3)
- [8. 接口定义](#8-接口定义)
- [9. 代码规范](#9-代码规范)
- [10. 附录](#10-附录)

---

## 1. 项目概述

### 1.1 项目目标

本项目的最终目标是在位置控制型机械臂（UR20）上，通过深度强化学习（SAC）与时序卷积网络（TCN），实现目标位姿不确定下的高精度销孔（peg-in-hole）接触装配任务。

### 1.2 分阶段路线

| 阶段 | 名称 | 核心内容 | 周期 | 硬件依赖 |
|---|---|---|---|---|
| **1a** | MuJoCo 仿真力控基础 | UR20 MuJoCo模型、力控算法库、验证脚本 | ~6-8周 | 无 |
| **1b** | UR20 真机力控部署 | ROS2控制栈、真机力控验证 | ~4-6周 | UR20 IP + F/T传感器 |
| **2** | MuJoCo RL 仿真 + SAC入门 | PegInHole Gym环境、SAC基线 | ~8-10周 | 无（GPU可选） |
| **3** | 论文复现 + Sim2Real | TCN双分支策略、域随机化、真机迁移 | ~12-16周 | UR20 + F/T |
| **4** | 扩展改进 + 论文 | 消融实验、算法改进、论文产出 | ~12周 | UR20 + F/T |

### 1.3 论文核心贡献（本研究目标）

1. **TCN 多模态时序策略网络**：融合本体位姿与六维力/力矩时序触觉数据
2. **Sim2Real + 域随机化**：仿真预训练 → 真机少量微调
3. **双层控制框架**：500Hz 柔顺控制器内环 + 20Hz SAC 策略外环
4. **一套策略同时输出运动子目标 + 自适应柔顺参数**

### 1.4 关键名词

| 名词 | 定义 |
|---|---|
| **SAC** | Soft Actor-Critic — 最大熵离线强化学习算法 |
| **TCN** | Temporal Convolutional Network — 时序卷积网络 |
| **Peg-in-Hole** | 销孔装配 — 将销钉（peg）插入孔（hole）的接触任务 |
| **Compliance Controller** | 柔顺控制器 — 力控底层闭环 |
| **Selection Matrix S** | 选择矩阵 — 各自由度在"力控"与"位控"之间的分配 |
| **Parallel Force-Position** | 并行力位控制 — 力控和位控每个自由度上并行叠加 |
| **Domain Randomization** | 域随机化 — 仿真参数随机化以弥合 Sim2Real 差距 |
| **Sim2Real** | 仿真到现实迁移 |

---

## 2. 系统架构

### 2.1 总体架构

```
┌────────────────────────────────────────────────────────────────────────┐
│                       外环：SAC/RL 策略 (20Hz)                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐     │
│  │ MLP Encoder  │    │ TCN Encoder  │    │ 共享MLP Head        │     │
│  │ (Propriocep) │    │ (F/T Hist)   │    │ → Actor, Critic     │     │
│  └──────┬───────┘    └──────┬───────┘    └─────────┬────────────┘     │
│         │                  │                       │                  │
│         └────► Concat ◄────┘                       │                  │
│                       │                            │                  │
│         输出动作 a=[a_x, a_p] 24维 ◄───────────────┘                  │
└──────────────────────────┬─────────────────────────────────────────────┘
                           │ 每 50ms (20Hz)
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   插值层 (Interpolator)                                  │
│  子目标位姿 ── 线性/样条插值 ──→ 500Hz平滑轨迹                           │
│  柔顺参数  ── 线性插值 ───────→ 500Hz平滑参数                            │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ 每 2ms (500Hz)
                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│              内环：柔顺控制器 (500Hz)                                     │
│                                                                        │
│  τ = S · [Kp_f·(F_d-F) + Ki_f·∫(F_d-F)dt]                            │
│      + (I-S) · [Kp_x·(x_d-x) + Kd_x·(ẋ_d-ẋ)]                         │
│                                                                        │
│  输入: 子目标 x_d, 柔顺参数 Kp_x,Kp_f,S, 当前状态 x,ẋ,F                │
│  输出: 关节控制量 τ                                                     │
│  安全约束: 力/位置超限保护                                               │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────────┐
              │  被控对象 (MuJoCo / UR20)    │
              │  q, dq, x, dx, F/T         │
              └────────────────────────────┘
```

### 2.2 仿真架构 (Phase 1a/2)

```
                         ROS2 总线
  /joint_states  /ft_sensor_raw  /joint_cmd  /action_cmd
       ↑              ↑              ↑           ↑
  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐  ┌──┴─────────┐
  │MuJoCo  │   │F/T模拟  │   │力控     │  │RL策略     │
  │仿真节点  │   │节点     │   │控制器   │  │推理节点   │
  │(Python) │   │(Python) │   │(C++)   │  │(Python)  │
  │500Hz   │   │500Hz   │   │500Hz   │  │20Hz     │
  └─────────┘   └─────────┘   └─────────┘  └────────────┘
```

### 2.3 RL 训练架构 (Phase 2)

```
脱离ROS2，MuJoCo Python API 直接跑
┌──────────────────────────────────────┐
│  RL训练脚本 (standalone)              │
│  ┌───────────┐  ┌──────────────────┐│
│  │ MuJoCo   │  │ SAC + TCN       ││
│  │ 引擎      │◄─┤ (PyTorch)       ││
│  └───────────┘  └──────────────────┘│
└──────────────────────────────────────┘
       │ 训练完导出模型 .pt
       ▼
加载到 ROS2 RL 推理节点 → 真机部署
```

### 2.4 真机架构 (Phase 1b/3)

```
                         ROS2 总线
  /joint_states  /ft_sensor_raw  /joint_cmd
       ↑              ↑              ↑
  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐  ┌──┴─────────┐
  │UR20    │   │F/T真机  │   │力控     │  │RL策略     │
  │驱动     │   │驱动     │   │控制器   │  │推理节点   │
  │(ur_robot│   │(Python) │   │(C++)   │  │(Python)  │
  │_driver) │   │         │   │500Hz   │  │20Hz     │
  └─────────┘   └─────────┘   └─────────┘  └────────────┘
```

**力控控制器 (C++) 仿真和真机完全同一份代码**，只换底层驱动。

### 2.5 语言分工

| 模块 | 语言 | 理由 |
|---|---|---|
| MuJoCo 仿真节点 | **Python** | 仅仿真用，迭代快 |
| 力控控制器节点 | **C++** | 仿真+真机共享，500Hz实时 |
| F/T 传感器节点 | Python | 读数据+滤波，逻辑简单 |
| RL 策略推理节点 | Python | PyTorch 原生，20Hz 低频 |
| RL 训练脚本 | Python | 脱离 ROS2，standalone 训练 |

### 2.6 各阶段模块启用范围

| 阶段 | 启用模块 | 未启用模块 |
|---|---|---|
| **1a** | MuJoCo仿真节点(Python) + 力控控制器(C++) + F/T模拟(Python) + 双速率插值 | SAC, TCN, RL, 真机 |
| **1b** | UR20驱动 + 力控控制器(C++) + F/T真机(Python) | SAC, TCN, RL |
| **2** | Phase 1a全部 + Gym Env + SAC基线(standalone) | TCN, Sim2Real |
| **3** | 全部 | — |

### 2.7 文件系统组织

```
force-contrl-rl/
├── simulations/                        # MuJoCo 仿真
│   ├── models/                         # UR20, 销孔场景
│   ├── mujoco_sim_node.py             # MuJoCo → ROS2 桥接 (Python)
│   └── ft_sensor_node.py              # F/T 传感器模拟 (Python)
├── real_robot/                         # ROS2 控制包
│   └── ur20_control/                  # ROS2 C++ 包 (力控控制器)
│       ├── include/ur20_control/       # 头文件
│       ├── src/                        # 源文件 + ROS2 节点
│       ├── launch/                     # 启动文件
│       └── config/                     # 参数配置
├── rl/                                 # RL 算法 (Phase 2+)
│   ├── sac/                            # SAC 实现 (Python)
│   └── networks/                       # TCN, MLP (Python)
├── envs/                               # Gymnasium RL 环境
├── sim2real/                           # 域随机化、模型导出
├── configs/                            # 配置文件
├── experiments/                        # 实验记录
├── tests/                              # 测试
├── scripts/                            # 训练/评估
├── SPEC.md
├── CLAUDE.md
└── requirements.txt
```

---

## 3. 仿真环境规格

### 3.1 MuJoCo 规格

| 参数 | 规格 |
|---|---|
| Python包 | `mujoco>=3.1` |
| 默认时间步 | `dt = 0.002s` (500Hz，对齐内环控制器) |
| 求解器 | Newton, `tolerance=1e-8`, `iterations=100` |
| 接触模型 | Soft contact (默认) |
| 运行模式 | **ROS2 节点** (Python)，发布/订阅 ROS2 话题 |

### 3.2 UR20 MuJoCo 模型规格

| 属性 | 来源 | 格式 |
|---|---|---|
| 运动学 | `ur_description` 官方URDF → MJCF | XML (MJCF) |
| 碰撞几何 | 胶囊体近似（简化网格） | MuJoCo primitives |
| 关节 | 6个旋转关节 (shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3) | revolute |
| 力传感器 | MuJoCo `<sensor><force/><torque/></sensor>` 在 wrist 3 link 末端 | site frame |

### 3.3 销孔场景规格 (Phase 2+)

| 组件 | 几何 | 接触参数 |
|---|---|---|
| Peg (销钉) | 圆柱/胶囊体，直径~10mm，长度~50mm | `condim=6`，摩擦 [0.3, 0.005, 0.0001] |
| Hole (孔) | 圆柱体空心/盒体，间隙 ~0.5-1mm | `condim=6`，摩擦同上 |
| 工作台 | 平面 | `condim=3` |

---

## 4. 力控算法规格

### 4.1 递进路径

```
导纳控制 (Admittance)          最简，力→Δx映射
    ↓ 引入动力学
阻抗控制 (Impedance)            M·ẍ + D·ẋ + K·x = F
    ↓ 引入选择矩阵
力位混合 (Hybrid FP)              S·ForceCtrl + (I-S)·PosCtrl (S ∈ {0,1})
    ↓ 并行叠加 (论文)
并行力位 (Parallel FP) [论文]    S·ForceCtrl + (I-S)·PosCtrl (S ∈ [0,1])
```

### 4.2 统一控制器接口 (C++)

```cpp
// 所有力控控制器实现的统一接口
#include <Eigen/Dense>

virtual Eigen::VectorXd compute(
    const Eigen::VectorXd& x_desired,   // (6,) 期望位姿
    const Eigen::VectorXd& x_current,   // (6,) 当前位置
    const Eigen::VectorXd& dx_current,  // (6,) 当前速度
    const Eigen::VectorXd& force_ext    // (6,) 外部力/力矩
) = 0;  // 返回 (6,) 控制输出
```

### 4.3 导纳控制

```
Δx = F_ext / K_adm

参数: K_adm ∈ [100, 10000] N/m
工作模式: 外力 → 位置修正 → IK → 关节指令
```

### 4.4 阻抗控制

```
M·(ẍ_d - ẍ) + D·(ẋ_d - ẋ) + K·(x_d - x) = F_ext

参数: M ∈ [1,50] kg, K ∈ [100,10000] N/m, D = 2·ζ·√(M·K)
推荐阻尼比 ζ = 0.7 ~ 1.0
```

### 4.5 力位混合控制

```
τ = S · [Kp_f·(F_d-F) + Ki_f·∫(F_d-F)dt]
  + (I-S) · [Kp_x·(x_d-x) + Kd_x·(ẋ_d-ẋ)]

S ∈ {0,1}⁶ (离散切换)
Pi 力控 + PD 位控
```

### 4.6 并行力位控制 — 论文方案

```
τ = S · [Kp_f·(F_d-F) + Ki_f·∫(F_d-F)dt]  
  + (I-S) · [Kp_x·(x_d-x) + Kd_x·(ẋ_d-ẋ)]

S ∈ [0,1]⁶ (连续值)
```

**18 个可调参数**:

| 参数 | 维度 | 范围 | 说明 |
|---|---|---|---|
| Kp_x | 6 | [100, 10000] | 位置比例增益 |
| Kp_f | 6 | [0.1, 100] | 力比例增益 |
| S | 6 | [0, 1] | 选择矩阵（连续） |
| Kd_x | 6 | Kp_x × 0.1 | 位置微分增益（自动派生） |
| Ki_f | 6 | Kp_f × 0.01 | 力积分增益（自动派生） |

### 4.7 双速率插值器

| 任务 | 20Hz输出 | 500Hz插值方法 |
|---|---|---|
| 子目标位姿 | 每50ms更新 | 三次样条插值 |
| 柔顺参数 | 每50ms更新 | 线性插值 |

---

## 5. RL 算法规格 (Phase 2+)

### 5.1 SAC 核心参数

| 参数 | 值 | 说明 |
|---|---|---|
| 算法 | SAC (Haarnoja 2018) | 最大熵离线RL |
| 实现基础 | CleanRL `sac_continuous_action.py` | 保留训练循环，修改网络架构 |
| 策略 | Gaussian (μ, log_std), Tanh squashed | 连续动作分布 |
| Q网络 | 双Q (min取保守值) | 缓解 Q 值高估 |
| 自动熵调节 | ✅ | 目标熵 = -动作维度 |
| 延迟策略更新 | 每2步 | 稳定训练 |

**超参数初始化**:

| 参数 | 初始值 | 调参范围 |
|---|---|---|
| Learning rate | 3e-4 | 1e-4 ~ 1e-3 |
| Buffer size | 1e6 | — |
| Batch size | 256 | 128 ~ 512 |
| γ (discount) | 0.99 | 0.95 ~ 0.999 |
| τ (target update) | 0.005 | 0.001 ~ 0.01 |
| α_init | 0.2 | 自动调节 |

### 5.2 观测空间

```
obs = {
  "proprioception": [
    Δx (位姿误差 6-DOF),
    dx (末端速度 6-DOF),
    F_desired (期望插入力 scalar),
    a_prev (上一步动作 12/24-DOF)
  ] → 总 ~25-37维

  "force_torque_history": [
    历史12帧 × 6维 F/T
  ] → 形状 (batch, 6, 12)
}
```

### 5.3 动作空间

| 维度 | 含义 | 范围 |
|---|---|---|
| a_x(6) | 子目标位姿 (x,y,z,rx,ry,rz) | [-1,1] → 物理映射 |
| a_p(Kp_x)(6) | 位置增益 | [0,1] → 物理映射 |
| a_p(Kp_f)(6) | 力增益 | [0,1] → 物理映射 |
| a_p(S)(6) | 选择矩阵 | [0,1] 连续值 |

### 5.4 奖励函数

**Phase 2 (简单基线)**:
```
R = -dist * 0.1 - ||F - F_d||² * 0.01 + (10 if completed else 0)
```

**Phase 3 (论文分段加权)**:
```
R = w_f · R_force + w_k · R_sparse

R_force = -||F_actual - F_desired||²
R_sparse = α·(T-t)  (完成, dist<1mm) | -β (碰撞/超力) | 0 (其他)
```

推荐权重: w_f=0.1, w_k=1.0, α=1.0, β=10.0

### 5.5 经验回放

- **Phase 2**: Uniform Replay (CleanRL 原生)
- **Phase 3**: 可选实现 Prioritized Experience Replay (Sum Tree, Schaul 2016)

---

## 6. 网络架构规格 (Phase 3)

### 6.1 TCN 网络

```
Input: (batch, 6, 12)   # F/T: 6通道 × 12时间步
  ↓
TCN Block 1: Conv1d(6→64) dilation=1, kernel=2, causal, residual
TCN Block 2: Conv1d(64→64) dilation=2, kernel=2, causal, residual
TCN Block 3: Conv1d(64→64) dilation=4, kernel=2, causal, residual
TCN Block 4: Conv1d(64→64) dilation=8, kernel=2, causal, residual
  ↓
Global Avg Pooling (over time dim)
  ↓
Output: (batch, 64)      # 力觉特征向量
```

每个 TCN Block:
```
输入 → Conv1d(dilation) → ReLU → Dropout → Conv1d(dilation) → ReLU → Dropout
  ├→ 1×1 Conv (通道数变化时)
  └→ + (残差连接) → ReLU → 输出
```

### 6.2 双分支策略网络

```
Proprioception(25-37)           Force-Torque History(6, 12)
       │                                  │
       ▼                                  ▼
┌─────────────────┐           ┌───────────────────────┐
│ MLP Encoder     │           │ TCN Encoder            │
│ FC(256)→ReLU    │           │ 4层 causal Conv1d      │
│ FC(128)→ReLU    │           │ dilation[1,2,4,8]      │
│ → 64-dim        │           │ → 64-dim               │
└────────┬────────┘           └───────────┬───────────┘
         │                                │
         └─────────── Concat ◄────────────┘
                          │
                          ▼
               ┌──────────────────────┐
               │ Shared MLP Head      │
               │ FC(256)→ReLU         │
               │ FC(256)→ReLU         │
               └──────────┬───────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
  Actor(μ,σ)        Critic Q1        Critic Q2
  24-d动作           1-d Q             1-d Q
```

### 6.3 CleanRL 集成

保留 CleanRL 的训练循环、损失函数、超参数逻辑，替换网络结构：
- `SACActor` → `TCNActor` (双分支 MLP + TCN)
- `SoftQNetwork` → `TCNCritic` (双分支 + action 拼接)
- `ReplayBuffer` → 扩展为存储时序数据

---

## 7. Sim2Real 规格 (Phase 3)

### 7.1 域随机化参数

| 参数 | 范围 | 说明 |
|---|---|---|
| 初始/目标位姿误差 | ±5mm, ±5° | 模拟视觉定位误差 |
| 表面刚度 | ±50% | 覆盖橡胶→金属 |
| 摩擦系数 | [0.3, 1.5] | 低摩擦到高摩擦 |
| 期望插入力 | 5-15 N | 任务无关随机化 |
| F/T传感器噪声 | ~2% 量程高斯 | 模拟真实传感器 |
| 控制延迟 | 1-2步 | 通信延迟 |
| 工件质量 | ±20% | 惯性随机 |

### 7.2 F/T 传感器噪声模型

```python
noisy_ft = raw_ft + gaussian(σ=[0.1, 0.1, 0.1, 0.005, 0.005, 0.005])
noisy_ft += bias_drift  # 零漂
filtered = lowpass(noisy, cutoff=20Hz, fs=500Hz)  # 低通滤波
```

### 7.3 Sim2Real 流程

```
仿真预训练 (50万步)
    → 模型导出 (TorchScript / ONNX)
    → ROS2 推理节点 (加载 Actor 网络)
    → 真机零微调测试 (评估成功率)
    → 真机微调 (~5000步, ~20分钟)
    → 对比分析 (Sim vs Real)
```

---

## 8. 接口定义

### 8.1 控制器统一接口 (C++, Phase 1a)

```cpp
// 所有力控控制器实现此接口
#include <Eigen/Dense>

class ForceController {
public:
    virtual ~ForceController() = default;
    
    virtual Eigen::VectorXd compute(
        const Eigen::VectorXd& x_desired,   // (6,) 期望位姿
        const Eigen::VectorXd& x_current,   // (6,) 当前位置
        const Eigen::VectorXd& dx_current,  // (6,) 当前速度
        const Eigen::VectorXd& force_ext    // (6,) 外部力/力矩
    ) = 0;                                   // (6,) 控制输出
};
```

### 8.2 启动命令 (Phase 1a)

```bash
# 仿真模式 — 启动 MuJoCo 仿真节点 + 力控控制器
ros2 run ur20_control mujoco_sim_node          # MuJoCo (Python)
ros2 run ur20_control force_controller_node    # 力控控制器 (C++)

# 真机模式 (Phase 1b) — 只换底层驱动
ros2 launch ur_robot_driver ur20.launch.py     # UR20 驱动
ros2 run ur20_control force_controller_node    # 力控控制器 (C++, 完全不变)
```

### 8.3 Gymnasium 环境接口 (Phase 2+)

```python
env = gym.make('PegInHole-v0', config=cfg)
obs, info = env.reset(seed=42)          # Dict / Box
action = policy(obs)                    # (6,) or (24,)
obs, reward, term, trunc, info = env.step(action)
```

### 8.4 ROS2 话题接口

**仿真和真机共用同一套话题接口**，只换底层驱动节点。

| 话题 | 类型 | 方向 | 说明 |
|---|---|---|---|
| `/joint_states` | sensor_msgs/JointState | 发布 | 关节位置/速度 (仿真 or 真机) |
| `/ft_sensor_raw` | geometry_msgs/WrenchStamped | 发布 | F/T 原始数据 (仿真 or 真机) |
| `/ft_sensor_filtered` | geometry_msgs/WrenchStamped | 发布 | 滤波后 F/T |
| `/joint_cmd` | sensor_msgs/JointState | 订阅 | 关节位置/力控制指令 |
| `/action_cmd` | 自定义 (Phase 3) | 订阅 | RL 策略输出的子目标+柔顺参数 |

**切换仿真/真机只需换启动命令**：

```bash
# 仿真模式
ros2 run ur20_control mujoco_sim_node.py       # MuJoCo 仿真节点 (Python)
ros2 run ur20_control force_controller_node     # 力控控制器 (C++, 同一份代码)

# 真机模式
ros2 launch ur_robot_driver ur20.launch.py      # UR20 驱动 (替换仿真节点)
ros2 run ur20_control force_controller_node     # 力控控制器 (C++, 完全不变)
```

---

## 9. 代码规范

### 9.1 Python 风格

- **Python 3.10+**
- 类型注解：函数参数和返回值标注类型
- Docstring: Google style
- 最大行宽：100字符

### 9.2 命名规范

| 类别 | 规范 | 示例 |
|---|---|---|
| 文件/目录 | snake_case | `parallel_fp.py` |
| 类 | PascalCase | `AdmittanceController` |
| 函数/方法 | snake_case | `compute_reward()` |
| 变量 | snake_case | `force_external` |
| 常量 | UPPER_SNAKE | `MAX_FORCE_SAFE` |
| 配置键 | snake_case | `kp_x_range` |

### 9.3 Git 规范

```
分支: main(stable) | dev | feature/* | phase/*
提交: [Phase 1a] Add admittance controller
```

---

## 10. 附录

### 10.1 依赖清单

```
mujoco>=3.1
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
pyyaml>=6.0
torch>=2.0
gymnasium>=0.29
tensorboard>=2.13
```

### 10.2 推荐参考资源

| 主题 | 资源 |
|---|---|
| 力控综述 | Handbook of Robotics Ch.8 — Force Control |
| 阻抗控制 | Hogan 1985 — Impedance Control: An Approach to Manipulation |
| SAC | Haarnoja et al. 2018 — Soft Actor-Critic |
| TCN | Bai et al. 2018 — TCN paper (locuslab/TCN) |
| Prioritized Replay | Schaul et al. 2016 — Prioritized Experience Replay |
| CleanRL | Huang et al. 2022 — CleanRL: High-quality Single-file RL |
| RL 教材 | Sutton & Barto — Reinforcement Learning: An Introduction |
| MuJoCo 文档 | https://mujoco.readthedocs.io |
| UR20 ROS2 驱动 | github.com/UniversalRobots/Universal_Robots_ROS2_Driver |

### 10.3 论文引用

```bibtex
@article{li2020variable,
  title={Variable Compliance Control for Robotic Peg-in-Hole Assembly: 
         A Deep-Reinforcement-Learning Approach},
  author={Li, X. and others},
  journal={Applied Sciences},
  year={2020},
  volume={10},
  number={19}
}
```

---

*本规格文档随项目进展持续更新。每阶段开始时补充对应细节。*