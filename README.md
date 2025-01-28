# radar_opt
Radar scheduling algorithm for BM, CM, and aircraft

## 环境设置

采用雷达网络接力跟踪的方式，每一部雷达都有自己单独的通道，同一时间只能跟踪一个目标。目标行进轨迹按照弹道导弹、巡航导弹和战斗机为蓝本设计其基本路线。

## 优化目标

优化目标是最大化全体目标的累积跟踪时长。

## 约束


目标在第一次进进入空域后，每个目标在其存在的时间内，任意时刻最多只能被一部雷达跟踪
每部雷达在任意时间跟踪的目标数量不能超过其通道限制
雷达只能跟踪在其探测半球内的目标
雷达而言，目标仅存在跟上和未跟上的状态，因此目标的虚实具有二元性特点

## 所需方程

### 雷达方程

雷达基本方程：

$$P_r=\frac{P_tG_tG_r\lambda\sigma}{(4\pi)^3R^4L}$$

式中$P_r$表示雷达接收电磁波的回波功率，$G_t$和$G_r$分别表示雷达天线的发射增益和接收增益，$\lambda$表示雷达发射的电磁波波长，$\sigma$表示目标截散射截面积（RCS），$R$为目标与雷达的距离，$L$为系统传播损耗（包括大气吸收、杂波等）。

因此我们可以得到雷达的最大探测距离，方程如下。

$$R_{\max}=(\frac{P_tG_tG_r\lambda^2\sigma}{(4\pi)^3P_{\min}L})^{\frac{1}{4}}$$

式中$R_{\max}$表示雷达的最大探测距离，$P_{\min}$表示雷达最小可检测功率，一般由信噪比和噪声水平计算得到。为便于后续雷达网络任务的调度计算，我们将$P_t$，$G_t$，$G_r$，$\lambda$，$\sigma$，$P_{\min}$和$L$都定义为固定值，则每部雷达的最大可检测范围即为固定值，同时忽略雷达的俯仰角对顶部盲区的影响，则$R_{\max}$对于每部雷达而言将会是常量。因此我们将雷达辐射范围近似为半球体，其半径为$R_{\max}$。

### 目标运动方程

任何目标都有一个基础运动模型，公式如下。

$$\mathbf{f}(\mathbf{x}^t_i)=\mathbf{f}(\mathbf{x}^{t-1}_{i})+\mathbf{\zeta}^{t-1}_{i}$$

式中$\mathbf{f}(\cdot)$表示目标的运动模型，$\mathbf{\zeta}^t_i$表示目标的随机扰动量.进一步地，我们可得出任一时刻的战场态势模型。

$$\mathbf{s}(\mathbf{X}^t)=(\mathbf{f}(\mathbf{x}^{t-1}_{i})+\mathbf{\zeta}^{t-1}_{i})\oplus(\mathbf{R}(\mathbf{r}))$$

式中$\mathbf{s}(\mathbf{X}^t)$表示$t$时刻全体目标的态势融合结果，$\mathbf{R}(\mathbf{r})$表示雷达网络的状态信息，并且有$\mathbf{R}(\mathbf{r})=\{\mathbf{r}_j|j=1,2,\cdots,M\}$。“$\oplus$”符号表示目标运动状态与雷达状态的信息融合。

## 项目结构

项目结构如下所示。

```
├── algorithm
│   ├── HRO.py
│   └── final_corrected_rule.py
├── radar.md
└── trajectories
    └── generate_trajectories.py

```

