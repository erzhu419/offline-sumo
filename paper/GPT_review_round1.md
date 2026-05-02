Recommendation: Major Revision / 当前版本不建议直接接收

Summary

本文构建了一个 12 条公交线路、389 辆公交、353 个站点的 Changsha-SUMO 公交控制 benchmark，比较 offline RL、online SAC、WSRL、RLPD 等 offline-to-online 方法，并提出 RE-SAC offline，即 10-head Q ensemble + LCB pessimism + BC anchor 的纯离线方法。论文的核心结论是：在该公交 holding 任务上，良好正则化的纯 offline RL 可以匹配甚至超过 offline-to-online fine-tuning。

这个方向是有价值的，benchmark 规模也不错，post-hoc checkpoint evaluation 和代码/数据/checkpoint 释放都是加分项。但当前稿件还有较多会影响可信度的问题，尤其是数字不一致、baseline 公平性、交通指标解释不足和统计处理不严谨。

Strengths

任务设定有应用价值：多线路、全网统一策略控制，比单线公交 holding 更接近真实系统。
SUMO 场景规模较大，包含公交、社会车辆、乘客需求和真实线路拓扑，作为 benchmark 有潜在贡献。
post-hoc evaluation 每个 checkpoint 重新跑 10 个 SUMO episode，比训练中 2-episode eval 更严谨。
论文不仅比较 BC/H2O+/CQL/SAC/WSRL/RLPD，还做了 RLPD offline ratio 和 RE-SAC ablation，实验覆盖面较广。
加入 operational metrics 是正确方向，说明作者意识到 shaped RL return 不足以说服交通领域读者。
Major Concerns

核心数字存在多处内部不一致，当前版本可信度受影响。
Table II 中 RE-SAC best/final 是 -4184 / -4550，但摘要和 Results R2 中仍写 -4267 / -4520。RLPD/WSRL 的若干文本描述也和表格不一致。p-value 也有冲突：正文说 p < 1e-6，结论又写 p=0.0019。这些不是小 typo，因为它们直接支撑主结论。

baseline 公平性仍然不足。
RE-SAC 使用 10-head ensemble，而 WSRL/RLPD/SAC 基本是 twin-Q、UTD=1。RLPD 原方法的效果往往依赖高 UTD、critic stabilization，部分实现也使用 ensemble。当前对比更像是“一个更强 critic 架构的 offline 方法”对比“较简化的 offline-to-online 方法”。建议补充 architecture-matched RLPD/WSRL，例如 10-head critic 或 REDQ-style critic；否则应弱化结论，改成“在我们的 twin-Q baseline 实现下，RE-SAC 更强”。

ablation 与论文叙事不完全一致。
Table IV 显示 twin-Q offline none 已经达到 -4444，超过所有 offline-to-online baseline；ensemble-only 达到 -4303；LCB-only/full 进一步到约 -4191/-4184。这说明主增益不完全来自 LCB，甚至一个未作为主方法报告的 plain offline actor-critic + BC anchor 已经很强。论文说 “LCB is the single dominant contributor” 过强。还需要 ablate BC anchor，否则无法判断 RE-SAC 的关键因素到底是 BC anchor、ensemble、LCB，还是训练协议。

交通意义还没有被充分证明。
Operational table 中所有方法 bunching rate 都是 0.0，forward/backward headway deviation 差别也很小。论文自己说 return 差异主要来自 symmetry term 和 large-deviation tanh penalty，而不是粗粒度 headway deviation。这会让交通领域审稿人质疑：RL return 的提升是否真的改善服务质量？建议补充 passenger waiting time、in-vehicle delay、total travel time、large-gap rate、per-line headway CV、holding-time distribution、completed trips/arrivals 等指标。

aggregate return 可能受 decision event 数量影响。
episode return 似乎是异步 stop-arrival reward 的总和。不同策略 holding/speed action 会改变固定 2.5 小时内发生的 stop arrivals 数量。如果某策略产生更少决策事件，它的总负 reward 可能“看起来更好”。需要报告每个 episode 的 decision count、per-decision reward、per-passenger 或 per-stop normalized metrics。

统计分析需要更严谨。
论文把 episode-level samples 直接 pool 起来做 Welch t-test，但训练 seed、checkpoint selection、episode randomness 是层级结构。建议使用 seed-level mean 作为主统计，或做 hierarchical bootstrap。若可能，所有方法使用 common random numbers 做 paired evaluation。heuristic baseline 目前只有 single deterministic run，也不能和其他方法的 30/50 episode evaluation 直接比较。

泛化 claim 过强。
摘要说从 operational logs 泛化到 new traffic conditions，但当前实验基本是在同一 SUMO 分布下训练和评估，只是 episode seed 有变化。若要支撑泛化，需要 demand shift、traffic intensity shift、route disruption、passenger OD shift，或至少在文字上把 claim 改成 same-scenario stochastic evaluation。

Minor / Clarity Issues

Table caption “Lower-is-better for |return|” 很容易误导。建议直接写 “returns are negative; less negative is better”。
Appendix 中 station embedding 写成 1 -> 2，但正文说有 353 stops 且 observation 包含 station id。这里必须澄清是文档错误还是实现确实没有 station identity。
Training checkpoint interval 正文写 5K，Appendix 写 10K；349 checkpoints 的 breakdown 也需要 inventory table。
SUMO seed 一处说 “reset seed varied”，另一处说 “not set by our code”，互相矛盾。
CQL 不应称为 naive offline method；它是标准 conservative offline RL baseline。
Algorithm 中 policy loss equation 没写 entropy term，但伪代码里把 Q - alpha log pi 放进 ensemble stats，需要统一。
L1 相关叙述仍显混乱。既然 ablation 显示 L1 不帮助，应把主方法明确改成 ensemble + LCB，或解释为什么主表仍报告 full LCB+L1。
Bottom Line

这篇论文有一个不错的应用 benchmark 和一个有潜力的实验发现：在该 SUMO 公交 holding 设置中，纯 offline RL 可能不弱于 offline-to-online fine-tuning。但当前稿件需要先把数字、统计、baseline 公平性和交通指标解释补扎实。我的建议是 Major Revision。如果目标是交通类期刊/会议，最关键的补实验是：正式 heuristic/no-hold/classical holding baseline、passenger/service-level metrics、architecture-matched RLPD/WSRL，以及至少一个 demand/traffic shift evaluation。