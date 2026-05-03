# GPT Review Round 2

## Recommendation

**Major Revision, but substantially improved from Round 1.**

这一轮修改确实解决了不少第一轮的核心问题：主表数字基本统一了，加入了 architecture-matched WSRL/RLPD E=10 对比，补了 operational metrics、per-decision reward、BC-anchor ablation，也把统计检验从 naive pooled Welch t-test 改成了 hierarchical bootstrap。整体说服力比第一版明显强。

但当前版本我仍不建议直接接收。主要原因已经不是“缺一两个实验”，而是修改后产生了新的内部不一致，并且交通意义和算法归因还没有完全站稳。若目标是 ML workshop，这版接近可投；若目标是 T-ITS/TR-C 级别，还需要再清理一轮。

## What Improved

1. **数字主线比第一轮更清楚。** 摘要和 Table II 现在一致报告 RE-SAC best/final 为 `-4184 / -4550`，不再混用旧的 `-4267 / -4520`。
2. **补了 architecture-matched baselines。** WSRL-E10 和 RLPD-E10 的加入直接回应了第一轮最大的公平性问题。
3. **补了 operational metrics。** 新表加入 large-gap rate、holding mean/p90、per-decision reward，至少开始把 shaped return 和交通指标联系起来。
4. **统计处理更合理。** hierarchical bootstrap 比直接 pool 50 episodes 做 Welch t-test 更合适。
5. **BC anchor 被单独 ablate。** no-BC row 有助于说明 RE-SAC 不是简单 BC regularization。
6. **station embedding 的问题被解释了。** Appendix 现在说明 station cardinality 为 1，是一个保留 slot。

## Major Remaining Issues

### 1. Checkpoint / seed / reproducibility accounting is still inconsistent

这是当前最需要先修的问题。

- Abstract 说 **565 post-hoc-evaluated checkpoints**。
- Contribution、Experimental protocol、Appendix、Code availability 仍然说 **349 checkpoints**。
- 本地 `eval_results.csv` 有 564 data rows，`eval_E10.csv` 另有 36 rows，`eval_noBC.csv` 另有 30 rows。如果这些都用于论文，实际总数不是 349，也不是 565。
- Main protocol 说四个主方法扩展到 5 seeds，但 Appendix common hyperparameters 仍写 random seeds 是 `{42,123,456}`。
- Evaluation 说 “reset seed varied”，Appendix 又说 SUMO seed is not set by code。这两句话仍然冲突。

建议加一个 checkpoint inventory table：method、seeds、checkpoint grid、number of evaluated checkpoints、source csv、whether included in figures/table。否则可复现性叙述仍然不可信。

### 2. Several result statements are stale or internally inconsistent

一些文字还停留在旧版本，和现在的表格对不上：

- Abstract 说 offline-to-online final returns are `-4819` to `-5008`，但 Table II 里 RLPD-0.75 final 是 `-4689`，WSRL-E10 final 是 `-4634`。
- R3 说 WSRL/RLPD best returns form a cluster in `[-4597, -4552]`，但 Table II 里 WSRL 是 `-4626`，RLPD-0.50 是 `-4644`，而 WSRL-E10 是 `-4365`。
- SAC600 section 说 RE-SAC best/final differ by only 253 across three seeds，但 Table II 的 five-seed values是 `-4184` 到 `-4550`，差 366。
- Ablation text 仍说 LCB-only reaches `-4157 ± 40`，但 Table IV 现在是 `-4191 ± 68`。
- Ablation text 仍提到 three-seed full `-4267` vs LCB-only `-4157`，但表格主线已换成 five-seed `-4184` vs `-4191`。
- Table IV 把 full variant 的 Final `-4550` 加粗，但 LCB-only Final `-4411` 更好。Final column 的 bold 应该给 LCB-only。
- Operational discussion 说 RLPD mean hold 是 `14.4s`，表中是 `15.9s`。

这些问题会让审稿人怀疑结果是手工拼接后没有统一核对。建议全篇重新跑一遍数字 grep，尤其是 `-4157`, `-4267`, `253`, `349`, `565`, `14.4`, `-5008`。

### 3. Algorithmic attribution is still over-claimed

论文现在反复说 “LCB term is the single dominant contributor”。Table IV 不完全支持这个说法。

从 Table IV 看：

- twin-Q none: `-4444`
- ensemble none: `-4303`
- ensemble LCB-only: `-4191`
- full LCB+L1: `-4184`

也就是说：

- plain offline twin-Q already beats all standard twin-Q offline-to-online baselines。
- ensemble alone gives a large part of the improvement。
- LCB helps on top of ensemble, but according to current table the gain is about 112 return, not text里的 146。
- LCB-only has better final return than full LCB+L1。

更稳妥的叙述应该是：**offline actor-critic with BC anchor is already strong; ensemble improves it; LCB further improves ensemble-based offline training; L1 is not useful for this offline benchmark.** 不建议继续说 LCB 是唯一 dominant factor。

### 4. Architecture-matched baselines are useful, but the conclusion should be more careful

WSRL-E10 和 RLPD-E10 是很好的补充，但当前写法 “rules out ensemble alone” 仍有点强。

- WSRL-E10 best `-4365` 已经明显接近 RE-SAC `-4184`，比原 WSRL 强很多。
- WSRL-E10 final `-4634` 和 RE-SAC final `-4550` 只差 84 return；在 3 seeds vs 5 seeds 情况下，需要统计检验或至少 CI discussion。
- E10 variants 没有出现在 hyperparameter table 中：REDQ ensemble size、sampled heads M、target Q rule、UTD、是否 warm-start critic，都应该写清楚。
- RLPD 仍是 UTD=1。原 RLPD 的强性能通常和 implementation details / UTD / critic stabilization 有关。当前方法可以叫 “our RLPD-style data-mixing baseline”，但直接称为 full state-of-the-art RLPD 仍偏强。

建议对 RE-SAC vs WSRL-E10 / RLPD-E10 也做 hierarchical bootstrap，至少报告 best/final 的差值和 CI。

### 5. Transportation relevance is improved but still not sufficient

Operational table 是进步，但结果本身也暴露了一个问题：交通指标差异很弱。

- Bunching rate 全部为 0。
- Large-gap rate 只有 18-21%，区分度很低。
- Forward headway deviation 79-89s，backward 136-144s，各方法差异很小。
- 作者自己承认 return 差异主要来自 reward symmetry term 和 tanh penalty，而不是 gross headway deviation。

这会让交通领域审稿人问：RE-SAC 的 return 提升是否真的改善乘客体验？目前还缺：

- passenger waiting time；
- passenger in-vehicle delay / total travel time；
- completed trips / arrivals；
- per-line headway CV；
- per-line performance table；
- no-holding baseline；
- classical headway-based holding baseline。

如果暂时不能补乘客级指标，至少应把结论收窄为 “improves the benchmark reward and per-decision reward”，不要直接暗示 operational deployment superiority。

### 6. Per-decision normalization was added, but decision count should be reported

Table III 有 `R/dec`，这是好修改。但正文说 decision counts are within 5%，表里没有 decision count，也没有 mean/std。

因为 aggregate return 是否受固定 2.5h 内 decision event 数影响是第一轮的关键问题，建议把 `Dec./ep` 加入 Table III。比如直接报告每个方法 average decisions per episode，这比正文一句 “within 5%” 更有说服力。

### 7. Heuristic baseline and “BC ceiling” still need cleanup

Table II 仍然只有 heuristic single deterministic run。作为数据生成控制器，这不够正式。

另外 “behavior cloning's ceiling is therefore near heuristic by construction” 仍不严谨，因为 BC best `-8440` 明显好于 heuristic `-9783`。可以说 BC final collapses toward the behavior-policy level，但不要说 ceiling。

如果不能重新评估 heuristic，也建议明确：heuristic row is descriptive only, not a statistical baseline。更好的是补 30 episodes heuristic / no-holding / classical holding。

### 8. Observation features may still involve future information

Observation 里有 “predicted boarding queue” 和 “predicted travel time to next stop”。这些特征可能合理，但当前没有解释如何预测。

必须说明它们是否只使用当前可观测状态和历史统计。如果它们来自 SUMO realized future travel time / future passenger arrivals，就会构成 information leakage。这个问题对离线 RL benchmark 的有效性很关键。

### 9. Station embedding explanation is better, but main text still overstates it

Appendix 说 station embedding cardinality is 1，即 station identity 实际是常数；但 MDP section 仍说 categorical features include station and index learned embedding tables。严格说这不是 station identity。

建议把 main text 改成：the state contains a reserved station slot but no per-station identity; spatial position is represented by route progress and local features。否则读者会以为策略真的能区分 353 个站点。

### 10. Equation and algorithm still disagree on entropy

Eq. (LCB policy loss) 写的是：

`-E[mu_Q + beta sigma_Q] + beta_bc L_BC`

Algorithm 里 ensemble statistics 是对 `Q_i(s,a) - alpha log pi(a|s)` 求 mean/std。Entropy term 是否进入 sigma 也会影响方法定义。需要把 equation 和 algorithm 统一。

## Minor Issues

- Table I 仍写 “Bus vehicles per day”，但正文实际是 2.5-hour episode 内的 scheduled bus vehicles/trips。
- Appendix code availability 仍说 `auto_run.py` launches 22-experiment suite，但论文现在加入 E10 和 noBC，实验数量不再显然是 22。
- Figure 3 / curves 似乎没有包含 E10 offline-to-online baselines；如果表中把 E10 作为主结果，图注应说明 curves only show original variants。
- “any checkpoint past 20K is deployable” 仍偏强。Best checkpoint 是 post-hoc benchmark metric，不是无需 simulator 的真实部署选择。
- Calibration against real data 仍是 assertion，没有 calibration error 或 validation summary。
- WSRL/RLPD E10 使用 `chen2021redq` citation，refs 已有该条目；这一点没问题，但 Methods/Appendix 需要更完整描述。

## Suggested Revision Priority

1. First fix all stale numbers and checkpoint/seed accounting.
2. Add a checkpoint inventory table.
3. Add hierarchical bootstrap for RE-SAC vs WSRL-E10 and RLPD-E10.
4. Reframe algorithm attribution: offline training + ensemble + LCB, not “LCB single dominant”.
5. Add `Dec./ep` to operational table and fix operational text mismatches.
6. Either add passenger/classical baseline metrics, or explicitly narrow transportation claims.
7. Clarify observation feature generation and SUMO seed handling.
8. Make equation, algorithm, appendix hyperparameters, and code availability consistent.

## Bottom Line

这版已经比第一轮强很多，尤其是 E10 baselines 和 hierarchical bootstrap 让主结论更可信。但当前仍有太多 manuscript-level inconsistencies，会让审稿人对结果整理和复现细节不放心。

我的第二轮建议仍是 **Major Revision**。如果把数字、checkpoint accounting、seed handling、ablation narrative 和 operational interpretation 清理干净，这篇可以变成一个有价值的 SUMO offline RL benchmark paper。若目标是交通顶会/期刊，还需要再补乘客级或经典控制 baseline，才能支撑“公交运营改善”的强结论。
