# GPT Review Round 4

## Recommendation

**Weak Accept / Minor Revision for an ML or simulation-benchmark venue.**

如果目标是偏交通工程的 venue（例如需要证明真实乘客服务质量改进的 T-ITS / TR-C 风格论文），我仍会要求 **Major Revision**，主要因为 passenger-level metrics 和经典交通控制 baselines 还没有补齐。但如果论文定位是 offline RL benchmark / applied ML on SUMO，这版已经基本可以过了，剩下是少量表述、格式和统计口径收尾。

这一轮最大的变化是：参考文献问题已经解决，LaTeX 日志里没有 citation/reference warning；结论也从“offline RL beats all offline-to-online”改成了更准确的 “matches or directionally exceeds, with WSRL-E10 statistically tied”。这让主张可信很多。

## What Is Now Fixed

1. **References are fixed.** 当前 PDF citation 正常编号，`main.log` 没有 citation / reference warning。
2. **Appendix warning 基本解决。** 上一轮的 `Ignoring useless \section in Appendix` 没有再出现。
3. **Main claim 更保守。** Intro 和 conclusion 都明确承认 WSRL-E10 在 Best 上 95% CI crosses zero，Final 也是 statistically tied，不再强说 offline RL categorically dominates。
4. **Conclusion 已同步当前数字。** 现在写的是 RLPD gap `+462`、RLPD-E10 gap `+371`、WSRL-E10 Best `+181` with CI crossing zero、Final `p=0.65`，比上一轮 stale numbers 好很多。
5. **Checkpoint accounting 修好了。** Abstract / protocol / compute environment / code availability 都已经统一到 630 evaluated checkpoints。
6. **Extended online SAC 被正确降级。** 现在明确是 single-seed indicative study，而不是正式统计结论。
7. **Operational table 展示更清楚。** `Hold-p90`、decision count、per-decision reward、passenger-level caveat 都比之前清楚。
8. **Methods count 和 E10 baseline 区分更合理。** 现在说 method families，并说明 WSRL-E10 / RLPD-E10 是作者的 architecture-matched variants，不是完整复现原论文所有 tuning。

## Remaining Issues

### 1. PDF 里仍有一个明显的 appendix cross-reference 问题

当前 PDF 在 MDP observation 段落里仍渲染为：

> Appendix B-0a

这比上一轮的 `Appendix 0a` 好，但还是很像格式错误。原因大概率是 `\label{appendix:categorical}` 放在 appendix 里的 `\paragraph{Categorical embedding layer.}` 后面，IEEEtran 把 paragraph 编号拼成了 `B-0a`。

建议直接改成以下任一方式：

- 不引用 paragraph 编号，写成 “see Appendix B for architecture details”。
- 把 categorical embedding 做成正式 appendix subsection，再 label subsection。
- 如果只是补充说明，可以直接删掉 cross-reference，在正文括号里短句说明 station slot cardinality is 1。

这是当前最显眼的 presentation bug。

### 2. Limitations 里 WSRL-E10 Final 的统计描述不准确

正文 R4 和 conclusion 已经写对了：WSRL-E10 Final 是 `+84`, seed-level `p=0.65`，应解读为 statistically indistinguishable。

但 limitations 里仍写：

> the WSRL-E10 contrast at the Final metric is statistically borderline

`p=0.65` 不是 borderline。真正 borderline-ish 的是 WSRL-E10 **Best** contrast（CI crosses zero, `Pr=0.93`），不是 Final。建议改成：

> the WSRL-E10 Best contrast remains borderline, while the Final contrast is statistically indistinguishable under the present 3-vs-5 seed comparison.

### 3. Common hyperparameter table 的 seed row 仍是旧口径

Appendix 后面的 Seeds 段已经写清楚：主比较和两个 RE-SAC cells 用五个 seeds，其他 cells 用三个 seeds。但 common hyperparameter table 仍写：

> Random seeds: `{42, 123, 456}`

这会让读者以为所有方法都只有 3 seeds。建议改成：

> Base seeds `{42,123,456}`; five-seed cells add `{789,1024}`; see Table II and Appendix seed paragraph.

这是小问题，但会直接影响 reproducibility perception。

### 4. Statistical reporting 仍然有一点混杂

目前统计口径包括：

- main table: seed mean ± seed sigma，同时给 episode-level Welch CI；
- R2/R4: hierarchical bootstrap；
- E10 table: Best 用 hierarchical bootstrap，Final 用 seed-level Welch；
- ablation: pooled per-episode Welch p-values；
- abstract: seed-level sigma uses `ddof=0`。

这些不一定错，但读者会问：哪个是不应该被引用的“主统计证据”？现在正文已经说 hierarchical bootstrap 更保守，建议再把 Table III caption 或 protocol 口径收紧：

> Seed mean ± sigma is descriptive; hierarchical bootstrap is the primary inferential evidence where reported; pooled episode-level Welch intervals are secondary diagnostics because episodes are nested within training seeds.

另外，E10 table caption 说 Final 的 per-episode returns 没有完整重录，只能做 seed-level Welch；这和 main table 里 Final 又有 episode-level CI 看起来有点冲突。建议明确是 “E10 final per-episode traces for hierarchical paired bootstrap were not retained”，而不是泛泛说 Final 没有 per-episode returns。

### 5. “Best checkpoint is deployable” 的措辞仍可再保守

Metrics 段说：

> Best is the stronger metric from the practitioner's view (one would deploy the best checkpoint)

R1 里也说 CQL 等方法不能 “deployable without online SUMO rollouts to identify a peak”。

这在 benchmark 论文里可以理解，但严格说 Best 是 post-hoc simulator-selected checkpoint；如果没有独立 validation protocol，它不等于可部署 checkpoint。建议把 Best 定义成：

> an oracle/model-selection upper bound under the post-hoc evaluator

然后把 “deploy” 改成 “select for further validation”。这样和 conclusion 里 “stop short of recommending production deployment” 更一致。

### 6. Transportation-side evidence 仍是根本限制，但现在已经诚实承认

这个问题不一定要在当前版本补实验，但需要维持保守定位：

- operational metrics 是 agent-decision aggregates，不是 passenger service quality；
- 没有 passenger waiting time、in-vehicle delay、completed-trip count；
- data-generating heuristic 只有 one deterministic SUMO run；
- 没有 no-holding baseline；
- 没有 classical headway-based holding controller 的 multi-seed 对照；
- dataset calibration against Changsha OD is described, but no calibration error / validation summary is reported。

对 ML benchmark 来说，这些 caveat 可以接受；对交通 venue 来说，这是主要 missing evidence。

### 7. SUMO evaluation seed protocol 最好在 artifact 里显式列出

论文现在写 “10 distinct `--seed` arguments to SUMO, fixed across method/seed/ckpt comparisons”。这比之前清楚很多。提交 artifact 时建议 README 或 appendix 直接列出这 10 个 SUMO seeds，或者在 eval scripts 里给出固定 seed list。这样可以避免 reviewer 担心 “same reset sequence” 只是文字描述而不是可复现设置。

## Minor Polish

- `Appendix B-0a` 必修；这是肉眼可见的排版瑕疵。
- `statistically borderline` for WSRL-E10 Final 改掉；`p=0.65` 应该叫 indistinguishable。
- `Random seeds` table row 改成 base/five-seed split。
- `Best` metric 不要说 “one would deploy the best checkpoint”；说 model-selection upper bound 更稳。
- Table VI caption 解释清楚为什么 Final 不能做 hierarchical bootstrap。
- 结论里当前 conservative claim 写得比较好，别再加强成 categorical dominance。

## Bottom Line

这版已经把上一轮最危险的问题基本修掉了：参考文献、stale conclusion、checkpoint accounting、E10 architecture confound、single-seed SAC overclaim 都处理到可接受水平。

我的审稿判断会从上一轮的 “Minor Revision” 上调到 **Weak Accept after minor polishing**。提交前我建议只做一轮轻量清理：修掉 `Appendix B-0a`、WSRL-E10 Final 的 borderline 误述、seed 表格旧口径，以及 Best/deployment wording。科学主线现在已经站得住。
