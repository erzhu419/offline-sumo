# GPT Review Round 3

## Recommendation

**Minor Revision for a benchmark / applied ML venue; Major Revision only if targeting a transportation journal that requires passenger-service validation.**

这一轮比 Round 2 又明显进了一步。参考文献已经正常，checkpoint inventory 补上了，630 checkpoints 的 accounting 基本闭合，station slot / future-information leakage / decision-count confound / WSRL-E10 significance 这些问题也都有明确回应。现在稿件的主实验故事已经比较可信。

我现在不再认为核心算法实验是主要 blocker。剩下的问题主要是：少量 stale text、appendix/section formatting、结论段没有同步到当前更谨慎的 R4 叙述，以及交通运营层面的证据仍然不足以支持强部署 claim。

## What Is Now Fixed

1. **References are fixed.** 当前 PDF citation 已经是正常编号，不再是 `[?]`。
2. **Checkpoint accounting mostly closes.** Table II 给出 19 cells、71 runs、630 evaluated checkpoints；这和 `eval_results.csv` 564 rows + `eval_E10.csv` 36 rows + `eval_noBC.csv` 30 rows 对上。
3. **Architecture confound is handled much better.** WSRL-E10 / RLPD-E10 加入后，论文不再只拿 10-head RE-SAC 对 twin-Q baselines。
4. **E10 statistical conclusion更诚实。** 当前 R4 明确承认 RE-SAC vs WSRL-E10 Best 的 95% CI crosses zero，Final 也 statistically indistinguishable。
5. **Information leakage concern is addressed.** Observation section 解释了 predicted boarding queue 和 travel time estimate 不使用 future realization。
6. **Decision-count confound is addressed.** Operational table 加了 Dec./ep 和 R/dec。
7. **BC ceiling wording corrected.** 现在说 heuristic 是 behavior-policy reference，不再说 BC ceiling。

## Remaining Issues

### 1. Appendix / cross-reference formatting is currently broken

This is the most visible remaining presentation issue.

The PDF renders:

- “Appendix 0a” in the Observation paragraph.
- “Sec. VII-0c” when referring to limitations.
- Appendix sections are not formatted as real appendix sections.

The log still contains:

```text
WARNING: Ignoring useless \section in Appendix
```

This is not a scientific flaw, but it looks unpolished and may annoy reviewers. The appendix labels should be rendered as normal Appendix A/B/C or avoided entirely. Also, Table numbering becomes awkward because the checkpoint inventory is Table II before the main result table, making the main result table Table III; that is acceptable, but less natural than making the main comparison table earlier or moving the inventory to appendix.

### 2. Conclusion is stale and contradicts the updated R4 interpretation

The conclusion still says:

- “pure offline twin-Q ... exceeds all offline-to-online baselines (`-4444` vs best OtO best return `-4626` for WSRL)”
- “LCB ... adds another `Δ = 146`, both significant at `p <= 0.01`”
- practitioners should consider “deploying offline RL directly”

These are no longer aligned with the current paper:

- WSRL-E10 best is `-4365`, so `-4626` is not the best offline-to-online result anymore.
- Current ablation text says LCB adds `Δ = +112`, `p = 0.023`, not `146` and not `p <= 0.01`.
- R4 says RE-SAC vs WSRL-E10 best is not significant at 95%, and final is statistically indistinguishable.
- The operational section explicitly says the results should not be read as direct deployment superiority.

The conclusion should be rewritten to match the current conservative claim:

> RE-SAC clearly beats the standard twin-Q baselines and RLPD-E10; it remains directionally ahead but statistically tied with WSRL-E10 under the present 3-vs-5 seed comparison. The main empirical lesson is that offline actor-critic + ensemble + LCB is a strong zero-online baseline, not that offline RL categorically dominates all offline-to-online methods.

### 3. Some stale reproducibility text remains

Two examples:

- Appendix common hyperparameter table still lists random seeds only as `{42,123,456}`, even though many key cells use `{42,123,456,789,1024}`.
- Compute environment still says the full set is **349 checkpoints** in roughly 5 hours, while the rest of the paper now says **630 checkpoints**.

These are easy fixes but important because Round 2 specifically criticized checkpoint accounting.

### 4. Statistical reporting is improved, but still mixed in a confusing way

The paper uses several uncertainty conventions:

- Table III reports seed mean ± seed sigma and episode-level Welch CI.
- R2/R4 use hierarchical bootstrap.
- Table VI uses hierarchical bootstrap for Best but seed-level Welch for Final.
- Some ablation p-values use pooled per-episode Welch tests.
- Abstract says seed-level sigma uses `ddof=0` to match figures.

None of these are individually wrong, but the paper should make the hierarchy clearer. The safest phrasing is:

- Main claims should cite hierarchical bootstrap where available.
- Seed-level mean ± sigma is descriptive.
- Episode-level Welch CI should not be presented as the primary inferential uncertainty because episodes are nested within seeds/checkpoints.

Right now Table III’s 95% CI column may still look more precise than the seed-level evidence supports.

### 5. Extended online SAC claim remains too strong in contributions

The abstract says the 600-epoch SAC experiment “suggests” the gap is not closed, which is fine. But the contribution bullet says it “demonstrates pure online RL cannot close the gap”.

Because this is single-seed, the bullet should be softened:

> A preliminary extended-budget online SAC study suggests that simply doubling the online SAC budget does not reliably close the gap.

### 6. Operational evidence is still limited for transportation venues

The paper now handles this honestly in Table IV and limitations, but it remains a real limitation:

- Bunching rate is 0 for all methods.
- Large-gap rate differs only weakly.
- Mean headway deviations are very close across methods.
- No passenger waiting time, in-vehicle delay, completed trips.
- No multi-seed heuristic, no-holding, or classical headway-based controller.

For an applied ML / benchmark paper, this is acceptable if framed as a shaped-reward benchmark. For a transportation journal, it is still a major limitation.

### 7. Main claim should consistently distinguish standard baselines from E10 baselines

The title “matches offline-to-online fine-tuning” is now appropriate. But some intro wording still says “none surpass RE-SAC’s pure-offline performance” and “we answer negatively” to whether offline-to-online adds value.

Given WSRL-E10 is close and statistically tied on Final, the safer framing is:

> In this benchmark, a strong pure-offline learner matches or exceeds the tested offline-to-online baselines, including architecture-matched variants, while using zero online rollouts. The present evidence does not show a reliable advantage for offline-to-online fine-tuning over RE-SAC, but WSRL-E10 narrows the gap.

This preserves the paper’s point without overclaiming.

## Minor Issues

- Figure/paragraph references like “Sec. VII-0c” are visually awkward; use unnumbered limitation paragraph references or a normal subsection.
- The operational table header renders oddly in PDF as `Holdg 90`; consider replacing `\widetilde{\text{Hold}}_{90}` with `Hold p90`.
- Methods says “We compare seven methods,” but Table III has more rows including E10 variants and ratio ablations. Say “method families” or update the count.
- RLPD/WSRL are described early as state-of-the-art approaches, but later E10 variants are explicitly “our baseline variants” rather than full reimplementations. This distinction should appear earlier.
- “The reward signal is dominated by sporadic bunching events” conflicts slightly with operational metrics where bunching rate is 0 under the reported threshold. Maybe say “large headway deviations/gaps” instead.

## Bottom Line

这版已经基本把 Round 2 的主要科学问题处理了。我的当前判断是：**主实验结论可接受，剩余主要是表述、格式和交通外部有效性问题。**

如果目标是 ML / simulation benchmark venue，我会给 **Weak Accept after Minor Revision**。如果目标是 T-ITS/TR-C 这类交通 venue，仍需要补 passenger-level metrics 和 classical/no-holding baselines，否则只能作为 RL benchmark，而不是完整交通运营改进论文。

最该立刻修的是：appendix/cross-reference formatting、conclusion stale numbers、349 checkpoint stale text、extended SAC wording，以及把 final claim 改成“matches or directionally exceeds, with WSRL-E10 statistically tied at current seed budget”。
