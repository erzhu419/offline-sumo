Since the source has no line numbers, I cite by section/table/figure/equation.

## Overall assessment

This is a promising applied offline/offline-to-online RL benchmark paper with real strengths: the all-12-line control setting is valuable, the SUMO scale is credible for T-ITS/TR-C, the release of code/data/checkpoints is unusually strong, and the post-hoc checkpoint evaluation protocol is a good contribution. The paper is much more convincing after correcting the earlier “Line 7X only” issue.

However, I would not recommend acceptance in the current form. The main blockers are not “top-tier ML” issues; they are venue-appropriate concerns about fair baselines, transportation relevance, reproducibility, and overclaiming. The most serious problems are:

1. The main conclusion “offline RE-SAC beats offline-to-online” is confounded by architecture: RE-SAC uses a 10-head ensemble, while SAC/WSRL/RLPD are reported as twin-Q methods. The ablation shows the ensemble itself is the dominant factor.
2. The paper reports only shaped RL return, with no operational transit metrics such as headway deviation, bunching rate, passenger waiting time, holding time, travel time, or per-line effects.
3. Important protocol details are internally inconsistent: checkpoint interval, checkpoint counts, SUMO seed control, embedding dimensions, evaluation time, and some claims about variance.
4. The paper overstates statistical significance and causal explanations, especially given the claimed per-episode return standard deviation of roughly 2000.
5. Several claims are unsupported by presented evidence, especially the “30% overstatement” of during-training best metrics and the “budget does not matter” conclusion from a single 600-epoch SAC seed.

My recommendation would be **Major Revision**. The work is salvageable and potentially useful, but the current manuscript needs substantial cleanup and at least a small number of targeted additional evaluations.

---

# 1. Factual errors and internal inconsistencies

## 1.1 Station embedding dimension contradicts the MDP/network description

- **Sec. 3.2** says the observation has five categorical features: line, bus, station, time-period, direction.
- **Sec. 3.1/Table I/Fig. 1** state there are 353 per-line bus stops.
- **Appendix B** states: `station (1 -> 2)` and `time-period (1 -> 2)`.

This is a serious inconsistency. If the station embedding table has cardinality 1, the policy does **not** condition on station identity, despite the MDP claiming it does. If this is a documentation error, fix it. If this is the actual implementation, the paper must say so, because it materially changes the interpretation of the multi-line policy.

The parameter-count statement in Appendix B also appears inconsistent with the listed embedding sizes. Please provide exact embedding cardinalities and parameter counts from the released model.

## 1.2 Checkpoint interval and checkpoint counts are inconsistent

- **Sec. 4, Training** says offline checkpoints are saved every **5K** steps.
- **Appendix A/Table IV** says checkpoint interval is **10K** for offline methods.
- **Sec. 5.4** says ablation checkpoints are saved every **10K** steps.
- **Sec. 4, Evaluation** claims 349 evaluated checkpoints with a breakdown of 151 main, 66 BC/H2O+ intermediate, 99 ablation, 33 CQL.

The checkpoint accounting does not add up cleanly. In particular, “151 main comparison” is not divisible by three seeds. If all runs use three seeds and a fixed checkpoint schedule, most checkpoint counts should be multiples of 3. Please include a checkpoint inventory table: method, seeds, checkpoint epochs/steps, number of checkpoints, and whether the initial checkpoint is included.

## 1.3 SUMO seed control is contradictory

- **Sec. 4, Evaluation** says checkpoints are evaluated on 10 independent SUMO episodes “with reset seed varied.”
- **Appendix C, Seeds** says “The SUMO simulator seed is governed internally … and is not set by our code.”

These statements conflict. For a reproducible benchmark, the SUMO seed should be explicitly controlled and logged. Ideally, use common random numbers across methods/checkpoints for paired comparisons. If SUMO seeds were not controlled, the paper should not claim fully reproducible post-hoc evaluation.

## 1.4 Main-table variance claim is false

In **Sec. 5, R2**, the paper says RE-SAC’s seed variance “is also the smallest.” This is not true in **Table II**.

For Best return:

- WSRL: \(-4587 \pm 61\)
- RLPD \(\rho=0.75\): \(-4597 \pm 110\)
- RE-SAC: \(-4267 \pm 140\)

For Final return:

- RLPD \(\rho=0.75\): \(-4689 \pm 48\)
- RE-SAC: \(-4520 \pm 269\)

RE-SAC has the best mean return, but not the smallest across-seed standard deviation in the main table. Also, saying \(\sigma=140\) is “an order of magnitude below BC’s” is inaccurate; it is about 4.7x smaller than BC’s Best std and 3.7x smaller than BC’s Final std.

## 1.5 “Lower-is-better for |return|” is confusing

**Table II caption** says “Lower-is-better for \(|\)return\(|\).” The actual return is negative, and the intended interpretation is that **higher / less negative return is better**. Please state this directly. Otherwise readers may misread the table.

## 1.6 The paper credits \(L_1\) after showing \(L_1\) hurts

Several places still imply that both LCB and \(L_1\) explain RE-SAC’s performance:

- **Sec. 5, R2**: “LCB pessimism prevents … while the \(L_1\) weight regularization keeps individual ensemble heads simple…”
- **Fig. 3 caption**: “value of the LCB and \(L_1\) regularizers we introduce.”

But **Table III** shows:

- LCB only: \(-4157 \pm 40\) Best
- Full LCB+\(L_1\): \(-4267 \pm 140\) Best

So \(L_1\) is not helpful in this setting. The narrative should consistently say: the effective method is ensemble + LCB; \(L_1\) is historical/optional and should be dropped for the offline benchmark.

## 1.7 “Doubling SAC budget confirms algorithmic advantage” is too strong

The abstract says doubling SAC from 300 to 600 epochs “confirms” the gap is algorithmic. But **Sec. 5.3** correctly says the 600-epoch run is single-seed and preliminary. The abstract and contributions should be toned down: this is suggestive, not confirmatory.

## 1.8 The “BC ceiling” statement is not quite correct

**Sec. 3.3** says the heuristic gets roughly \(-9800\) and “behavior cloning’s ceiling is therefore near this value by construction.” But **Table II** reports BC Best \(-8440\), substantially better than \(-9800\). The final BC value is near the heuristic, but the “ceiling by construction” claim is not accurate.

At minimum, include the heuristic controller directly in the main table with mean/std.

## 1.9 Equation/Algorithm mismatch for the SAC entropy term

**Eq. (1)** defines the RE-SAC policy loss as:

\[
-\mathbb{E}[\mu_Q+\beta\sigma_Q]+\beta_{\text{bc}}\mathcal{L}_{BC}.
\]

But **Algorithm 1** computes \(\mu_Q,\sigma_Q\) over \(Q_{\theta_i}(s,a^\pi)-\alpha\log\pi(a^\pi|s)\). Since the entropy term affects the mean, the equation should include it, or the algorithm should be revised.

## 1.10 “Bus vehicles per day” vs per episode

**Table I** says “Bus vehicles per day: 389,” while the text repeatedly describes 389 buses entering over a 2.5-hour episode. Use consistent terminology: “scheduled bus trips per episode” or “bus vehicles in the simulated peak window.”

---

# 2. Methodological gaps that block acceptance at this venue tier

## 2.1 Missing transportation baselines

For a T-ITS/TR-C class paper, the main table should include at least:

1. The data-generating heuristic controller.
2. A no-holding / schedule-only / do-nothing baseline.
3. Ideally a classical headway-based holding controller, e.g., a Daganzo/Bartholdi-style rule or the internal heuristic if that is the deployed rule.

BC is not a substitute for the actual heuristic. The paper repeatedly references the heuristic’s \(\sim -9800\) performance but does not report it formally. This is a must-fix.

## 2.2 Only shaped RL return is reported

The paper’s results are entirely based on the shaped reward. This is not enough for a transportation journal. Readers need to know whether the learned policy improves actual service quality.

Please report at least some of the following for the main methods:

- Mean absolute headway deviation.
- Headway coefficient of variation.
- Bunching rate, e.g., fraction of arrivals with headway below a threshold.
- Large-gap rate.
- Average holding time.
- Distribution of holding times.
- Passenger waiting time.
- Passenger in-vehicle delay / travel time.
- Number of completed stop arrivals or completed trips.
- Per-line performance, especially because one policy controls all 12 lines.

This is particularly important because the reward has no explicit holding-time penalty. A policy might improve the reward by holding aggressively while harming passengers. The paper needs to rule out such behavior.

## 2.3 Return may be confounded by variable number of decision events

The episode return appears to be the sum of rewards over asynchronous bus-stop arrivals. But policies can affect the number of arrivals within the fixed 2.5-hour episode through holding and speed actions. If one policy produces fewer decision events, its total return may be less negative for reasons unrelated to better control.

Please report:

- Number of decision transitions per episode by method.
- Average reward per decision.
- Operational metrics normalized per passenger, per bus, or per stop arrival.

Without this, aggregate return is hard to interpret.

## 2.4 Main comparison is architecture-confounded

RE-SAC uses a 10-head ensemble. According to **Appendix A/Table V**, SAC, WSRL, RLPD, and CQL use twin-Q critics. But **Table III** shows that even “Ensemble only” achieves \(-4303\), better than all offline-to-online methods.

This means the headline result may be:

> A 10-head ensemble critic beats twin-Q offline-to-online baselines,

not necessarily:

> Pure offline RL beats offline-to-online RL.

This is a serious fairness issue. To support the main claim, the paper should either:

- Add architecture-matched offline-to-online baselines, e.g., RLPD/WSRL with the same 10-head ensemble or a REDQ-style critic; or
- Reframe the claim more narrowly: “our ensemble offline method outperforms our twin-Q implementations of WSRL/RLPD.”

Given that RLPD commonly uses high UTD and, in many implementations, critic ensembles/layer normalization, the current RLPD baseline may not be faithful enough to support strong claims against RLPD.

## 2.5 RLPD implementation may not match the cited method

**Sec. 4** describes RLPD as random initialization plus offline/online data mixing. **Appendix A** lists UTD \(=1\) and twin-Q critics. The original RLPD recipe is not just arbitrary replay mixing; its performance is tied to implementation details such as high update-to-data ratio, critic stabilization, and often ensembles.

If the paper uses a simplified “SAC + offline replay mixing” variant, that is acceptable, but it should be named as such. If it is called RLPD, the authors should justify which key RLPD ingredients are included and which are omitted.

## 2.6 Statistical uncertainty is not handled carefully enough

The paper says per-evaluation episode returns have standard deviation around 2000, yet **Table II** reports only standard deviation across three seed-level means. This can create a misleading sense of precision.

Three training seeds are acceptable for this venue, but the paper should report uncertainty honestly. Recommended:

- For each method, report mean over seed-level means and std across seeds, as currently done.
- Also report episode-level variability or confidence intervals using hierarchical bootstrap over seeds and evaluation episodes.
- For key claims, e.g., RE-SAC vs RLPD \(\rho=0.5\), report paired confidence intervals if common evaluation seeds are used.

The current gap between RE-SAC Best \(-4267\) and RLPD \(\rho=0.5\) Best \(-4552\) is about 285. Given the stated episode-level std of \(\sim 2000\), this is not obviously statistically robust unless pairing or hierarchical analysis supports it.

## 2.7 The “best checkpoint” metric is still not deployable for offline RL

The paper correctly criticizes during-training 2-episode best metrics. But it still reports “Best” selected using post-hoc simulator evaluation. For pure offline RL, this is also not available in a real deployment without simulator evaluation.

This is not fatal, because this is a simulation benchmark. But the paper should be clearer:

- “Best” is a retrospective benchmark metric.
- “Final” or a pre-specified checkpoint is closer to a deployable offline policy.
- The claim that RE-SAC is deployable should rely mostly on its stable final/plateau behavior, not its post-hoc Best score.

## 2.8 Feature computation may involve future-information leakage

**Sec. 3.2** includes features such as “predicted boarding queue” and “predicted travel time to next stop.” These may be valid real-time estimates, but the paper must explain how they are computed.

Please clarify whether these features use:

- Current observable state only;
- Historical/statistical predictors;
- SUMO future ground truth.

If any feature uses realized future travel time or future passenger arrivals, the benchmark is not a valid online decision problem.

## 2.9 Simulation calibration is asserted but not documented

**Sec. 3.1** says the network was calibrated against real topology and passenger OD survey/card data. For a transportation venue, this needs at least a short validation summary:

- What real data were used?
- What quantities were matched?
- Were travel times, passenger counts, or headways validated?
- What is the calibration error?

If detailed calibration is outside the paper scope, use more modest wording such as “constructed from” rather than “calibrated against.”

---

# 3. Narrative and writing issues

## 3.1 The title says “matches,” but the abstract/conclusion say “outperforms”

The title is appropriately cautious. The abstract and conclusion repeatedly say RE-SAC “outperforms” offline-to-online methods. Given the statistical uncertainty and architecture confound, “matches or exceeds in our benchmark” is safer.

## 3.2 “Naive offline methods” is unfair to CQL

CQL is not a naive offline method; it is a standard conservative offline RL baseline. If CQL performs worse here, say that. But avoid wording that suggests CQL lacks pessimism or is inherently naive.

## 3.3 Causal explanations are stronger than the evidence

Examples:

- **Sec. 5.1/Fig. 3 caption** attributes degradation to extrapolation error.
- **Sec. 5.3** attributes SAC oscillation to value-function instability.
- **Sec. 5, R2** says LCB prevents Q explosion.

These are plausible, but the paper does not show Q-value diagnostics, uncertainty calibration, OOD action analysis, or value traces. Phrase these as interpretations unless diagnostics are added.

## 3.4 The “30% overstatement” claim is not supported in the results

The abstract and contributions claim that the common 2-episode during-training “best” metric can overstate performance by up to 30%. But no table or figure actually shows this comparison.

This is a good contribution if substantiated. Add a table:

- Method
- During-training 2-episode best
- Post-hoc N=10 value of same checkpoint
- Post-hoc best checkpoint
- Relative overstatement

If you do not include this evidence, remove or soften the claim.

## 3.5 H2O+ offline should be renamed or clarified

The described “H2O+ offline” baseline is essentially IQL/AWR-style offline learning. If the dynamics discriminator / trust mechanism from H2O+ is not used, call it:

- “IQL/AWR”
- “H2O+-style AWR”
- “AWR baseline from our prior H2O+ implementation”

Calling it H2O+ may mislead readers.

## 3.6 The \(L_1\) term should not be called “aleatoric” without stronger support

A weight-norm penalty is not, by itself, an aleatoric uncertainty model. Since the cited RE-SAC prior work is “in preparation,” this terminology is not adequately supported. Rename it “weight-norm regularization” unless you can justify the aleatoric interpretation.

## 3.7 The abstract is too dense

The abstract is long and contains many numerical claims. For IEEE style, consider shortening it and moving detailed numbers into the results section. Keep only the key benchmark scale, methods, protocol, and main conclusion.

---

# 4. Things that would strengthen the paper without requiring new experiments

Assuming the logs/checkpoints already exist:

1. **Add hierarchical confidence intervals** using existing per-episode post-hoc returns.
2. **Add the missing during-training vs post-hoc comparison table** to support the “30% overstatement” claim.
3. **Provide a checkpoint manifest** resolving the 349-checkpoint count.
4. **Fix all protocol inconsistencies**: 5K vs 10K, 5h vs 6h, online update counts, online epochs, checkpoint inclusion.
5. **Clarify evaluation action selection**: deterministic policy mean or stochastic sampling? This matters for SAC/BC.
6. **Clarify SUMO seed handling** and list the exact evaluation seeds if available.
7. **Clarify all observation features** and explicitly state that no future ground truth is used.
8. **Reframe the main claim** to “matches or exceeds tested offline-to-online baselines” rather than definitive superiority.
9. **Move \(L_1\)-heavy discussion to the appendix** and present ensemble+LCB as the recommended method.
10. **Add dataset diagnostics** from the HDF5 file: per-line transition counts, action distributions, reward/headway distributions.
11. **Remove or replace the unpublished/in-preparation citation** as evidence for key claims.
12. **Report exact implementation details for WSRL/RLPD/CQL**, including whether actor/critic/entropy temperature are warm-started, replay buffer reset behavior, and online gradient updates per episode.

---

# 5. New experiments ranked by acceptance lift per cost

## Rank 1 — Add transportation baselines and operational metrics  
**Lift:** Very high  
**Cost:** Low to moderate  

Evaluate at least the data-generating heuristic and no-hold/schedule-only controller under the same N=10 protocol. For selected learned policies, report headway deviation, bunching rate, holding time, passenger waiting time, passenger travel time, and per-line metrics.

This is the most important addition for T-ITS/TR-C.

## Rank 2 — Re-evaluate final/top checkpoints with controlled common SUMO seeds  
**Lift:** High  
**Cost:** Low to moderate  

You do not need to re-evaluate all 349 checkpoints. Re-evaluate the final and best/selected checkpoints of the main methods using the same 20–30 evaluation seeds. This would support paired comparisons and reduce noise.

## Rank 3 — Add an architecture-matched offline-to-online baseline  
**Lift:** High  
**Cost:** Moderate to high  

Run RLPD-style data mixing with the same 10-head critic/REDQ-style ensemble used by RE-SAC, or implement RLPD closer to the original recipe. This addresses the biggest fairness concern.

If compute is limited, run only RLPD \(\rho=0.5\) with the ensemble, since that is the strongest offline-to-online baseline in the current table.

## Rank 4 — Small CQL sensitivity check  
**Lift:** Medium  
**Cost:** Low to moderate  

CQL with \(\alpha_{\text{cql}}=1.0\) is acceptable as a default, but because reward scale is task-specific, a small sweep such as \(\{0.1, 1.0, 5.0\}\) would make the CQL comparison more credible. This is not an exhaustive sweep; it is a sanity check.

## Rank 5 — Use LCB-only as the main reported method  
**Lift:** Medium  
**Cost:** Very low if already trained  

Since Table III shows LCB-only is better than full LCB+\(L_1\), the paper should either make LCB-only the main “ours” result or clearly say the main table uses an older full variant and the final recommended method is LCB-only.

## Rank 6 — Domain-shift evaluation  
**Lift:** Medium to high  
**Cost:** Moderate  

Evaluate on altered demand/traffic levels or a held-out scenario file. This would support the opening motivation about generalization to new traffic conditions. Not required for acceptance if the claims are toned down, but valuable.

## Rank 7 — Three-seed 600-epoch SAC  
**Lift:** Low to medium  
**Cost:** Moderate  

This is only needed if the authors want to keep the strong “budget does not matter” claim. Easier alternative: keep the single-seed result but explicitly label it preliminary.

---

# 6. Section-by-section comments

## Title

“Matches offline-to-online fine-tuning” is appropriately cautious. However, the body often claims “outperforms.” Align the narrative with the title unless stronger statistical evidence and architecture-matched baselines are added.

## Abstract

Strong but overcompressed. Main issues:

- “Generalize … to new traffic conditions” is not directly tested.
- “CQL … degrades at convergence” should be “at the end of training budget.”
- “Doubling SAC’s training budget … confirming” is too strong for a single seed.
- The 30% overstatement claim needs a result table.
- Consider saying “matches or exceeds tested offline-to-online baselines” rather than “best overall” as a definitive claim.

## Introduction

The motivation is good. The central question is important for transit RL. However:

- “We answer negatively” is too strong given the architecture confound.
- The introduction should mention that evaluation is in the same calibrated SUMO environment as data collection, not under true domain shift.
- The contributions should be rewritten after deciding whether LCB-only or full RE-SAC is the actual proposed method.

## Related Work

Adequate but thin for a transportation journal. Consider adding more recent bus holding / headway control work and possibly conventional control baselines. Also, avoid relying on an “in preparation” citation for key RE-SAC claims.

## Problem Setup

The SUMO description is detailed and useful. Needed fixes:

- Clarify whether 353 stops are physical stops or per-line stop instances.
- Fix “bus vehicles per day.”
- Document calibration/validation.
- Explain all “predicted” observation features and rule out future leakage.
- Clarify whether speed multipliers can exceed legal/lane speed limits or are constrained by SUMO.
- Explain how returns are normalized, if at all, given variable numbers of bus-stop events.

## Offline Data

Good that the dataset size and heuristic are described. But the heuristic should appear in the main results table. Also add dataset coverage diagnostics: transitions per line, action distributions, reward/headway distributions.

## Methods

BC, CQL, SAC, WSRL, RLPD, and RE-SAC are mostly described, but baseline fidelity needs work.

- Clarify whether H2O+ offline is actually IQL/AWR.
- Specify WSRL warm-start details: actor only or actor+critic? entropy coefficient? replay buffer reset?
- Clarify whether RLPD is faithful to the cited algorithm or a simplified replay-mixing variant.
- Fix the RE-SAC equation/algorithm entropy mismatch.
- Rename \(L_1\) as weight-norm regularization rather than aleatoric uncertainty unless justified.
- Since \(L_1\) hurts, do not foreground it as a core method contribution.

## Experimental Protocol

This section needs the most cleanup.

- Resolve checkpoint interval inconsistencies.
- Explain online update counts precisely.
- Explain how many online decision transitions are collected per epoch.
- Clarify evaluation policy mode: deterministic or stochastic.
- Control and report SUMO seeds.
- Explain how “Best” is selected and why “Final” is more relevant for offline deployment.

## Results

The main results are interesting but overclaimed.

- Add heuristic/no-hold/classical baselines.
- Add operational metrics.
- Add confidence intervals or statistical tests.
- Correct the false variance claim.
- Avoid saying CQL/BC/H2O+ are useless; CQL final still improves substantially over the heuristic.
- Avoid causal claims about extrapolation/value instability without diagnostics.

## RLPD Ablation

Useful. But if RLPD itself is simplified, label it as an offline-ratio ablation for SAC+offline replay rather than a definitive RLPD study.

## Extended SAC

Keep this as a preliminary observation only. Do not use it as evidence that budget cannot close the gap unless more seeds are run.

## RE-SAC Ablation

This is one of the strongest parts of the paper. It is honest and useful. But it changes the story:

- The dominant factor appears to be the ensemble architecture.
- LCB gives a smaller additional gain.
- \(L_1\) is unnecessary or harmful.

The main method description and conclusion should reflect this.

## Discussion

Good discussion of same-environment evaluation and domain shift. This should be made more prominent earlier. Also add limitations about shaped reward, missing operational metrics, and simulator calibration.

## Appendix / Reproducibility

The release is a major strength. But fix:

- Station embedding cardinality.
- Checkpoint interval.
- SUMO seed handling.
- Compute time inconsistency.
- “Shared embeddings with separate clones” wording.
- Exact package/SUMO command and commit hash.

---

# Mock journal review

## Summary

The paper presents a SUMO benchmark for offline and offline-to-online reinforcement learning in multi-line bus holding control. A single embedding-conditioned policy controls 12 bus lines and 389 buses using 3.1M offline transitions collected under a heuristic controller. The authors compare BC, H2O+/AWR, CQL, SAC, WSRL, RLPD, and a proposed ensemble-based offline SAC variant with lower-confidence-bound pessimism. They introduce a post-hoc evaluation protocol over saved checkpoints and report that their offline ensemble method matches or exceeds tested offline-to-online baselines while using no online interaction.

## Strengths

- Important and practical problem: offline RL for bus holding is highly relevant.
- Whole-network 12-line setting is more realistic than single-line toy benchmarks.
- Code/data/checkpoint release is a major reproducibility strength.
- Inclusion of BC, CQL, SAC, WSRL, and RLPD-style baselines is directionally appropriate.
- Post-hoc checkpoint evaluation is a valuable protocol contribution.
- The ablation is unusually honest: it shows \(L_1\) is not helpful and LCB-only is better.
- Three seeds are acceptable for this venue if uncertainty is reported carefully.

## Weaknesses

- No formal comparison to transportation baselines such as the data-generating heuristic, no-hold control, or classical headway-based holding.
- Results report only shaped RL return, not operational transit metrics.
- Main comparison is architecture-confounded: RE-SAC uses a 10-head ensemble, while offline-to-online baselines use twin-Q critics.
- RLPD implementation may be a simplified replay-mixing variant rather than the full cited method.
- Several internal inconsistencies: checkpoint interval, checkpoint counts, SUMO seed handling, embedding sizes, evaluation time, variance claims.
- Statistical uncertainty is underreported relative to the stated high episode-level variance.
- Several claims are too strong, especially “outperforms,” “confirms,” and causal explanations of overfitting/instability.
- The “30% overstatement” claim is not supported by a presented result.
- The simulator calibration and feature construction are insufficiently documented.

## Required Revisions, must-fix

1. Add the data-generating heuristic and at least one simple transportation baseline to the main results.
2. Report operational metrics beyond shaped return, including at minimum headway deviation, bunching rate, holding time, and per-line performance.
3. Resolve all protocol inconsistencies: checkpoint interval, checkpoint counts, SUMO seed control, online update counts, evaluation action mode, and embedding dimensions.
4. Provide proper uncertainty reporting using existing N=10 episode evaluations, preferably hierarchical confidence intervals or paired comparisons.
5. Correct the false variance claim in Sec. 5 and temper superiority claims.
6. Either implement a more faithful/architecture-matched RLPD baseline or rename the current method as a simplified SAC+offline-replay baseline.
7. Clarify feature computation and rule out future-information leakage.
8. Add evidence for the “30% overstatement” claim or remove it.
9. Reframe RE-SAC around the actual ablation result: ensemble + LCB is the useful method; \(L_1\) is dispensable.
10. Include at least a concise simulator calibration/validation description or soften the calibration claims.

## Recommended Revisions, nice-to-have

1. Add a small CQL \(\alpha_{\text{cql}}\) sensitivity check.
2. Use LCB-only as the main proposed method if it is consistently better.
3. Add domain-shift evaluation under different traffic/passenger demand.
4. Add Q-value/uncertainty diagnostics if making causal claims about extrapolation error.
5. Run the 600-epoch SAC study for three seeds only if the budget-related claim remains central.
6. Shorten the abstract and reduce numerical clutter.
7. Replace or remove the in-preparation citation as support for methodological claims.

## Editorial Recommendation

**Major Revision**

The paper has a strong applied benchmark idea and unusually good artifact release, but the current version has several acceptance-blocking issues in baseline fairness, transportation evaluation, reproducibility, and claim calibration. I would encourage resubmission after targeted revisions rather than rejection.

## Confidence

**4 / 5**