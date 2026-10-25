# Wrop-eval

Trajectory-grounded, category-decomposed automatic evaluation for object permanence in generative world models. Built on the WROP corpus (https://object-permanence.world).

Work in progress 

# WROP-Eval

An automatic, trajectory-grounded way to score **object permanence** in generated video, built on the [WROP](https://object-permanence.world) dataset.

A video model is shown a clip in which an object moves behind something. The model then generates what happens next. WROP-Eval checks the generated video against the exact ground-truth trajectories that ship with WROP, and reports an **Object Permanence Score (OPS)** per sample, per cognitive category, and per model.

> **Status: early research code.**
> The full pipeline runs end to end and is tested on synthetic data. It has **not yet been run on real WROP data**, so no number from this repo says anything about a real model yet. See [Limitations](#limitations) for exactly what is and isn't verified.



## Background

WROP ("World Reasoning with Object Permanence") provides 150 Blender-rendered tasks, a large training corpus, a 300-question exam, and a leaderboard of video models. Per the dataset card, every sample has a 60-frame input clip, a 60-frame target clip, and a `trajectory.npz` with per-frame ground truth for the objects in the scene, including ones hidden behind occluders. The project site ranks models on a leaderboard built from human comparisons.

Human comparison is slow and expensive to repeat for every new model or checkpoint. The ground-truth trajectories, meanwhile, are sitting in every sample. This project asks whether they can be used to score a generated video automatically.

## Related work and the gap

We searched for related work before building this. It is a limited search, not a systematic review.

- **Violation-of-expectation benchmarks** such as [IntPhys 2](https://arxiv.org/abs/2506.09849) test whether a model can tell a physically possible clip from an impossible one. That is a judging task, not scoring a model's own generated continuation.
- **Physical-plausibility benchmarks for generated video**, such as [T2VPhysBench](https://arxiv.org/abs/2505.00337), [PhyGround](https://arxiv.org/abs/2605.10806), and [Physion-Eval](https://arxiv.org/abs/2603.19607), score generated video with human raters or learned judges. They don't compare against exact ground-truth 3D trajectories.
- **[Morpheus](https://arxiv.org/abs/2504.02918)** is the closest work we found. It tracks objects in generated video and scores them against physical laws, using real recorded experiments. As we read it, it also cautions that comparing a generated trajectory to a single recorded one can unfairly punish a different-but-valid outcome when the model is only weakly conditioned.

**The gap, to our knowledge:** 
we did not find an automatic metric that combines (1) exact per-frame ground truth, (2) a task set organized by type of permanence reasoning, and (3) a scoring rule built to avoid penalizing valid alternative outcomes. We also checked WROP's own repository. Its `audit.py` validates that the dataset rendered correctly. We found no code there that scores a model's generated video against the shipped trajectories.

If you know of prior work that already does this, please open an issue. That would change how this project should be framed.


## How it works

For each sample, the pipeline tracks objects in the generated video and compares them with the ground truth.

**1. Hard checks** run on every sample. They don't depend on which valid continuation the model chose.

| Check | Meaning |
|---|---|
| `disappearance_rate` | The target object is never seen again in the final part of the clip. |
| `identity_swap_rate` | The object that comes back has the wrong color. |
| `hallucinated_object_rate` | The tracker found an object that matches nothing in the ground truth. |

**2. Soft checks** measure how far the object is from where it should be. They run only on samples that pass a **gate**.

- **Gate.** A constant-acceleration model is fit to the object's ground-truth motion during the 60 input frames. If it fits closely, the future is predictable from the context, so position error is fair to measure. If the motion is not that simple (a collision, a sudden change), we skip the soft checks rather than punish a model for choosing a different valid outcome.
- **Alignment.** Ground truth is in Blender world coordinates and the tracker works in image coordinates, and we don't have the camera parameters. So we fit a scale-and-shift transform between the two, using **only the input frames**. The continuation is then judged against that fixed transform. Fitting on all frames would let the fit absorb an offset the model introduced, hiding about half of a constant error in our tests.
- **Measures.** `reappearance_position_error` (distance at the first frame the object is seen again), plus average and final displacement over the end of the clip.

**3. Combining.** Each rate is between 0 and 1. Soft errors are squashed into 0 to 1 with `1 - exp(-error / 0.15)`. Then:

```
hard = mean(disappearance, identity_swap, hallucinated)
soft = mean(squashed RPE, ADE, FDE)        # 0 if the sample was not gated
OPS  = 1 - (1.0 * hard + 0.5 * soft) / 1.5
```

These weights are simple choices, not tuned values. They are the first thing to revisit once real data is available.

**4. Reporting.** Scores are averaged per task, per cognitive category, and per model, and a validation script correlates per-model OPS with human leaderboard scores.