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



## Quickstart

Requires Python 3.10+.

```bash
git clone ..... 
cd wrop-eval
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Run the tests**

```bash
python -m pytest tests -q
```

**Run the synthetic demo.** It renders tiny videos of four made-up "models" (one correct, three with a typical failure), scores them, and compares the scores to a *mock* Elo ranking.

```bash
python -m wrop_eval.demo --out-dir demo_out
```

Expected output (the exact digits can vary slightly):

```
correct          OPS=0.990   Elo=1650
position_drift   OPS=0.799   Elo=1520
identity_swap    OPS=0.767   Elo=1400
disappear        OPS=0.778   Elo=1280
Spearman r = 0.800
```

The Elo values are invented for the demo, so this only shows that the machinery runs. It says nothing about real models. Charts and per-model reports are written to `demo_out/`.

**Score a model on a benchmark-style folder.** Build a tiny fake benchmark, then run the command-line scorer on it:

```bash
python scripts/make_fake_benchmark.py
python -m wrop_eval.cli --eval-root test_out/fake/eval --results-root test_out/fake/results --model good_model
python -m wrop_eval.cli --eval-root test_out/fake/eval --results-root test_out/fake/results --model vanish_model
```

The first model should score about 0.997. The second, where the ball never comes back, should show `Disappearance 1.000` and score about 0.778.

**Expected folder layout** for real data:

```
eval/<shard>/<task>_task/<sample_id>/scene_state_graph.json   (or trajectory.npz)
results/<model_name>/<task>_task/<sample_id>.mp4
```

**Correlate with human scores.** With one JSON file of per-model OPS and one of per-model Elo, run the following (it needs at least 3 models):

```bash
python -m wrop_eval.validate ops_per_model.json elo_per_model.json
```

## Project layout

```
wrop_eval/
  constants.py   clip geometry from the dataset card (60 input + 60 target frames)
  schema.py      ground-truth data classes and loaders
  taxonomy.py    task name -> cognitive category (a keyword heuristic, see Limitations)
  colors.py      color name -> RGB, for the identity check
  align.py       scale-and-shift alignment between world and image coordinates
  tracker.py     baseline tracker, plus a synthetic video generator for tests
  metrics.py     the scoring logic (gating, hard checks, soft checks, OPS)
  aggregate.py   per-task / per-category / per-model reports
  validate.py    correlation of OPS with human scores
  cli.py         score a model over a benchmark-style folder
  demo.py        end-to-end synthetic demonstration
scripts/
  inspect_dataset.py       download one small task archive and print its real file layout
  make_fake_benchmark.py   build a tiny fake benchmark for trying the CLI
tests/                     unit and end-to-end tests
```


## Limitations

### What is verified, and what is not

| Verified | Not verified |
|---|---|
| The pipeline runs end to end on synthetic videos. | Anything on real WROP data. |
| Each failure mode (vanishing, wrong color, position error) is detected on synthetic cases, and the ranking correct > drift > swap/vanish holds. | Whether OPS agrees with human judgments of real models. The demo's "Elo" is invented. |
| The alignment fit no longer hides a constant continuation offset (unit-tested). | That the alignment model is adequate for WROP's real cameras. |
| Clip geometry (60 + 60 frames, 24 fps) matches the dataset card. | The real layout of `trajectory.npz`. |

The demo's correlation with a mock ranking only shows the machinery works. It is not evidence that the metric is valid.


### Known weaknesses and assumptions

**Data format**
- The `trajectory.npz` loader guesses at array names. It has only been tested on files we made ourselves. If your real files differ, patch `_from_npz_dict` in `schema.py`.
- We assume the model's output video covers all 120 frames, so its first 60 frames can be used to fit the alignment. If real result videos contain only the 60 continuation frames, `cli.py` and the alignment step must also track the input video.
- The six cognitive categories come from a keyword heuristic over task names (`taxonomy.py`), not from the dataset authors' labels. Per-category results are only as meaningful as that mapping.

**Scoring**
- **The disappearance check assumes the target object is visible again in the last 20% of the clip.** That is wrong for tasks where an object stays hidden on purpose, such as one left inside a closed box. On those tasks a correct model would be penalized. This needs a per-task or per-sample "should be visible" annotation before the numbers can be trusted.
- Identity is judged by average color alone. Two objects of the same color cannot be told apart, and shading or lighting in real renders will shift observed colors. The threshold (70 in RGB distance) was chosen from a synthetic example and needs calibrating on real frames.
- Any predicted track that matches no ground-truth object counts as a hallucination. Tracker noise, background clutter, and one object split into two tracks all inflate this.
- Only objects flagged as targets are scored. If none are flagged, the first object is used.
- The gate tolerance, the soft-error scale (0.15), and the 1.0 / 0.5 hard / soft weights are untuned defaults. The gate tolerance was set on synthetic data with coordinates between 0 and 1. Real world coordinates in Blender units will behave differently.

**Alignment**
- It assumes a fixed camera that looks roughly side-on, so the world x axis and the height axis map to image x and y. The depth axis is ignored, and perspective is not modeled. Objects that barely move during the context cannot be aligned, so those samples get no soft scores.

**Tracking**
- The bundled tracker uses background subtraction and nearest-neighbor matching. It is for testing the pipeline. On photorealistic renders with shadows, clutter, or several similar objects it will fail, and a stronger tracker (SAM2, Co-Tracker) should replace it before any real study. Swapping it in only requires matching the `track_with_color` interface.

**Claims**
- The gap described in "Related work" rests on a limited search. Check for prior work before claiming novelty.
- "OPS" is also used for other things in object-permanence research. Consider a more distinctive name before publishing.


## Roadmap

1. Run `scripts/inspect_dataset.py` on real data and fix the `trajectory.npz` loader to match.
2. Find out what the result videos contain (60 or 120 frames) and adapt the pipeline.
3. Add a per-sample "should be visible" signal so tasks with legitimately hidden objects are scored fairly.
4. Replace the baseline tracker with a learned one.
5. Score the models in the benchmark's `results/` folder and correlate per-model OPS with the human leaderboard. This is the experiment that decides whether the metric is useful.
6. Tune thresholds and weights on a held-out set of tasks, not the ones used for the final claim.

## License

WROP's data and code are released under CC BY-NC 4.0, so any use of this project with that data should respect those terms. This repository contains no WROP data.

