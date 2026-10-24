# Wrop-eval

Trajectory-grounded, category-decomposed automatic evaluation for object permanence in generative world models. Built on the WROP corpus (https://object-permanence.world).

Work in progress 

# WROP-Eval

An automatic, trajectory-grounded way to score **object permanence** in generated video, built on the [WROP](https://object-permanence.world) dataset.

A video model is shown a clip in which an object moves behind something. The model then generates what happens next. WROP-Eval checks the generated video against the exact ground-truth trajectories that ship with WROP, and reports an **Object Permanence Score (OPS)** per sample, per cognitive category, and per model.

> **Status: early research code.**
> The full pipeline runs end to end and is tested on synthetic data. It has **not yet been run on real WROP data**, so no number from this repo says anything about a real model yet. See [Limitations](#limitations) for exactly what is and isn't verified.