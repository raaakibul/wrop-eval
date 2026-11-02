"""
Print the details inspect_dataset.py cut off, from the samples already
extracted into data/inspect (no network needed).

Run:  python scripts/inspect_more.py > inspect_more_output.txt 2>&1
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.join("data", "inspect")
HINTS = ("occlu", "reveal", "visib", "hidden", "cover", "frame", "camera", "split")


def key_paths(obj, prefix=""):
    """Yield every key path in nested JSON, e.g. parameters.seed"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            yield path, v
            yield from key_paths(v, path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            yield from key_paths(v, f"{prefix}[{i}]")


def show_bodies(npz_path):
    with np.load(npz_path, allow_pickle=True) as z:
        names, roles = z["body_names"], z["body_roles"]
        shapes, colors, tgt = z["body_shapes"], z["body_colors"], z["is_target"]
        xpos = z["xpos"]
    print(f"{'#':>2} {'name':<24}{'role':<18}{'shape':<10}{'color':<9}{'target':<7}"
          f"{'moves':>7}  start(xyz) -> end(xyz)")
    for i in range(len(names)):
        move = float(np.linalg.norm(xpos[-1, i] - xpos[0, i]))
        a = np.round(xpos[0, i], 2).tolist()
        b = np.round(xpos[-1, i], 2).tolist()
        print(f"{i:>2} {names[i]:<24}{roles[i]:<18}{shapes[i]:<10}{colors[i]:<9}"
              f"{str(bool(tgt[i])):<7}{move:>7.2f}  {a} -> {b}")
    print(f"\nx range {xpos[..., 0].min():.2f}..{xpos[..., 0].max():.2f} | "
          f"y range {xpos[..., 1].min():.2f}..{xpos[..., 1].max():.2f} | "
          f"z range {xpos[..., 2].min():.2f}..{xpos[..., 2].max():.2f}")


def main():
    sample_dirs = sorted({os.path.dirname(p) for p in
                          glob.glob(os.path.join(ROOT, "**", "trajectory.npz"), recursive=True)})
    if not sample_dirs:
        print("No samples found in data/inspect. Run scripts/inspect_dataset.py first.")
        return
    for d in sample_dirs[:1]:
        print(f"=== {d} ===\n")
        show_bodies(os.path.join(d, "trajectory.npz"))
        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)
        print("\n--- top-level metadata keys ---")
        print(list(meta.keys()))
        print("\n--- metadata.parameters keys ---")
        print(list(meta.get("parameters", {}).keys()))
        print("\n--- key paths that may matter (occlusion / frames / camera) ---")
        for path, value in key_paths(meta):
            if any(h in path.lower() for h in HINTS) and not isinstance(value, (dict, list)):
                print(f"  {path} = {value}")
        print("\n--- full prompt ---")
        print(open(os.path.join(d, "prompt.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()