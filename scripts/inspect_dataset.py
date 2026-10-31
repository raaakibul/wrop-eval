"""
Look at the REAL WROP file layout without downloading a whole archive.

The training corpus is one big .tar per task (hundreds of MB each). This
script streams the first tar from Hugging Face and stops as soon as it has
read two complete samples, so only a few MB are transferred. It then prints
what is really inside: file names, trajectory.npz arrays, metadata.json,
and the frame count / resolution of each video.

Run:  python scripts/inspect_dataset.py
Then paste the whole output back so the loaders can be fixed to match.
"""
import json
import os
import tarfile

import numpy as np

REPO_ID = "Hokin/object-permanence"
OUT_DIR = os.path.join("data", "inspect")


def extract_first_samples(fileobj, out_dir, n_samples=2, show_first=15):
    """Stream a tar from `fileobj`, extract the first `n_samples` directories
    of files, and stop. Returns (list_of_member_names_seen, [extracted dirs])."""
    os.makedirs(out_dir, exist_ok=True)
    seen_names, chosen = [], []
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:  # streaming mode
        for member in tf:
            if len(seen_names) < show_first:
                seen_names.append(member.name)
            if not member.isfile():
                continue
            folder = os.path.dirname(member.name)
            if folder not in chosen:
                if len(chosen) >= n_samples:
                    break  # a third folder started: we have enough
                chosen.append(folder)
            tf.extract(member, out_dir)
    return seen_names, [os.path.join(out_dir, c) for c in chosen]


def describe_array(name, arr):
    line = f"  {name}: shape={getattr(arr, 'shape', None)} dtype={arr.dtype}"
    print(line)
    if arr.dtype == object:
        flat = arr.ravel()
        print(f"      object array, first element type: {type(flat[0]).__name__}")
        print(f"      first element (truncated): {str(flat[0])[:300]}")
    elif arr.size <= 12:
        print(f"      values: {arr.tolist()}")
    else:
        print(f"      first row/value: {np.asarray(arr).reshape(arr.shape[0], -1)[0][:6].tolist()}")


def describe_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"  {os.path.basename(path)}: {n} frames, {w}x{h}, {fps:.1f} fps")


def describe_sample(sample_dir):
    print(f"\n=== {sample_dir} ===")
    for fname in sorted(os.listdir(sample_dir)):
        full = os.path.join(sample_dir, fname)
        print(f"  file: {fname}  ({os.path.getsize(full)} bytes)")
    for fname in sorted(os.listdir(sample_dir)):
        full = os.path.join(sample_dir, fname)
        if fname.endswith(".npz"):
            print(f"\n--- {fname} arrays ---")
            with np.load(full, allow_pickle=True) as npz:
                for k in npz.files:
                    describe_array(k, npz[k])
        elif fname.endswith(".json"):
            print(f"\n--- {fname} ---")
            with open(full) as f:
                data = json.load(f)
            print(json.dumps(data, indent=2)[:2500])
        elif fname.endswith(".txt"):
            print(f"\n--- {fname} (first 300 chars) ---")
            print(open(full, encoding="utf-8", errors="replace").read()[:300])
        elif fname.endswith(".mp4"):
            print(f"\n--- {fname} ---")
            describe_video(full)


def main():
    from huggingface_hub import HfApi, HfFileSystem

    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    tars = sorted(f for f in files if f.endswith(".tar"))
    print(f"Found {len(tars)} .tar files. Streaming the first: {tars[0]}")

    fs = HfFileSystem()
    with fs.open(f"datasets/{REPO_ID}/{tars[0]}", "rb") as f:
        names, dirs = extract_first_samples(f, OUT_DIR)

    print("\nFirst entries in the tar (shows the folder structure):")
    for n in names:
        print("  ", n)
    for d in dirs:
        describe_sample(d)


if __name__ == "__main__":
    main()