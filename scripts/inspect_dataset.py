"""
One-off utility: download the smallest task archive from the real wrop
training corpus, extract a couple of samples, and print out exactly what
trajectory.npz and metadata.json contain -- so wrop_eval/schema.py can be
written against the real files instead of assumptions.

"""
import json
import tarfile
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Hokin/object-permanence"
DATA_DIR = Path("data")


def pick_smallest_task_tar():
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    tar_files = [f for f in files if f.endswith(".tar")]
    print(f"Found {len(tar_files)} task archives.")
    infos = api.get_paths_info(REPO_ID, tar_files, repo_type="dataset")
    infos.sort(key=lambda i: i.size)
    smallest = infos[0]
    print(f"Smallest: {smallest.path}  ({smallest.size / 1e6:.1f} MB)")
    return smallest.path


def main():
    DATA_DIR.mkdir(exist_ok=True)
    tar_path_in_repo = pick_smallest_task_tar()

    local_tar = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename=tar_path_in_repo, local_dir=str(DATA_DIR),
    )
    print(f"Downloaded to {local_tar}")

    extract_dir = DATA_DIR / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with tarfile.open(local_tar) as tf:
        members = tf.getmembers()
        # Only pull out the first 2 sample folders' worth of files, not all 10,000.
        first_two_dirs = sorted({m.name.split("/")[2] for m in members
                                  if len(m.name.split("/")) > 3})[:2]
        print(f"Extracting samples: {first_two_dirs}")
        wanted = [m for m in members
                  if len(m.name.split("/")) > 3 and m.name.split("/")[2] in first_two_dirs]
        tf.extractall(extract_dir, members=wanted)

    sample_dirs = sorted(extract_dir.glob("*/*/*"))
    sample_dirs = [d for d in sample_dirs if d.is_dir()]
    print(f"\nExtracted sample dirs: {sample_dirs}")

    sample = sample_dirs[0]
    print(f"\n=== Files in {sample} ===")
    for f in sorted(sample.iterdir()):
        print(" ", f.name, f.stat().st_size, "bytes")

    npz_path = sample / "trajectory.npz"
    if npz_path.exists():
        print(f"\n=== trajectory.npz keys ===")
        with np.load(npz_path, allow_pickle=True) as npz:
            for k in npz.files:
                arr = npz[k]
                print(f"  {k}: shape={getattr(arr, 'shape', None)} dtype={arr.dtype}")

    meta_path = sample / "metadata.json"
    if meta_path.exists():
        print(f"\n=== metadata.json ===")
        print(json.dumps(json.load(open(meta_path)), indent=2)[:3000])


if __name__ == "__main__":
    main()