"""
Probe the model answers in the benchmark repo WITHOUT downloading them all.

Run:  python scripts/inspect_results.py > inspect_results_output.txt 2>&1
"""
import os
from collections import defaultdict

REPO_ID = "Hokin/object-permanence-benchmark"
OUT_DIR = os.path.join("data", "results_probe")
MAX_MB = 30


def group_by_model(paths):
    """{model: {"<task>_task/<sample>.mp4": full_path}} for results/ files."""
    groups = defaultdict(dict)
    for p in paths:
        parts = p.split("/")
        if len(parts) >= 4 and parts[0] == "results" and p.endswith(".mp4"):
            groups[parts[1]]["/".join(parts[2:])] = p
    return dict(groups)


def pick_common_question(groups):
    """The alphabetically-first question that EVERY model answered."""
    common = None
    for answers in groups.values():
        common = set(answers) if common is None else common & set(answers)
    return sorted(common)[0] if common else None


def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return {"frames": n, "width": w, "height": h, "fps": fps,
            "seconds": (n / fps) if fps else 0.0}


def main():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    groups = group_by_model(files)
    print(f"Models found: {len(groups)}")
    for model in sorted(groups):
        print(f"  {model}: {len(groups[model])} answers")

    question = pick_common_question(groups)
    print(f"\nProbing question: {question}\n")
    print(f"{'model':<28}{'frames':>7}{'size':>12}{'fps':>7}{'sec':>7}{'MB':>7}")
    for model in sorted(groups):
        path = groups[model].get(question)
        info = api.get_paths_info(REPO_ID, [path], repo_type="dataset")
        size_mb = (info[0].size or 0) / 1e6
        if size_mb > MAX_MB:
            print(f"{model:<28} skipped ({size_mb:.0f} MB)")
            continue
        local = hf_hub_download(REPO_ID, path, repo_type="dataset", local_dir=OUT_DIR)
        v = video_info(local)
        print(f"{model:<28}{v['frames']:>7}{str(v['width'])+'x'+str(v['height']):>12}"
              f"{v['fps']:>7.1f}{v['seconds']:>7.2f}{size_mb:>7.2f}")


if __name__ == "__main__":
    main()