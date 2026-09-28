"""
constants.py — dataset specs confirmed directly from the Hugging Face
dataset card (https://huggingface.co/datasets/Hokin/object-permanence,
"At a glance" table), not inferred or guessed.

These replace what was previously a per-sample metadata lookup with a
fallback guess (n_input_frames = n_frames // 2) -- the real corpus uses a
FIXED clip geometry for every sample, so we don't need to guess.
"""

RESOLUTION = (1280, 720)
FPS = 24

N_INPUT_FRAMES = 60
N_TARGET_FRAMES = 60     
N_TOTAL_FRAMES = N_INPUT_FRAMES + N_TARGET_FRAMES  

N_TASKS = 150
SAMPLES_PER_TASK_TRAIN = 10_000
SAMPLES_PER_TASK_BENCHMARK = 2


SAMPLE_FILES = [
    "input_video.mp4",
    "target_video.mp4",
    "prompt.txt",
    "trajectory.npz",
    "metadata.json",
]