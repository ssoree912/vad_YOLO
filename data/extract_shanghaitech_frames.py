import cv2
import os
from pathlib import Path

# Allow overriding via environment variables if needed.
VIDEOS_ROOT = Path(os.environ.get("SH_VIDEOS_DIR", "shanghaitech/training/videos"))
FRAMES_ROOT = Path(os.environ.get("SH_FRAMES_DIR", "shanghaitech/training/frames"))

FRAMES_ROOT.mkdir(parents=True, exist_ok=True)

films = []
if not VIDEOS_ROOT.exists():
    raise FileNotFoundError(f"Video directory not found: {VIDEOS_ROOT}")

for file in sorted(VIDEOS_ROOT.iterdir()):
    if file.is_file():
        print(file.stem, "is a file!")
        films.append(file)

for film in films:
    vidcap = cv2.VideoCapture(str(film))
    if not vidcap.isOpened():
        print(f"[warn] failed to open video: {film}")
        continue

    success, image = vidcap.read()
    video_frame_dir = FRAMES_ROOT / film.stem
    video_frame_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    while success:
        frame_path = video_frame_dir / f"{count}.jpg"
        cv2.imwrite(str(frame_path), image)
        success, image = vidcap.read()
        count += 1
