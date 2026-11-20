"""Join metrics for run_eval.py outputs: how smoothly shot B continues shot A at the cut.

Per joined video (A frames, then B's new frames), at the join j (first B frame):
  jump_ratio    mean |F[j] - F[j-1]| over the median adjacent-frame difference in the 2 s around the join (1.0 = no jump)
  ssim_join     SSIM of F[j-1] and F[j]
  flow_cos      cosine of the mean optical flow over A's last 3 frame pairs and B's first 3 (camera and motion direction)
  flow_ratio    magnitude of that mean flow after the join over before it (speed continuity)
  replay_k      k in 1..72 maximizing SSIM(F[j], F[j-k]); 1 means B starts after A's last frame, larger means B restarts
                from an earlier moment of A
  audio_db      RMS level in the 0.5 s after the join minus the 0.5 s before it

Usage: uv run --with opencv-python-headless --with numpy --with scikit-image python metrics.py runs/<name>
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

FPS = 24
WIDTH = 320
REPLAY_WINDOW = 72


def frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            return out
        h, w = frame.shape[:2]
        out.append(cv2.resize(frame, (WIDTH, round(h * WIDTH / w)), interpolation=cv2.INTER_AREA))


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def mean_flow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    flow = cv2.calcOpticalFlowFarneback(gray(a), gray(b), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow.reshape(-1, 2).mean(axis=0)


def audio(path: Path) -> np.ndarray:
    wav = path.with_suffix(".mono.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", str(wav)],
                   check=True)
    with wave.open(str(wav)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    wav.unlink()
    return data


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-8))


def join_metrics(joined: Path, a_frames: int) -> dict:
    f = frames(joined)
    j = a_frames
    # Blur first so codec noise between separately encoded clips does not count as motion.
    smooth = [cv2.GaussianBlur(x, (0, 0), 2).astype(np.float32) for x in f]
    diffs = [float(np.abs(smooth[t + 1] - smooth[t]).mean()) for t in range(j - FPS, min(j + FPS, len(f) - 1))]
    at_join = diffs[FPS - 1]
    baseline = float(np.median(diffs[:FPS - 1] + diffs[FPS:]))
    before = np.mean([mean_flow(f[t], f[t + 1]) for t in range(j - 4, j - 1)], axis=0)
    after = np.mean([mean_flow(f[t], f[t + 1]) for t in range(j, j + 3)], axis=0)
    norm = np.linalg.norm(before) * np.linalg.norm(after)
    ssim = [structural_similarity(gray(f[j]), gray(f[j - k])) for k in range(1, REPLAY_WINDOW + 1)]
    sound = audio(joined)
    cut = round(j / FPS * 16000)
    half = 8000
    return {
        "jump_ratio": round(at_join / baseline, 2),
        "ssim_join": round(ssim[0], 3),
        "flow_cos": round(float(before @ after / norm), 2) if norm > 1e-6 else None,
        "flow_ratio": round(float(np.linalg.norm(after) / np.linalg.norm(before)), 2) if np.linalg.norm(before) > 1e-6 else None,
        "replay_k": int(np.argmax(ssim)) + 1,
        "replay_ssim": round(float(max(ssim)), 3),
        "audio_db": round(rms_db(sound[cut:cut + half]) - rms_db(sound[cut - half:cut]), 1),
    }


def contact_sheet(rows: list[tuple[str, Path]], a_frames: int, out: Path) -> None:
    """Frames j-3..j+2 of each joined video, one row per arm; a red line marks the cut."""
    tiles = []
    for label, path in rows:
        f = frames(path)
        strip = np.concatenate([f[t] for t in range(a_frames - 3, a_frames + 3)], axis=1)
        x = strip.shape[1] // 2
        cv2.line(strip, (x, 0), (x, strip.shape[0]), (0, 0, 255), 2)
        bar = np.full((28, strip.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, strip], axis=0))
    cv2.imwrite(str(out), np.concatenate(tiles, axis=0))


def join_gif(rows: list[tuple[str, Path]], a_frames: int, out: Path) -> None:
    """The second before and after the cut, arms side by side; a red border marks frames from shot B."""
    clips = [(label, frames(path)) for label, path in rows]
    tmp = out.with_suffix("")
    tmp.mkdir(exist_ok=True)
    for i, t in enumerate(range(a_frames - FPS, a_frames + FPS)):
        tiles = []
        for label, f in clips:
            tile = f[t].copy()
            if t >= a_frames:
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255), 4)
            bar = np.full((24, tile.shape[1], 3), 255, np.uint8)
            cv2.putText(bar, f"{label}  {'B' if t >= a_frames else 'A'} {t - a_frames:+d}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(np.concatenate([bar, tile], axis=0))
        cv2.imwrite(str(tmp / f"{i:03d}.png"), np.concatenate(tiles, axis=1))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS // 2), "-i", str(tmp / "%03d.png"),
                    "-vf", "split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer",
                    "-loop", "0", str(out)], check=True)
    for png in tmp.iterdir():
        png.unlink()
    tmp.rmdir()


def main() -> None:
    root = Path(sys.argv[1])
    results = []
    for scene in sorted(p for p in root.iterdir() if (p / "shot_a.mp4").exists()):
        a_frames = len(frames(scene / "shot_a.mp4"))
        sheet_rows = []
        for joined in sorted(scene.glob("*/seed*_joined.mp4")):
            arm, seed = joined.parent.name, joined.name.split("_")[0]
            row = {"scene": scene.name, "arm": arm, "seed": seed, **join_metrics(joined, a_frames)}
            results.append(row)
            print(json.dumps(row), flush=True)
            if seed == sorted(scene.glob(f"{arm}/seed*_joined.mp4"))[0].name.split("_")[0]:
                sheet_rows.append((arm, joined))
        order = {"last_frame": 0, "tail_2s": 1, "tail_2s_only": 2, "native_22f": 3}
        sheet_rows.sort(key=lambda r: order.get(r[0], 9))
        contact_sheet(sheet_rows, a_frames, scene / "join_sheet.png")
        join_gif(sheet_rows, a_frames, scene / "join.gif")
    (root / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
