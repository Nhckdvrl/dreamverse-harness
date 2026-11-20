"""Shot-continuation comparison for fastvideo-project/dreamverse-harness PR #154 on FastH3 OmniRef (Ref2VA).

For each scene, render shot A from one reference image, then render shot B three ways:
  last_frame  - A's last frame as an extra image reference cited as the first frame (DreamVerse `continue_from`).
  tail_2s     - A's last 2 s, cut with PR #154's ffmpeg command, as an ordinary `video_reference` (`continue_video_from`).
  native_22f  - A's last 22 frames as FastVideo's `continuation_video`; the regenerated 22 frames are dropped.
  tail_2s_only - tail_2s without the reference image (optional arm, pass --arms tail_2s_only).

Usage: python run_eval.py --server http://127.0.0.1:8000 --out runs/<name> [--seeds 1234 42] [--scenes skate ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import subprocess
import time
import urllib.request
from pathlib import Path

FPS = 24
NATIVE_FRAMES = 22
# MiniMax-H3 frame counts sit on the causal-VAE grid 17n+5; FastVideo's docs give 158 for a 22-frame clip plus 5 s,
# which leaves 136 new frames after the 22 regenerated ones are dropped.
NATIVE_NUM_FRAMES = 158
TAIL_SECONDS = 2.0
ASSETS = Path.home() / "projects/FastVideo-dv/assets/images"

# Shot B must pick up an action that is mid-flight at the end of shot A.
SCENES = {
    "skate": {
        "image": "mixkit-boy-skating-with-a-skateboard-in-a-park-with-ramps-34389.png",
        "subject": "<Picture 1> shows a boy with a skateboard in a skate park with ramps; keep his face, clothes and board.",
        "a": "[Shot 1] Medium tracking shot. The boy from <Picture 1> pushes off and rolls from left to right across the "
             "concrete park, the camera tracking alongside at his speed. Near the end he crouches low, about to ollie "
             "onto a low ledge. Sound: wheels rolling on concrete, each push scraping, a distant crowd.",
        "b": "[Shot 1] The same medium tracking shot continues. The boy pops the ollie, lands on the ledge, grinds along "
             "it from left to right and rolls away, the camera still tracking right at his speed. Sound: the snap of the "
             "tail, the grind on the ledge, wheels rolling on concrete.",
        "end_state": "the boy crouching low, rolling right, about to ollie onto a low ledge",
    },
    "rider": {
        "image": "mixkit-a-rancher-riding-a-horse-at-sunset-1143.png",
        "subject": "<Picture 1> shows a rancher on a horse at sunset; keep the rider, the horse and the warm sunset light.",
        "a": "[Shot 1] Wide side-on shot. The rancher from <Picture 1> canters his horse from left to right across the "
             "field, the camera panning right to follow. Near the end he raises his right hand to take off his hat. "
             "Sound: hooves drumming on dry ground, the horse breathing, wind.",
        "b": "[Shot 1] The same wide side-on shot continues. The rancher lifts the hat off, waves it above his head and "
             "keeps cantering right, the camera still panning right to follow. Sound: hooves drumming on dry ground, the "
             "horse breathing, wind, the rancher whooping once.",
        "end_state": "the rancher cantering right with his right hand at the brim of his hat",
    },
    "sax": {
        "image": "saxophonist.jpg",
        "subject": "<Picture 1> shows a saxophonist; keep the musician, the saxophone and the stage lighting.",
        "a": "[Shot 1] Slow push-in. The saxophonist from <Picture 1> plays a slow, sustained jazz melody, swaying "
             "slightly; the camera dollies slowly toward him from a medium shot. Near the end he leans back and starts a "
             "long rising note. Sound: a solo tenor saxophone melody, room reverb.",
        "b": "[Shot 1] The same slow push-in continues toward a close-up. The saxophonist holds the long rising note, "
             "then resolves it into the melody and lowers the horn slightly. Sound: the same solo tenor saxophone "
             "melody continuing without a break, room reverb.",
        "end_state": "the saxophonist leaning back at the start of a long rising note, camera mid push-in",
    },
}

# PR #154 adds this instruction to the fasth3 prompt skill for video-tail continuation.
PR154_INSTRUCTION = ("The events in <Video 1> have already happened. Generate only new footage immediately after its "
                     "final frame. At output time 00:00, advance the action while maintaining ongoing motion, camera "
                     "direction and speed, and any ongoing sound. Do not replay even a single reference frame, repeat "
                     "its ending action, freeze on its last frame, or restart the motion or audio.")


def prompt(subjects: list[str], summary: str, detail: str) -> str:
    return "\n".join([
        "subject_definitions:", *subjects,
        "summary:", summary,
        "detailed_description:", detail,
    ])


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def render(server: str, body: dict, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".request.json")).write_text(json.dumps(
        {k: (v if k not in {"image_reference", "video_reference", "continuation_video"} else "<data urls omitted>")
         for k, v in body.items()}, indent=2))
    if out.exists():
        return json.loads(out.with_suffix(".meta.json").read_text())
    req = urllib.request.Request(f"{server}/v1/videos/sync", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        out.write_bytes(resp.read())
        meta = {"wall_s": round(time.time() - started, 1), "inference_s": resp.headers.get("X-Inference-Time-S")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta))
    return meta


def probe(path: Path) -> dict:
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    info = json.loads(raw)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {"duration": float(video.get("duration") or info["format"]["duration"]), "frames": int(video["nb_frames"]),
            "audio": any(s["codec_type"] == "audio" for s in info["streams"])}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def cut_inputs(a: Path, d: Path) -> dict[str, Path]:
    """The three continuation inputs cut from shot A; existing cuts are reused, so parallel runs can share a scene."""
    cuts = {"last_frame": d / "last_frame.png", "tail_2s": d / "tail_2s.mp4", "native_22f": d / "native_22f.mp4"}
    if all(path.exists() for path in cuts.values()):
        return cuts
    info = probe(a)
    last = d / "last_frame.png"
    ffmpeg("-i", str(a), "-vf", f"select=eq(n\\,{info['frames'] - 1})", "-vframes", "1", str(last))
    # PR #154 prepareContinuation: the same argv, with the duration from ffprobe.
    seconds = min(TAIL_SECONDS, info["duration"])
    tail = d / "tail_2s.mp4"
    ffmpeg("-i", str(a), "-ss", str(info["duration"] - seconds), "-t", str(seconds), "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
           str(tail))
    start = (info["frames"] - NATIVE_FRAMES) / FPS
    clip = d / "native_22f.mp4"
    ffmpeg("-i", str(a), "-vf", f"select=gte(n\\,{info['frames'] - NATIVE_FRAMES}),setpts=N/{FPS}/TB",
           "-af", f"atrim=start={start},asetpts=N/SR/TB", "-frames:v", str(NATIVE_FRAMES),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip))
    return {"last_frame": last, "tail_2s": tail, "native_22f": clip}


def drop_head(src: Path, frames: int, dst: Path) -> None:
    ffmpeg("-i", str(src), "-vf", f"select=gte(n\\,{frames}),setpts=N/{FPS}/TB",
           "-af", f"atrim=start={frames / FPS},asetpts=N/SR/TB",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst))


def join(a: Path, b: Path, dst: Path) -> None:
    ffmpeg("-i", str(a), "-i", str(b), "-filter_complex", "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
           "-pix_fmt", "yuv420p", "-c:a", "aac", str(dst))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1234, 42])
    parser.add_argument("--scenes", nargs="+", default=list(SCENES))
    parser.add_argument("--arms", nargs="+", default=["last_frame", "tail_2s", "native_22f"])
    args = parser.parse_args()

    for name in args.scenes:
        scene = SCENES[name]
        d = args.out / name
        image = ASSETS / scene["image"]
        a = d / "shot_a.mp4"
        meta = render(args.server, {
            "prompt": prompt([scene["subject"]], "[full reference] A single continuous shot.", scene["a"]),
            "image_reference": [{"image_url": data_url(image)}], "seed": 7, "num_frames": 124,
        }, a)
        print(name, "shot_a", meta, flush=True)
        cuts = cut_inputs(a, d)
        for arm in args.arms:
            for seed in args.seeds:
                raw = d / arm / f"seed{seed}_raw.mp4"
                if arm == "last_frame":
                    body = {
                        "prompt": prompt(
                            [scene["subject"], f"<Picture 2> is the first frame of [Shot 1]: {scene['end_state']}."],
                            "[keyframe completion] The shot starts exactly from <Picture 2> and continues the action.",
                            "Starting from <Picture 2>. " + scene["b"]),
                        "image_reference": [{"image_url": data_url(image)}, {"image_url": data_url(cuts[arm])}],
                        "num_frames": 124,
                    }
                elif arm == "tail_2s_only":
                    # The minimal PR #154 call: continue_video_from alone satisfies the reference requirement.
                    body = {
                        "prompt": prompt(
                            [f"<Video 1> is the previous clip; it ends with {scene['end_state']}."],
                            "[full reference] Continue <Video 1>. " + PR154_INSTRUCTION,
                            PR154_INSTRUCTION + " " + scene["b"]),
                        "video_reference": [{"video_url": data_url(cuts["tail_2s"])}],
                        "num_frames": 124,
                    }
                elif arm == "tail_2s":
                    body = {
                        "prompt": prompt(
                            [scene["subject"], f"<Video 1> is the previous clip; it ends with {scene['end_state']}."],
                            "[full reference] Continue <Video 1>. " + PR154_INSTRUCTION,
                            PR154_INSTRUCTION + " " + scene["b"]),
                        "image_reference": [{"image_url": data_url(image)}],
                        "video_reference": [{"video_url": data_url(cuts[arm])}],
                        "num_frames": 124,
                    }
                else:
                    body = {
                        "prompt": prompt([scene["subject"]],
                                         f"[full reference] The shot continues from {scene['end_state']}.",
                                         f"Continuing from {scene['end_state']}. " + scene["b"]),
                        "image_reference": [{"image_url": data_url(image)}],
                        "continuation_video": data_url(cuts[arm]),
                        "num_frames": NATIVE_NUM_FRAMES,
                    }
                body["seed"] = seed
                meta = render(args.server, body, raw)
                b = d / arm / f"seed{seed}_new.mp4"
                if arm == "native_22f":
                    drop_head(raw, NATIVE_FRAMES, b)
                else:
                    b.write_bytes(raw.read_bytes())
                join(a, b, d / arm / f"seed{seed}_joined.mp4")
                print(name, arm, seed, meta, probe(raw), flush=True)


if __name__ == "__main__":
    main()
