"""Music, sound-effect and video-to-audio samples from FastVideo's Stable Audio Open and MMAudio servers.

The video inputs are FastH3 OmniRef clips from ~/projects/dv-continuation-eval/runs/v2. Each sample writes the
request (data URLs omitted), the generated WAV, and, where it has a picture, an MP4 with the new sound.

Usage: python make_samples.py --stable-audio http://127.0.0.1:8131 --mmaudio http://127.0.0.1:8132 [--only NAME ...]
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLIPS = Path.home() / "projects/dv-continuation-eval/runs/v2"
OUT = ROOT / "samples"

SAMPLES = {
    "score_rider": {
        "server": "stable-audio",
        "body": {"prompt": "Slow western film score, fingerpicked acoustic guitar and soft harmonica over a warm string "
                           "pad, 80 BPM, golden-hour mood, no vocals, cinematic",
                 "audio_end_in_s": 11.0, "num_inference_steps": 100, "guidance_scale": 7.0, "seed": 1},
        "video": CLIPS / "rider/native_22f/seed1234_joined.mp4",
        "mux": "score",
    },
    "sfx_skate": {
        "server": "stable-audio",
        "body": {"prompt": "Skateboard wheels rolling on smooth concrete, a sharp ollie pop, then a landing clack, "
                           "outdoor skate park ambience",
                 "audio_end_in_s": 6.0, "num_inference_steps": 100, "guidance_scale": 7.0, "seed": 1},
    },
    "v2a_skate": {
        "server": "mmaudio",
        "body": {"prompt": "skateboarding in an outdoor skate park, wheels on concrete", "negative_prompt": "music",
                 "audio_end_in_s": 5.0, "num_inference_steps": 25, "guidance_scale": 4.5, "seed": 42},
        "video": CLIPS / "skate/shot_a.mp4",
        "mux": "replace",
    },
    "v2a_rider": {
        "server": "mmaudio",
        "body": {"prompt": "a horse cantering on dry ground, hooves, wind", "negative_prompt": "music",
                 "audio_end_in_s": 5.0, "num_inference_steps": 25, "guidance_scale": 4.5, "seed": 42},
        "video": CLIPS / "rider/shot_a.mp4",
        "mux": "replace",
    },
}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def silent(video: Path, out: Path) -> Path:
    """The picture of `video` without its sound, so MMAudio hears nothing from FastH3's own track."""
    ffmpeg("-i", str(video), "-an", "-c:v", "copy", str(out))
    return out


def data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def render(server: str, body: dict, out: Path) -> dict:
    shown = {k: ("<data url omitted>" if k == "video_reference" else v) for k, v in body.items()}
    out.with_suffix(".request.json").write_text(json.dumps(shown, indent=2))
    req = urllib.request.Request(f"{server}/v1/videos/sync", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out.write_bytes(resp.read())
        meta = {"wall_s": round(time.time() - started, 1), "content_type": resp.headers.get("Content-Type"),
                "inference_s": resp.headers.get("X-Inference-Time-S")}
    out.with_suffix(".meta.json").write_text(json.dumps(meta))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable-audio", default="http://127.0.0.1:8131")
    parser.add_argument("--mmaudio", default="http://127.0.0.1:8132")
    parser.add_argument("--only", nargs="+", default=list(SAMPLES))
    args = parser.parse_args()
    servers = {"stable-audio": args.stable_audio, "mmaudio": args.mmaudio}
    OUT.mkdir(exist_ok=True)
    for name in args.only:
        spec = SAMPLES[name]
        body = dict(spec["body"])
        if spec["server"] == "mmaudio":
            picture = silent(spec["video"], OUT / f"{name}.silent.mp4")
            body["video_reference"] = {"video_url": data_url(picture, "video/mp4")}
        wav = OUT / f"{name}.wav"
        meta = render(servers[spec["server"]], body, wav)
        print(name, meta, flush=True)
        if spec.get("mux") == "replace":
            ffmpeg("-i", str(OUT / f"{name}.silent.mp4"), "-i", str(wav), "-map", "0:v", "-map", "1:a",
                   "-c:v", "copy", "-c:a", "aac", "-shortest", str(OUT / f"{name}.mp4"))
            ffmpeg("-i", str(spec["video"]), "-c", "copy", str(OUT / f"{name}.fasth3_original.mp4"))
        elif spec.get("mux") == "score":
            # Music alone, and music mixed 10 dB under FastH3's own sound.
            ffmpeg("-i", str(spec["video"]), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                   "-c:a", "aac", "-shortest", str(OUT / f"{name}.music_only.mp4"))
            ffmpeg("-i", str(spec["video"]), "-i", str(wav), "-filter_complex",
                   "[1:a]volume=-10dB[m];[0:a][m]amix=inputs=2:duration=first:normalize=0[a]",
                   "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", str(OUT / f"{name}.mix.mp4"))


if __name__ == "__main__":
    main()
