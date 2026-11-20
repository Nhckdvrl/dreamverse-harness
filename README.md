# Shot continuation on FastH3 OmniRef

Evidence for [fastvideo-project/dreamverse-harness#154](https://github.com/fastvideo-project/dreamverse-harness/pull/154). Shot B continues shot A four ways; `metrics.py` measures the join.

| Arm | Shot B input |
|---|---|
| `last_frame` | A's last frame as an extra image reference, cited as the first frame |
| `tail_2s` | A's last 2 s, cut with PR #154's ffmpeg command, as a `video_reference`, plus the reference image |
| `tail_2s_only` | `tail_2s` without the reference image |
| `native_22f` | A's last 22 frames as FastVideo's `continuation_video`, plus the reference image; the 22 regenerated frames are dropped |

- Server: FastVideo `5af9897e7`, `fastvideo-kernel` built from that checkout for `12.0a`, `FastVideo/FastH3-OmniRef-NVFP4-QAD` at `f01e6a23`, one RTX PRO 6000, `serve.yaml` and `serve.sh`.
- Renders: `run_v2.sh` runs `run_eval.py` against two servers; `run_eval.py --arms tail_2s_only` adds the fourth arm.
- Metrics: `metrics.py runs/v2` writes `metrics.json`, `join_sheet.png` (frames −3..+2 around the cut, seed 1234) and `join.gif` (±1 s at half speed; a red border marks shot B).
