# Music, sound effects and video-to-audio from FastVideo

Samples for a DreamVerse Harness feature proposal: Stable Audio Open (text to music or sound effects) and MMAudio (video to audio), served by FastVideo's OpenAI-compatible server and called with `POST /v1/videos/sync`, which returns `audio/wav`.

| Sample | Model | Input | Output |
|---|---|---|---|
| [`score_rider`](samples/score_rider.mix.mp4) | Stable Audio Open 1.0 | Prompt (`samples/score_rider.request.json`) | 11 s score; `.mix.mp4` mixes it 10 dB under FastH3's own sound, `.music_only.mp4` replaces that sound |
| [`sfx_skate`](samples/sfx_skate.wav) | Stable Audio Open 1.0 | Prompt (`samples/sfx_skate.request.json`) | 6 s sound effect |
| [`v2a_rider`](samples/v2a_rider.mp4) | MMAudio large-44k-v2 | A FastH3 OmniRef clip with its sound removed, and a short prompt | 5 s sound for the picture; compare with [FastH3's own sound](samples/v2a_rider.fasth3_original.mp4) |
| [`v2a_skate`](samples/v2a_skate.mp4) | MMAudio large-44k-v2 | Same, skateboard clip | 5 s sound for the picture; compare with [FastH3's own sound](samples/v2a_skate.fasth3_original.mp4) |

`samples/spectra.png` and `samples/v2a_spectra.png` are spectrograms of the generated audio and of FastH3's own sound for the same pictures.

## Reproduce

- FastVideo `a04639fa9`, one RTX PRO 6000 (driver 580, CUDA 13). Both servers fit on one GPU together (about 16 GB).
- Stable Audio: `FastVideo/stable-audio-open-1.0-Diffusers`, plus `t5-base`, which its conditioner loads by name. Install `k_diffusion einops_exts alias_free_torch torchsde` (see FastVideo's `examples/inference/basic/basic_stable_audio.py`).
- MMAudio: FastVideo's converted repositories are not public, so the weights come from `hkchengrex/MMAudio` (`weights/mmaudio_large_44k_v2.pth`, `ext_weights/v1-44.pth`, `ext_weights/synchformer_state_dict.pth`), `apple/DFN5B-CLIP-ViT-H-14-384` and `nvidia/bigvgan_v2_44khz_128band_512x`, converted with FastVideo's `scripts/checkpoint_conversion/convert_mmaudio_to_diffusers.py --variant large_44k_v2` (needs `open_clip_torch`). The MMAudio weights are licensed CC-BY-NC-4.0.
- `serve.sh <stable-audio|mmaudio> <gpu> <port>` starts a server from `serve-*.yaml` (FastVideo's `examples/serving/openai_stable_audio.yaml` and `openai_mmaudio.yaml` with local paths); `make_samples.py` sends the requests.

| Server | Request | Time on one RTX PRO 6000 |
|---|---|---|
| Stable Audio Open | 11 s, 100 steps | 54 s |
| Stable Audio Open | 6 s, 100 steps | 7 s |
| MMAudio large-44k-v2 | 5 s from a 5 s clip, 25 steps | 12–20 s |
