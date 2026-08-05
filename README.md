# talkinghead

Type a script, get back a 1080p video of yourself delivering it.

**Cloud-first by design.** Every compute stage runs in a Kaggle notebook session.
Nothing needs installing on your own machine — Kaggle provides FFmpeg
preinstalled and a free GPU. Your laptop's job is to edit scripts and watch
finished videos.

## How it runs

| Stage | Where | Cost |
|---|---|---|
| 1. script prep | Kaggle CPU | free, unmetered |
| 2. TTS (Chatterbox, cloned voice) | Kaggle CPU or GPU | free on CPU |
| 3. audio assembly (FFmpeg) | Kaggle CPU | free, unmetered |
| 4. fit base loop to narration (FFmpeg) | Kaggle CPU | free, unmetered |
| 5. lipsync (LatentSync 1.6) | Kaggle **GPU** | 30 hr/week quota |
| 6. mux + encode (FFmpeg) | Kaggle CPU | free, unmetered |

Only stage 5 draws against the GPU quota. Kaggle CPU sessions are unmetered, so
iterate on the voice for free and switch to a GPU session only when the narration
is final.

## Getting started

1. **Record two things** — see [`assets/README.md`](assets/README.md) for the
   specs that actually matter (mid shot, mouth closed, don't speak).
   - `base_loop.mp4` — 60–90s silent 1080p loop of yourself
   - `reference.wav` — 15–20s of clean speech
2. **Upload them as a private Kaggle Dataset** named `talkinghead-assets`.
   Private is not optional: it is your likeness and your voice.
3. **Open `notebooks/kaggle_render.ipynb`** in a Kaggle session, attach the
   dataset via *+ Add Input*, and run the cells.

## Local use (optional)

You do not need a local setup to use this project. If you want one for editing
scripts or running the test suite:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

FFmpeg is the only local prerequisite, and only for the media tests — they skip
cleanly without it. The full suite runs in a cloud session.

```bash
talkinghead host       # what does this environment provide?
talkinghead check      # are the assets valid for the active profile?
talkinghead prep s.txt # how will this script be segmented?
talkinghead licenses   # what may and may not be used here
```

## Model licensing

This produces client-facing work, so every model's **weights** — not just its
code — must permit commercial use. Run `talkinghead licenses` for the current
position. Summary:

**Approved:** LatentSync 1.6 (Apache-2.0), MuseTalk (MIT), Chatterbox (MIT),
GFPGAN (Apache-2.0). All pulled from first-party repos, pinned by revision.

**Rejected** — these are the popular defaults, and all four are unusable here:

| Model | Why not |
|---|---|
| Wav2Lip | Weights trained on LRS2; commercial use strictly prohibited |
| CodeFormer | NTU S-Lab License 1.0, non-commercial |
| XTTS-v2 | Coqui CPML, non-commercial |
| VibeVoice | Microsoft scopes it to research, not commercial deployment |

The licensing gate in `config.py` turns a mistake here into an import-time crash
rather than a problem discovered after a video ships.

## Provenance

This is your own face and voice, which is the clean consent case. Output is still
marked so it cannot be mistaken for unedited footage: container metadata on every
render, an optional visible label, and Chatterbox's built-in PerTh watermarking
in the audio.
