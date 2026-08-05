# Assets

Both files below are **gitignored on purpose** — they are your likeness and your
voice. Keep them here locally and mirror them to a *private* Kaggle Dataset for
the GPU render stage. Never commit them, and never put them in a public dataset.

## `base_loop.mp4`

Recorded once, reused for every video.

- **1080p**, 25 or 30 fps, camera locked on a tripod (no handheld drift)
- 60–90 seconds long
- **Mid shot** — head and shoulders, face roughly 400–500px tall in frame,
  NOT filling it. LatentSync regenerates the mouth inside a 512px face crop, so
  a face larger than that gets upscaled and goes soft against the sharp frame.
- Seated, facing camera, eyes on the lens
- **Mouth closed / neutral. Do not speak.** Existing speech motion fights the
  lipsync model and produces double-articulation artifacts.
- Gentle idle motion only: small head movements, natural blinks
- Even, soft, frontal lighting; no harsh shadows across the face
- Plain background, nothing moving behind you
- Hands out of frame, hair off the face, watch for glasses glare
- Start and end in a similar pose so it loops cleanly. If it does not, the
  pipeline falls back to ping-pong looping (forward then reversed), which hides
  the seam at the cost of one visibly reversed motion.

## `reference.wav`

The voice Chatterbox clones. Recorded once.

- 15–20 seconds of clean, natural-paced speech
- Same microphone you would normally use
- Mono, 24 kHz or higher
- No background noise, no music, no room echo
- Read something neutral and conversational — matching the delivery you want in
  the output, since Chatterbox copies prosody as well as timbre.
