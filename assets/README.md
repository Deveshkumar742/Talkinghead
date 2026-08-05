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
- **Talk naturally through the whole take.** Read anything at all — the content
  is irrelevant, because the mouth region gets repainted. What matters is that
  your jaw and lips are *already moving*.

  > **This reverses earlier guidance in this file.** It originally said "mouth
  > closed, do not speak," reasoning that existing speech motion would fight the
  > model. That was wrong for this model family. LatentSync has a documented
  > failure mode — *lip shape leakage*, where the source mouth shape bleeds into
  > the output — and its own demo assets are people speaking. A rigidly closed
  > source mouth appears to bias it toward staying closed, which showed up as a
  > mouth that barely opened. Run `notebooks/diagnose_lipsync.ipynb` to confirm
  > on your own footage before re-recording.

- Natural head movement and blinks throughout; stillness reads as uncanny
- Even, soft, frontal lighting; no harsh shadows across the face
- Plain background, nothing moving behind you
- Hands out of frame, hair off the face, watch for glasses glare
- Start and end in a similar pose so it loops cleanly. If it does not, the
  pipeline falls back to ping-pong looping (forward then reversed), which hides
  the seam at the cost of one visibly reversed motion.
- Keep the audio track in your recording. The pipeline discards it, but having
  it makes the file useful for other purposes later.

## `latentsync-weights` — a second, optional dataset

Not a recording, but it belongs in the same place mentally. The LatentSync
checkpoints are ~9.8 GB and re-downloading them costs 8–12 minutes of *every*
Kaggle session — the largest single component of render time.

`notebooks/diagnose_lipsync.ipynb` has a final cell that bundles them into
`checkpoints.tar`. Upload that once as a private dataset named
`latentsync-weights`, attach it alongside this one, and
`LatentSync(cache_dir=...)` will link the weights instead of fetching them.

## `reference.wav`

The voice Chatterbox clones. Recorded once.

- 15–20 seconds of clean, natural-paced speech
- Same microphone you would normally use
- Mono, 24 kHz or higher
- No background noise, no music, no room echo
- Read something neutral and conversational — matching the delivery you want in
  the output, since Chatterbox copies prosody as well as timbre.
