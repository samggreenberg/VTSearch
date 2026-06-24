/**
 * Decode compressed/encoded audio bytes into an `AudioBuffer` for waveform
 * rendering, using an `OfflineAudioContext`.
 *
 * Why offline rather than a live `AudioContext`: decoding is the only thing we
 * need here (the actual playback runs through a plain `<audio>` element). A live
 * `AudioContext` allocates a hardware audio thread and counts against the
 * browser's per-page cap (~6 in Chrome); since `close()` is async, rapid media
 * navigation could create contexts faster than they tear down and exhaust that
 * budget (`NotSupportedError: number of hardware contexts reached maximum`).
 * `OfflineAudioContext` is purely computational — no hardware allocation, no
 * cap, and nothing to clean up — so callers never have to track its lifecycle.
 *
 * `decodeAudioData` resamples to the context's sample rate; 44.1 kHz mono is
 * fine for peak/waveform display. The 1-frame length is irrelevant because we
 * never call `startRendering()`.
 */
export async function decodeAudioBuffer(data: ArrayBuffer): Promise<AudioBuffer> {
  const OfflineCtx =
    window.OfflineAudioContext ||
    (window as unknown as { webkitOfflineAudioContext: typeof OfflineAudioContext }).webkitOfflineAudioContext;
  const ctx = new OfflineCtx(1, 1, 44100);
  return ctx.decodeAudioData(data);
}
