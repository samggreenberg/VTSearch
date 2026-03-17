"""Audio waveform generation utilities."""

import io
import math
import struct
import wave

GENERATOR_SAMPLE_RATE = 48000


def generate_wav(frequency: float, duration: float) -> bytes:
    """Generate a mono PCM WAV file containing a pure sine-wave tone.

    Produces a single-channel, 16-bit signed PCM WAV at 48 kHz. The amplitude
    is fixed at 50 % of the maximum value (32767) to avoid clipping.

    Args:
        frequency: Frequency of the sine wave in Hz (e.g. 440.0 for concert A).
        duration: Length of the tone in seconds.

    Returns:
        A ``bytes`` object containing a valid WAV file that can be written
        directly to disk or streamed as ``audio/wav``.
    """
    num_samples = int(GENERATOR_SAMPLE_RATE * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(GENERATOR_SAMPLE_RATE)
        samples = []
        for i in range(num_samples):
            t = i / GENERATOR_SAMPLE_RATE
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))
    return buf.getvalue()
