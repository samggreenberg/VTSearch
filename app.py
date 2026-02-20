import os

# Limit threads to reduce memory overhead in constrained environments
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Visual feedback for startup
print("⏳ Initializing VistaTotes...", flush=True)

import hashlib

print("⏳ Importing ML libraries (this may take a few seconds)...", flush=True)

from flask import Flask

# Import refactored modules
from config import DATA_DIR, NUM_CLIPS
from vistatotes.audio import generate_wav
from vistatotes.models import embed_audio_file, initialize_models
from vistatotes.routes import clips_bp, datasets_bp, exporters_bp, main_bp, sorting_bp
from vistatotes.utils import clips

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Clip generation
# ---------------------------------------------------------------------------


def init_clips():
    print("DEBUG: Generating synthetic waveforms...", flush=True)
    DATA_DIR.mkdir(exist_ok=True)
    temp_path = DATA_DIR / "temp_embed.wav"

    for i in range(1, NUM_CLIPS + 1):
        freq = 200 + (i - 1) * 50  # 200 Hz .. 1150 Hz
        duration = round(1.0 + (i % 5) * 0.5, 1)  # 1.0 – 3.0 s
        wav_bytes = generate_wav(freq, duration)

        # Generate embedding by saving to temp file
        temp_path.write_bytes(wav_bytes)
        embedding = embed_audio_file(temp_path)

        clips[i] = {
            "id": i,
            "type": "audio",
            "frequency": freq,
            "duration": duration,
            "file_size": len(wav_bytes),
            "md5": hashlib.md5(wav_bytes).hexdigest(),
            "embedding": embedding,
            "wav_bytes": wav_bytes,
        }

    # Clean up temp file
    if temp_path.exists():
        temp_path.unlink()


# Model initialization is now handled by vectorytones.models.initialize_models()


# ---------------------------------------------------------------------------
# Register Blueprints
# ---------------------------------------------------------------------------

app.register_blueprint(main_bp)
app.register_blueprint(clips_bp)
app.register_blueprint(sorting_bp)
app.register_blueprint(datasets_bp)
app.register_blueprint(exporters_bp)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VistaTotes — media explorer web app")
    parser.add_argument("--local", action="store_true", help="Run in local development mode")
    parser.add_argument(
        "--autodetect",
        action="store_true",
        help="Run a detector on a dataset from the command line and print predicted-Good items",
    )
    parser.add_argument("--dataset", type=str, help="Path to a dataset pickle file (used with --autodetect)")
    parser.add_argument("--detector", type=str, help="Path to a detector JSON file (used with --autodetect)")
    args = parser.parse_args()

    if args.autodetect:
        if not args.dataset or not args.detector:
            parser.error("--autodetect requires both --dataset and --detector")

        from vistatotes.cli import autodetect_main

        autodetect_main(args.dataset, args.detector)

    elif args.local:
        # Local development mode
        print("🚀 Running in LOCAL mode (accessible from other devices)", flush=True)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    else:
        # Production mode — models load lazily when the first dataset is loaded
        print("🚀 Running in PRODUCTION mode", flush=True)
        initialize_models()

        print("✅ VistaTotes is ready!", flush=True)
        print("🌐 Open http://localhost:5000 in your browser", flush=True)

        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
