import io
import pickle
from pathlib import Path

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

EMBEDDINGS_DIR = Path("data/embeddings")
CURRENT_DATASET = "animals"  # Default dataset

datasets = {}  # dataset_name -> {clips, audio_dir}
clips = {}  # id -> {id, duration, file_size, embedding, audio_path}
good_votes: set[int] = set()
bad_votes: set[int] = set()


def load_datasets():
    """Load all available datasets from disk."""
    global datasets

    if not EMBEDDINGS_DIR.exists():
        print("⚠️  No datasets found. Run setup_datasets.py first!")
        return

    for pkl_file in EMBEDDINGS_DIR.glob("*.pkl"):
        dataset_name = pkl_file.stem
        print(f"Loading dataset: {dataset_name}")

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            datasets[dataset_name] = data
            print(f"  ✓ Loaded {len(data['clips'])} clips")

    if not datasets:
        print("⚠️  No datasets loaded. Run setup_datasets.py first!")


def set_current_dataset(dataset_name: str) -> bool:
    """Switch to a different dataset."""
    global clips, good_votes, bad_votes, CURRENT_DATASET

    if dataset_name not in datasets:
        return False

    CURRENT_DATASET = dataset_name
    dataset = datasets[dataset_name]

    # Load clips with audio paths
    clips = {}
    audio_dir = Path(dataset['audio_dir'])

    for clip_id, clip_data in dataset['clips'].items():
        clips[clip_id] = {
            'id': clip_id,
            'category': clip_data['category'],
            'duration': clip_data['duration'],
            'file_size': clip_data['file_size'],
            'embedding': clip_data['embedding'],
            'audio_path': audio_dir / clip_data['filename']
        }

    # Reset votes when switching datasets
    good_votes = set()
    bad_votes = set()

    print(f"✓ Switched to dataset: {dataset_name} ({len(clips)} clips)")
    return True


# Initialize datasets on startup
load_datasets()
if datasets:
    set_current_dataset(CURRENT_DATASET)
else:
    print("\n" + "="*60)
    print("⚠️  NO DATASETS FOUND!")
    print("="*60)
    print("Please run the setup script first:")
    print("  python setup_datasets.py")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/datasets")
def list_datasets():
    """List all available datasets."""
    result = []
    for name, data in datasets.items():
        result.append({
            'name': name,
            'num_clips': len(data['clips']),
            'is_current': name == CURRENT_DATASET
        })
    return jsonify(result)


@app.route("/api/datasets/<dataset_name>/select", methods=["POST"])
def select_dataset(dataset_name):
    """Switch to a different dataset."""
    if set_current_dataset(dataset_name):
        return jsonify({'ok': True, 'dataset': dataset_name})
    else:
        return jsonify({'error': 'Dataset not found'}), 404


@app.route("/api/clips")
def list_clips():
    """List all clips in the current dataset."""
    result = []
    for c in clips.values():
        result.append({
            'id': c['id'],
            'category': c['category'],
            'duration': c['duration'],
            'file_size': c['file_size'],
        })
    return jsonify(result)


@app.route("/api/clips/<int:clip_id>")
def get_clip(clip_id):
    """Get detailed info for a specific clip including embedding."""
    c = clips.get(clip_id)
    if not c:
        return jsonify({'error': 'not found'}), 404

    return jsonify({
        'id': c['id'],
        'category': c['category'],
        'duration': c['duration'],
        'file_size': c['file_size'],
        'embedding_dim': len(c['embedding']),
        'embedding': c['embedding']  # Full CLAP embedding vector
    })


@app.route("/api/clips/<int:clip_id>/audio")
def clip_audio(clip_id):
    """Serve the audio file for a clip."""
    c = clips.get(clip_id)
    if not c:
        return jsonify({'error': 'not found'}), 404

    audio_path = c['audio_path']
    if not audio_path.exists():
        return jsonify({'error': 'audio file not found'}), 404

    return send_file(
        audio_path,
        mimetype="audio/wav",
        download_name=f"clip_{clip_id}.wav",
    )


@app.route("/api/clips/<int:clip_id>/vote", methods=["POST"])
def vote_clip(clip_id):
    """Vote on a clip (good/bad)."""
    if clip_id not in clips:
        return jsonify({'error': 'not found'}), 404

    data = request.get_json(force=True)
    vote = data.get('vote')

    if vote not in ('good', 'bad'):
        return jsonify({'error': 'vote must be "good" or "bad"'}), 400

    if vote == 'good':
        if clip_id in good_votes:
            good_votes.discard(clip_id)
        else:
            bad_votes.discard(clip_id)
            good_votes.add(clip_id)
    else:
        if clip_id in bad_votes:
            bad_votes.discard(clip_id)
        else:
            good_votes.discard(clip_id)
            bad_votes.add(clip_id)

    return jsonify({'ok': True})


@app.route("/api/votes")
def get_votes():
    """Get all current votes."""
    return jsonify({
        'good': sorted(good_votes),
        'bad': sorted(bad_votes),
    })


@app.route("/api/status")
def status():
    """Get application status."""
    return jsonify({
        'datasets_loaded': len(datasets),
        'current_dataset': CURRENT_DATASET,
        'num_clips': len(clips),
        'good_votes': len(good_votes),
        'bad_votes': len(bad_votes)
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
