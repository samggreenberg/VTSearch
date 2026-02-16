# VectoryTones

A Flask web application for audio clip evaluation with LAION-CLAP embeddings.

## Quick Start

### 1. Install Dependencies

Choose the requirements file based on your system:

**For GPU systems** (with NVIDIA GPU):
```bash
cd ~/vectorytones
source venv/bin/activate
pip install -r requirements-gpu.txt
```

**For CPU-only systems** (Chromebook, Mac, or systems without GPU):
```bash
cd ~/vectorytones
source venv/bin/activate
pip install -r requirements-cpu.txt
```

The main difference is the PyTorch version:
- `requirements-gpu.txt` installs full PyTorch with CUDA support (~2GB)
- `requirements-cpu.txt` installs PyTorch CPU-only (~500MB)

### 2. Download and Setup Datasets

Run the setup script to download ESC-50 audio dataset and generate CLAP embeddings:

```bash
python setup_datasets.py
```

This will:
- Download ~600MB of audio data
- Generate semantic embeddings using LAION-CLAP
- Create 4 themed datasets (animals, natural, urban, household)
- Takes ~10-20 minutes depending on your hardware

### 3. Run the Application

```bash
python app.py
```

Visit http://localhost:5000

## Features

- **Multiple Preset Datasets**: Choose from animals, natural sounds, urban sounds, or household sounds
- **LAION-CLAP Embeddings**: Each audio clip has a 512-dimensional semantic embedding
- **Voting System**: Rate clips as good or bad
- **REST API**: Full API for programmatic access

## API Endpoints

### Datasets
- `GET /api/datasets` - List available datasets
- `POST /api/datasets/<name>/select` - Switch dataset

### Clips
- `GET /api/clips` - List all clips in current dataset
- `GET /api/clips/<id>` - Get clip details with embedding
- `GET /api/clips/<id>/audio` - Stream audio file
- `POST /api/clips/<id>/vote` - Vote on a clip

### System
- `GET /api/status` - Application status
- `GET /api/votes` - Current votes

## Project Structure

```
vectorytones/
├── app.py                      # Flask application
├── setup_datasets.py           # Dataset download & embedding script
├── requirements-gpu.txt        # Python dependencies (GPU systems)
├── requirements-cpu.txt        # Python dependencies (CPU-only systems)
├── static/                     # Frontend assets
├── templates/                  # HTML templates
└── data/                       # Downloaded datasets (not in git)
    ├── ESC-50-master/          # ESC-50 audio files
    └── embeddings/             # Precomputed CLAP embeddings
```

## Tech Stack

- **Backend**: Flask, Python 3.11
- **Audio**: librosa, soundfile
- **Embeddings**: LAION-CLAP (Contrastive Language-Audio Pretraining)
- **Dataset**: ESC-50 (Environmental Sound Classification)

## License

- Application code: Your license
- ESC-50 dataset: CC BY-NC
- LAION-CLAP model: MIT

## References

- [LAION-CLAP](https://github.com/LAION-AI/CLAP)
- [ESC-50 Dataset](https://github.com/karolpiczak/ESC-50)
- [Setup Guide](SETUP_DATASETS.md)

## Development

For detailed setup instructions, see [SETUP_DATASETS.md](SETUP_DATASETS.md).

For production deployment, remember to:
- Set `debug=False` in `app.py`
- Use a production WSGI server (gunicorn, uwsgi)
- Set up proper authentication if needed
