"""Configuration and constants for VTSearch."""

import os
from pathlib import Path

# Dataset paths
DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
VIDEO_DIR = DATA_DIR / "video"
IMAGE_DIR = DATA_DIR / "images"
PARAGRAPH_DIR = DATA_DIR / "paragraphs"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MODELS_CACHE_DIR = (
    Path(os.environ["VTSEARCH_MODELS_DIR"]) if "VTSEARCH_MODELS_DIR" in os.environ else DATA_DIR / "models"
)

# Dataset URLs
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
SAMPLE_VIDEOS_URL = "https://github.com/sample-datasets/video-clips/archive/refs/heads/main.zip"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CALTECH101_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip"
CALTECH256_URL = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"
UCF101_SUBSET_URL = "https://huggingface.co/datasets/sayakpaul/ucf101-subset/resolve/main/UCF101_subset.tar.gz"
BBC_NEWS_URL = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
AG_NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GTZAN_URL = "https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz"
SPEECH_COMMANDS_V2_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
URBANSOUND8K_URL = "https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"
OXFORD_FLOWERS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
OXFORD_FLOWERS_LABELS_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
FOOD101_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
EUROSAT_URL = "https://huggingface.co/datasets/blanchon/EuroSAT_RGB/resolve/main/EuroSAT_RGB.zip"
STANFORD_DOGS_URL = "http://vision.stanford.edu/aditya86/ImageNetDogDataset/images.tar"
UCSF_IDL_API_URL = "https://metadata.idl.ucsf.edu/solr/ltdl3/query"
UCSF_IDL_DOWNLOAD_URL = "https://download.industrydocuments.ucsf.edu"

# Dataset size estimates
ESC50_DOWNLOAD_SIZE_MB = 600
SAMPLE_VIDEOS_DOWNLOAD_SIZE_MB = 150
CIFAR10_DOWNLOAD_SIZE_MB = 170
CALTECH101_DOWNLOAD_SIZE_MB = 131
CALTECH256_DOWNLOAD_SIZE_MB = 1200
UCF101_SUBSET_DOWNLOAD_SIZE_MB = 171
BBC_NEWS_DOWNLOAD_SIZE_MB = 2
AG_NEWS_DOWNLOAD_SIZE_MB = 30
IMDB_DOWNLOAD_SIZE_MB = 84
GTZAN_DOWNLOAD_SIZE_MB = 1200
SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB = 2300
URBANSOUND8K_DOWNLOAD_SIZE_MB = 6000
OXFORD_FLOWERS_DOWNLOAD_SIZE_MB = 330
FOOD101_DOWNLOAD_SIZE_MB = 5000
EUROSAT_DOWNLOAD_SIZE_MB = 90
STANFORD_DOGS_DOWNLOAD_SIZE_MB = 750
UCSF_IDL_DOWNLOAD_SIZE_MB = 50
MEDIAS_PER_CATEGORY = 40
MEDIAS_PER_VIDEO_CATEGORY = 150
IMAGES_PER_CIFAR10_CATEGORY = 100
IMAGES_PER_CALTECH101_CATEGORY = 80
IMAGES_PER_CALTECH256_CATEGORY = 80
TEXTS_PER_CATEGORY = 200

# Training
TRAIN_EPOCHS = 200
MLP_HIDDEN_MIN = 4
MLP_HIDDEN_MAX = 32
MLP_DROPOUT = 0.5

# Model IDs
CLAP_MODEL_ID = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE = 48000  # CLAP model expected input sample rate
XCLIP_MODEL_ID = "microsoft/xclip-base-patch32"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
E5_MODEL_ID = "intfloat/e5-base-v2"
SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"
CLAP_MUSIC_MODEL_ID = "laion/larger_clap_music_and_speech"
BGE_MODEL_ID = "BAAI/bge-base-en-v1.5"
