"""Demo dataset definitions — centralised registry of all demo datasets.

Every demo dataset that can be loaded through the
:class:`DemoDatasetImporter` is defined here, grouped by media type.
Keeping the definitions co-located with the importer ensures that
removing the demo importer also removes all demo-specific constants.

The ``load_demo_source()`` methods on each :class:`~vtsearch.media.base.MediaType`
still contain the media-type-specific download and embedding logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ------------------------------------------------------------------
# DemoDataset dataclass
# ------------------------------------------------------------------


@dataclass
class DemoDataset:
    """Metadata describing one demo dataset."""

    id: str
    """Unique key used throughout the app (e.g. ``"nature_sounds"``)."""

    label: str
    """Human-readable display name (e.g. ``"Animal & Nature Sounds"``)."""

    description: str
    """Long-form description shown in the UI."""

    categories: list
    """Category names used to filter the raw source data."""

    media_type: str
    """Media type identifier (e.g. ``"audio"``, ``"image"``)."""

    source: str = ""
    """Identifier for the raw data source (e.g. ``"cifar10_sample"``, ``"ucf101"``).
    Leave empty for sources that don't require an explicit identifier."""

    required_folder: Optional[Path] = None
    """Local directory that must exist for a cached ``.pkl`` to be usable.

    Audio and video datasets store references to external media files rather
    than inlining the bytes, so a stale ``.pkl`` left behind after the source
    directory was removed would incorrectly appear ready.  Set this to the
    directory that the importer places the source files into (e.g.
    ``DATA_DIR / "ESC-50-master" / "audio"``)."""

    slice_start: int = 0
    """Per-category start index for element slicing (inclusive)."""

    slice_end: Optional[int] = None
    """Per-category end index for element slicing (exclusive).
    ``None`` means take all remaining elements after ``slice_start``."""

    download_size_mb: float = 0
    """Estimated download size in megabytes for UI display."""


# ------------------------------------------------------------------
# Audio demo datasets
# ------------------------------------------------------------------

ESC50_CATEGORIES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creep", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw",
]

GTZAN_CATEGORIES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

SPEECH_COMMANDS_CATEGORIES = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow",
    "backward", "follow", "forward", "learn", "visual",
]

URBANSOUND8K_CATEGORIES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
]


def _audio_demo_datasets() -> list[DemoDataset]:
    from vtsearch.config import DATA_DIR
    from vtsearch.datasets.downloader import (
        ESC50_DOWNLOAD_SIZE_MB,
        GTZAN_DOWNLOAD_SIZE_MB,
        SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        URBANSOUND8K_DOWNLOAD_SIZE_MB,
    )

    folder = DATA_DIR / "ESC-50-master" / "audio"
    return [
        DemoDataset(
            id="esc50_s", label="ESC-50 (S)", media_type="audio",
            description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
            categories=ESC50_CATEGORIES, required_folder=folder,
            slice_start=0, slice_end=7, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="esc50_m", label="ESC-50 (M)", media_type="audio",
            description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
            categories=ESC50_CATEGORIES, required_folder=folder,
            slice_start=7, slice_end=20, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="esc50_l", label="ESC-50 (L)", media_type="audio",
            description="Real-world environmental recordings — animals, nature, cities, homes, and people.",
            categories=ESC50_CATEGORIES, required_folder=folder,
            slice_start=20, slice_end=40, download_size_mb=ESC50_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="gtzan_a", label="GTZAN Music Genre (A)", media_type="audio",
            description="30-second music excerpts, one per genre.",
            categories=GTZAN_CATEGORIES, source="gtzan",
            slice_start=0, slice_end=100, download_size_mb=GTZAN_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="speech_commands_v2_a", label="Speech Commands v2 (A)", media_type="audio",
            description="One-second keyword utterances from crowd-sourced speakers.",
            categories=SPEECH_COMMANDS_CATEGORIES, source="speech_commands_v2",
            slice_start=0, slice_end=3000, download_size_mb=SPEECH_COMMANDS_V2_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="urbansound8k_a", label="UrbanSound8K (A)", media_type="audio",
            description="Real urban field recordings, pre-segmented into labeled sounds.",
            categories=URBANSOUND8K_CATEGORIES, source="urbansound8k",
            slice_start=0, slice_end=873, download_size_mb=URBANSOUND8K_DOWNLOAD_SIZE_MB,
        ),
    ]


# ------------------------------------------------------------------
# Image demo datasets
# ------------------------------------------------------------------

CALTECH101_CATEGORIES = [
    "airplanes", "bonsai", "car_side", "chandelier", "cougar_face",
    "crab", "dalmatian", "dolphin", "elephant", "ferry",
    "flamingo", "grand_piano", "hawksbill", "helicopter", "ibis",
    "kangaroo", "ketch", "lamp", "laptop", "nautilus",
    "starfish", "stop_sign", "sunflower", "trilobite", "watch",
]

CALTECH256_CATEGORIES = [
    "003.backpack", "024.butterfly", "028.camel", "029.cannon", "038.chimp",
    "046.computer-monitor", "069.fighter-jet", "084.giraffe", "085.goat", "086.golden-gate-bridge",
    "096.hammock", "107.hot-air-balloon", "122.kayak", "129.leopards-101", "132.light-house",
    "145.motorbikes-101", "147.mushroom", "151.ostrich", "158.penguin", "167.pyramid",
    "178.school-bus", "200.stained-glass", "207.swan", "245.windmill", "246.wine-bottle",
]

OXFORD_FLOWERS_CATEGORIES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]

FOOD101_CATEGORIES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
    "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho",
    "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]

EUROSAT_CATEGORIES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]

STANFORD_DOGS_CATEGORIES = [
    "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu",
    "Blenheim_spaniel", "papillon", "toy_terrier", "Rhodesian_ridgeback", "Afghan_hound",
    "basset", "beagle", "bloodhound", "bluetick", "black-and-tan_coonhound",
    "Walker_hound", "English_foxhound", "redbone", "borzoi", "Irish_wolfhound",
    "Italian_greyhound", "whippet", "Ibizan_hound", "Norwegian_elkhound", "otterhound",
    "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier",
    "American_Staffordshire_terrier", "Bedlington_terrier", "Border_terrier",
    "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier", "Norwich_terrier",
    "Yorkshire_terrier", "wire-haired_fox_terrier", "Lakeland_terrier", "Sealyham_terrier",
    "Airedale", "cairn", "Australian_terrier", "Dandie_Dinmont", "Boston_bull",
    "miniature_schnauzer", "giant_schnauzer", "standard_schnauzer", "Scotch_terrier",
    "Tibetan_terrier", "silky_terrier", "soft-coated_wheaten_terrier",
    "West_Highland_white_terrier", "Lhasa", "flat-coated_retriever", "curly-coated_retriever",
    "golden_retriever", "Labrador_retriever", "Chesapeake_Bay_retriever",
    "German_short-haired_pointer", "vizsla", "English_setter", "Irish_setter",
    "Gordon_setter", "Brittany_spaniel", "clumber", "English_springer",
    "Welsh_springer_spaniel", "cocker_spaniel", "Sussex_spaniel", "Irish_water_spaniel",
    "kuvasz", "schipperke", "groenendael", "malinois", "briard", "kelpie", "komondor",
    "Old_English_sheepdog", "Shetland_sheepdog", "collie", "Border_collie",
    "Bouvier_des_Flandres", "Rottweiler", "German_shepherd", "Doberman",
    "miniature_pinscher", "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog",
    "Appenzeller", "EntleBucher", "boxer", "bull_mastiff", "Tibetan_mastiff",
    "French_bulldog", "Great_Dane", "Saint_Bernard", "Eskimo_dog", "malamute",
    "Siberian_husky", "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland",
    "Great_Pyrenees", "Samoyed", "Pomeranian", "chow", "keeshond",
    "Brabancon_griffon", "Pembroke", "Cardigan", "toy_poodle", "miniature_poodle",
    "standard_poodle", "Mexican_hairless", "dingo", "dhole", "African_hunting_dog",
]

UCSF_DOCUMENTS_CATEGORIES = [
    "Tobacco", "Food", "Drug", "Chemical", "Fossil Fuel", "Opioids",
]


def _image_demo_datasets() -> list[DemoDataset]:
    from vtsearch.config import DATA_DIR
    from vtsearch.datasets.downloader import (
        CALTECH101_DOWNLOAD_SIZE_MB,
        CALTECH256_DOWNLOAD_SIZE_MB,
        EUROSAT_DOWNLOAD_SIZE_MB,
        FOOD101_DOWNLOAD_SIZE_MB,
        OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        UCSF_IDL_DOWNLOAD_SIZE_MB,
    )

    caltech101_folder = DATA_DIR / "caltech-101" / "101_ObjectCategories"
    caltech256_folder = DATA_DIR / "caltech-256" / "256_ObjectCategories"
    oxford_flowers_folder = DATA_DIR / "oxford_flowers"
    food101_folder = DATA_DIR / "food-101" / "images"
    eurosat_folder = DATA_DIR / "EuroSAT_RGB"
    stanford_dogs_folder = DATA_DIR / "stanford_dogs" / "Images"
    ucsf_documents_folder = DATA_DIR / "ucsf_documents"
    return [
        DemoDataset(
            id="caltech101_s", label="Caltech-101 (S)", media_type="image",
            description="Centered, well-lit object photos — a classic vision benchmark.",
            categories=CALTECH101_CATEGORIES, source="caltech101", required_folder=caltech101_folder,
            slice_start=0, slice_end=20, download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech101_m", label="Caltech-101 (M)", media_type="image",
            description="Centered, well-lit object photos — a classic vision benchmark.",
            categories=CALTECH101_CATEGORIES, source="caltech101", required_folder=caltech101_folder,
            slice_start=20, slice_end=60, download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="caltech256_l", label="Caltech-256 (L)", media_type="image",
            description="Harder object photos with more varied, cluttered backgrounds than Caltech-101.",
            categories=CALTECH256_CATEGORIES, source="caltech256", required_folder=caltech256_folder,
            slice_start=0, slice_end=80, download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="oxford_flowers_102_a", label="Oxford Flowers 102 (A)", media_type="image",
            description="Close-up flower photography with fine-grained species variation.",
            categories=OXFORD_FLOWERS_CATEGORIES, source="oxford_flowers_102",
            required_folder=oxford_flowers_folder,
            slice_start=0, slice_end=80, download_size_mb=OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="food101_a", label="Food-101 (A)", media_type="image",
            description="Crowd-sourced food photos, some mislabeled — a deliberately noisy benchmark.",
            categories=FOOD101_CATEGORIES, source="food101",
            required_folder=food101_folder,
            slice_start=0, slice_end=1000, download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="eurosat_a", label="EuroSAT (A)", media_type="image",
            description="Sentinel-2 satellite imagery classified by land use type.",
            categories=EUROSAT_CATEGORIES, source="eurosat",
            required_folder=eurosat_folder,
            slice_start=0, slice_end=2700, download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="stanford_dogs_a", label="Stanford Dogs (A)", media_type="image",
            description="Fine-grained dog breed photos — many visually similar breeds.",
            categories=STANFORD_DOGS_CATEGORIES, source="stanford_dogs",
            required_folder=stanford_dogs_folder,
            slice_start=0, slice_end=171, download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucsf_documents_a", label="UCSF Documents (A)", media_type="image",
            description="Scanned industry document pages from the UCSF Industry Documents Library.",
            categories=UCSF_DOCUMENTS_CATEGORIES, source="ucsf_documents",
            required_folder=ucsf_documents_folder,
            slice_start=0, slice_end=25, download_size_mb=UCSF_IDL_DOWNLOAD_SIZE_MB,
        ),
    ]


# ------------------------------------------------------------------
# Text demo datasets
# ------------------------------------------------------------------

NEWSGROUPS_CATEGORIES = [
    "sports", "science", "cars", "hockey", "electronics",
    "religion", "world", "business", "technology", "medicine",
    "crypto", "atheism", "motorcycles", "mideast", "guns",
]

AG_NEWS_CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

BBC_NEWS_CATEGORIES = ["business", "entertainment", "politics", "sport", "tech"]

IMDB_CATEGORIES = ["pos", "neg"]


def _text_demo_datasets() -> list[DemoDataset]:
    return [
        DemoDataset(
            id="20newsgroups_s", label="20 Newsgroups (S)", media_type="text",
            description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
            categories=NEWSGROUPS_CATEGORIES, source="ag_news_sample",
            slice_start=0, slice_end=25, download_size_mb=15,
        ),
        DemoDataset(
            id="20newsgroups_m", label="20 Newsgroups (M)", media_type="text",
            description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
            categories=NEWSGROUPS_CATEGORIES, source="ag_news_sample",
            slice_start=25, slice_end=75, download_size_mb=15,
        ),
        DemoDataset(
            id="20newsgroups_l", label="20 Newsgroups (L)", media_type="text",
            description="Usenet posts from the early 1990s across technical, recreational, and political topics.",
            categories=NEWSGROUPS_CATEGORIES, source="ag_news_sample",
            slice_start=75, slice_end=200, download_size_mb=15,
        ),
        DemoDataset(
            id="ag_news_a", label="AG News (A)", media_type="text",
            description="Short news summaries, well-balanced across world, sports, business, and tech.",
            categories=AG_NEWS_CATEGORIES, source="ag_news",
            slice_start=0, slice_end=30000, download_size_mb=15,
        ),
        DemoDataset(
            id="bbc_news_a", label="BBC News (A)", media_type="text",
            description="Full BBC news articles — professionally written and cleanly labeled.",
            categories=BBC_NEWS_CATEGORIES, source="bbc_news",
            slice_start=0, slice_end=445, download_size_mb=15,
        ),
        DemoDataset(
            id="imdb_a", label="IMDB Movie Reviews (A)", media_type="text",
            description="Long-form user-written movie reviews with binary positive/negative sentiment labels.",
            categories=IMDB_CATEGORIES, source="imdb",
            slice_start=0, slice_end=25000, download_size_mb=15,
        ),
    ]


# ------------------------------------------------------------------
# Video demo datasets
# ------------------------------------------------------------------

UCF101_CATEGORIES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
]


def _video_demo_datasets() -> list[DemoDataset]:
    from vtsearch.datasets.downloader import UCF101_SUBSET_DOWNLOAD_SIZE_MB, VIDEO_DIR

    folder = VIDEO_DIR / "ucf101"
    return [
        DemoDataset(
            id="ucf101_s", label="UCF-101 (S)", media_type="video",
            description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
            categories=UCF101_CATEGORIES, source="ucf101", required_folder=folder,
            slice_start=0, slice_end=15, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_m", label="UCF-101 (M)", media_type="video",
            description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
            categories=UCF101_CATEGORIES, source="ucf101", required_folder=folder,
            slice_start=15, slice_end=40, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
        DemoDataset(
            id="ucf101_l", label="UCF-101 (L)", media_type="video",
            description="Action recognition videos sourced from YouTube, covering sports and everyday activities.",
            categories=UCF101_CATEGORIES, source="ucf101", required_folder=folder,
            slice_start=40, slice_end=150, download_size_mb=UCF101_SUBSET_DOWNLOAD_SIZE_MB,
        ),
    ]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def all_demo_datasets() -> dict:
    """Return a flat ``{dataset_id: info_dict}`` mapping of every demo dataset.

    Each value is a dict with the keys expected by the datasets routes:
    ``label``, ``description``, ``categories``, ``media_type``,
    optionally ``source``, and optionally ``required_folder``.
    """
    result: dict = {}
    for ds in _audio_demo_datasets() + _image_demo_datasets() + _text_demo_datasets() + _video_demo_datasets():
        entry: dict = {
            "label": ds.label,
            "description": ds.description,
            "categories": ds.categories,
            "media_type": ds.media_type,
            "slice_start": ds.slice_start,
            "slice_end": ds.slice_end,
            "download_size_mb": ds.download_size_mb,
        }
        if ds.source:
            entry["source"] = ds.source
        if ds.required_folder is not None:
            entry["required_folder"] = ds.required_folder
        result[ds.id] = entry
    return result
