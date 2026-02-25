"""Image media type — CLIP embeddings, JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from vtsearch.config import CLIP_MODEL_ID, MODELS_CACHE_DIR

if TYPE_CHECKING:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
    intercept_tqdm_progress,
)


def _extract_tensor(output: object) -> torch.Tensor:
    """Extract a plain tensor from model output.

    Depending on the transformers version, get_image_features() / get_text_features()
    may return either a raw tensor or a BaseModelOutputWithPooling dataclass.
    This helper handles both cases.
    """
    import torch  # noqa: PLC0415

    if isinstance(output, torch.Tensor):
        return output
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        val = getattr(output, attr, None)
        if isinstance(val, torch.Tensor):
            return val
    # Final fallback: treat as tuple-like and return first element
    return output[0]  # type: ignore[index]


_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageMediaType(MediaType):
    """Handles image clips using the CLIP model (openai/clip-vit-base-patch32).

    * Embeds images via CLIP's vision encoder (768-dim vectors).
    * Embeds text queries via CLIP's text encoder (same 768-dim space).
    * Serves clips as image files with MIME types inferred from extension.
    * Also exposes :meth:`embed_pil_image` for in-memory PIL Image objects
      (used when generating CIFAR-10 demo datasets).
    """

    def __init__(self) -> None:
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._on_progress: ProgressCallback = _noop_progress

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def type_id(self) -> str:
        return "image"

    @property
    def name(self) -> str:
        return "Image"

    @property
    def icon(self) -> str:
        return "🖼️"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]

    @property
    def folder_import_name(self) -> str:
        return "images"

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    # Shared categories for S and M image demo datasets.
    # Both sizes use the same 25 Caltech-101 categories; only the
    # underlying images differ (disjoint slices within each category).
    _DEMO_CATEGORIES_CALTECH101 = [
        "airplanes",
        "bonsai",
        "car_side",
        "chandelier",
        "cougar_face",
        "crab",
        "dalmatian",
        "dolphin",
        "elephant",
        "ferry",
        "flamingo",
        "grand_piano",
        "hawksbill",
        "helicopter",
        "ibis",
        "kangaroo",
        "ketch",
        "lamp",
        "laptop",
        "nautilus",
        "starfish",
        "stop_sign",
        "sunflower",
        "trilobite",
        "watch",
    ]

    # Categories for the L dataset from Caltech-256 — a different source
    # with its own numbered category naming scheme.
    _DEMO_CATEGORIES_CALTECH256 = [
        "003.backpack",
        "024.butterfly",
        "028.camel",
        "029.cannon",
        "038.chimp",
        "046.computer-monitor",
        "069.fighter-jet",
        "084.giraffe",
        "085.goat",
        "086.golden-gate-bridge",
        "096.hammock",
        "107.hot-air-balloon",
        "122.kayak",
        "129.leopards-101",
        "132.light-house",
        "145.motorbikes-101",
        "147.mushroom",
        "151.ostrich",
        "158.penguin",
        "167.pyramid",
        "178.school-bus",
        "200.stained-glass",
        "207.swan",
        "245.windmill",
        "246.wine-bottle",
    ]

    # Categories for Oxford Flowers 102 (102 flower species).
    _OXFORD_FLOWERS_CATEGORIES = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
        "sweet pea", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris",
        "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
        "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
        "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
        "stemless gentian", "artichoke", "sweet william", "carnation",
        "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
        "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
        "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
        "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
        "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
        "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
        "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
        "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
        "azalea", "water lily", "rose", "thorn apple", "morning glory",
        "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
        "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
        "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
        "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
        "mallow", "mexican petunia", "bromelia", "blanket flower",
        "trumpet creeper", "blackberry lily",
    ]

    # Categories for Food-101 (101 food categories).
    _FOOD101_CATEGORIES = [
        "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio",
        "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding",
        "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli",
        "caprese_salad", "carrot_cake", "ceviche", "cheesecake",
        "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings",
        "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
        "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
        "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame",
        "eggs_benedict", "escargots", "falafel", "filet_mignon",
        "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
        "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
        "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
        "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
        "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
        "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
        "macarons", "miso_soup", "mussels", "nachos", "omelette",
        "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
        "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop",
        "poutine", "prime_rib", "pulled_pork_sandwich", "ramen",
        "ravioli", "red_velvet_cake", "risotto", "samosa",
        "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
        "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
        "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
        "tiramisu", "tuna_tartare", "waffles",
    ]

    # Categories for EuroSAT (10 land use classes).
    _EUROSAT_CATEGORIES = [
        "AnnualCrop",
        "Forest",
        "HerbaceousVegetation",
        "Highway",
        "Industrial",
        "Pasture",
        "PermanentCrop",
        "Residential",
        "River",
        "SeaLake",
    ]

    # Categories for Stanford Dogs (120 breeds).
    _STANFORD_DOGS_CATEGORIES = [
        "Chihuahua", "Japanese_spaniel", "Maltese_dog", "Pekinese", "Shih-Tzu",
        "Blenheim_spaniel", "papillon", "toy_terrier", "Rhodesian_ridgeback",
        "Afghan_hound", "basset", "beagle", "bloodhound", "bluetick",
        "black-and-tan_coonhound", "Walker_hound", "English_foxhound",
        "redbone", "borzoi", "Irish_wolfhound", "Italian_greyhound",
        "whippet", "Ibizan_hound", "Norwegian_elkhound", "otterhound",
        "Saluki", "Scottish_deerhound", "Weimaraner", "Staffordshire_bullterrier",
        "American_Staffordshire_terrier", "Bedlington_terrier", "Border_terrier",
        "Kerry_blue_terrier", "Irish_terrier", "Norfolk_terrier",
        "Norwich_terrier", "Yorkshire_terrier", "wire-haired_fox_terrier",
        "Lakeland_terrier", "Sealyham_terrier", "Airedale", "cairn",
        "Australian_terrier", "Dandie_Dinmont", "Boston_bull", "miniature_schnauzer",
        "giant_schnauzer", "standard_schnauzer", "Scotch_terrier",
        "Tibetan_terrier", "silky_terrier", "soft-coated_wheaten_terrier",
        "West_Highland_white_terrier", "Lhasa", "flat-coated_retriever",
        "curly-coated_retriever", "golden_retriever", "Labrador_retriever",
        "Chesapeake_Bay_retriever", "German_short-haired_pointer", "vizsla",
        "English_setter", "Irish_setter", "Gordon_setter", "Brittany_spaniel",
        "clumber", "English_springer", "Welsh_springer_spaniel",
        "cocker_spaniel", "Sussex_spaniel", "Irish_water_spaniel", "kuvasz",
        "schipperke", "groenendael", "malinois", "briard", "kelpie",
        "komondor", "Old_English_sheepdog", "Shetland_sheepdog", "collie",
        "Border_collie", "Bouvier_des_Flandres", "Rottweiler",
        "German_shepherd", "Doberman", "miniature_pinscher",
        "Greater_Swiss_Mountain_dog", "Bernese_mountain_dog",
        "Appenzeller", "EntleBucher", "boxer", "bull_mastiff",
        "Tibetan_mastiff", "French_bulldog", "Great_Dane",
        "Saint_Bernard", "Eskimo_dog", "malamute", "Siberian_husky",
        "affenpinscher", "basenji", "pug", "Leonberg", "Newfoundland",
        "Great_Pyrenees", "Samoyed", "Pomeranian", "chow",
        "keeshond", "Brabancon_griffon", "Pembroke", "Cardigan",
        "toy_poodle", "miniature_poodle", "standard_poodle",
        "Mexican_hairless", "dingo", "dhole", "African_hunting_dog",
    ]

    @property
    def demo_datasets(self) -> list:
        cats101 = self._DEMO_CATEGORIES_CALTECH101
        cats256 = self._DEMO_CATEGORIES_CALTECH256
        return [
            DemoDataset(
                id="caltech101_s",
                label="Caltech-101 (S)",
                description=(
                    "~500 photographs across 25 categories — animals, vehicles,"
                    " household objects, and nature from the Caltech-101 dataset."
                ),
                categories=cats101,
                source="caltech101",
                slice_start=0,
                slice_end=20,
            ),
            DemoDataset(
                id="caltech101_m",
                label="Caltech-101 (M)",
                description=(
                    "~1,000 photographs across 25 categories — animals, vehicles,"
                    " household objects, and nature from the Caltech-101 dataset."
                ),
                categories=cats101,
                source="caltech101",
                slice_start=20,
                slice_end=60,
            ),
            DemoDataset(
                id="caltech256_l",
                label="Caltech-256 (L)",
                description=(
                    "~2,000 photographs across 25 categories — animals, landmarks,"
                    " vehicles, and everyday objects from the Caltech-256 dataset."
                ),
                categories=cats256,
                source="caltech256",
                slice_start=0,
                slice_end=80,
            ),
            DemoDataset(
                id="oxford_flowers_102_a",
                label="Oxford Flowers 102 (A)",
                description=(
                    "~8,189 photographs of 102 flower species — roses, sunflowers,"
                    " orchids, daisies, and many more from the Oxford Flowers dataset."
                ),
                categories=self._OXFORD_FLOWERS_CATEGORIES,
                source="oxford_flowers_102",
                slice_start=0,
                slice_end=80,
            ),
            DemoDataset(
                id="food101_a",
                label="Food-101 (A)",
                description=(
                    "~101,000 food photographs across 101 categories — sushi,"
                    " pizza, steak, ice cream, and more from the Food-101 dataset."
                ),
                categories=self._FOOD101_CATEGORIES,
                source="food101",
                slice_start=0,
                slice_end=1000,
            ),
            DemoDataset(
                id="eurosat_a",
                label="EuroSAT (A)",
                description=(
                    "~27,000 satellite images across 10 land use classes — forest,"
                    " residential, industrial, river, and more from the EuroSAT dataset."
                ),
                categories=self._EUROSAT_CATEGORIES,
                source="eurosat",
                slice_start=0,
                slice_end=2700,
            ),
            DemoDataset(
                id="stanford_dogs_a",
                label="Stanford Dogs (A)",
                description=(
                    "~20,580 photographs of 120 dog breeds — from Chihuahuas to"
                    " Great Danes, from the Stanford Dogs dataset."
                ),
                categories=self._STANFORD_DOGS_CATEGORIES,
                source="stanford_dogs",
                slice_start=0,
                slice_end=171,
            ),
        ]

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    @property
    def description_wrappers(self) -> list[str]:
        return [
            "a photo of {text}",
            "a photograph of {text}",
            "an image of {text}",
            "{text}",
            "a picture of {text}",
        ]

    def load_models(self) -> None:
        if self._model is not None:
            return
        import gc

        from transformers import CLIPModel, CLIPProcessor  # noqa: PLC0415

        gc.collect()
        cache_dir = str(MODELS_CACHE_DIR)
        # Use total=0 so the frontend shows an indeterminate progress bar.
        # If from_pretrained emits tqdm bars (e.g. first-time download),
        # intercept_tqdm_progress will override with actual progress.
        self._on_progress("loading", "Loading CLIP model weights…", 0, 0)
        # Older CLIP checkpoints include position_ids buffers that newer transformers
        # versions compute on-the-fly.  Tell the loader to silently ignore them.
        CLIPModel._keys_to_ignore_on_load_unexpected = [r".*position_ids.*"]
        with intercept_tqdm_progress(self._on_progress):
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_ID, low_cpu_mem_usage=True, cache_dir=cache_dir, token=False)
        self._on_progress("loading", "Loading CLIP processor…", 0, 0)
        with intercept_tqdm_progress(self._on_progress):
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, cache_dir=cache_dir, use_fast=True, token=False)

    def embed_media(self, file_path: Path) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            from PIL import Image  # noqa: PLC0415

            image = Image.open(file_path).convert("RGB")
            return self.embed_pil_image(image)
        except Exception as e:
            print(f"Error embedding {file_path}: {e}")
            return None

    def embed_pil_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """Embed a PIL Image that is already in memory (e.g. from CIFAR-10)."""
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            image = image.convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                embedding = _extract_tensor(outputs).detach().cpu().numpy()
            return embedding[0]
        except Exception as e:
            print(f"Error embedding PIL image: {e}")
            return None

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            self.load_models()
        if self._model is None or self._processor is None:
            return None
        try:
            import torch  # noqa: PLC0415

            inputs = self._processor(text=[text], return_tensors="pt")
            with torch.no_grad():
                text_vec = _extract_tensor(self._model.get_text_features(**inputs)).detach().cpu().numpy()[0]
            return text_vec
        except Exception as e:
            print(f"Error embedding text query for image: {e}")
            return None

    # internal helper used by loader.py's get_clip_model() bridge
    def _get_model_and_processor(self):
        if self._model is None:
            self.load_models()
        return self._model, self._processor

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_clip_data(self, file_path: Path) -> dict:
        from PIL import Image  # noqa: PLC0415

        with open(file_path, "rb") as f:
            clip_bytes = f.read()
        try:
            img = Image.open(file_path)
            width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "clip_bytes": clip_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def clip_response(self, clip: dict) -> MediaResponse:
        filename = clip.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        data = self._resolve_clip_bytes(clip)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"clip_{clip['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"clip_{clip['id']}{ext}",
        )
