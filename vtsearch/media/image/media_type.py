"""Image media type — JPEG/PNG/GIF/BMP/WEBP files."""

from __future__ import annotations

from pathlib import Path


from vtsearch.config import DATA_DIR
from vtsearch.media.base import (
    DemoDataset,
    MediaResponse,
    MediaType,
    ProgressCallback,
    _noop_progress,
)


_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ImageMediaType(MediaType):
    """Handles image medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.image.embedder.ImageClipEmbedder`.
    """

    def __init__(self) -> None:
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
        return "image"

    # ------------------------------------------------------------------
    # File import
    # ------------------------------------------------------------------

    @property
    def file_extensions(self) -> list:
        return ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]

    @property
    def folder_import_name(self) -> str:
        return "images"

    @property
    def tab_title(self) -> str:
        return "Images"

    @property
    def dir_key(self) -> str:
        return "image_dir"

    @property
    def pickle_extra_fields(self) -> list[str]:
        return ["width", "height"]

    # ------------------------------------------------------------------
    # Display metadata
    # ------------------------------------------------------------------

    def display_metadata(self, media: dict) -> dict:
        result: dict = {}
        cat = media.get("category")
        if cat and cat not in ("unknown", "custom"):
            result["Category"] = cat
        w, h = media.get("width"), media.get("height")
        if w and h:
            result["Dimensions"] = f"{w}\u00d7{h}"
        fs = media.get("file_size")
        if fs:
            result["File Size"] = fs
        return result

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------

    @property
    def loops(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Demo datasets
    # ------------------------------------------------------------------

    _DEMO_CATEGORIES_CALTECH101 = [
        "airplanes", "bonsai", "car_side", "chandelier", "cougar_face",
        "crab", "dalmatian", "dolphin", "elephant", "ferry",
        "flamingo", "grand_piano", "hawksbill", "helicopter", "ibis",
        "kangaroo", "ketch", "lamp", "laptop", "nautilus",
        "starfish", "stop_sign", "sunflower", "trilobite", "watch",
    ]

    _DEMO_CATEGORIES_CALTECH256 = [
        "003.backpack", "024.butterfly", "028.camel", "029.cannon", "038.chimp",
        "046.computer-monitor", "069.fighter-jet", "084.giraffe", "085.goat", "086.golden-gate-bridge",
        "096.hammock", "107.hot-air-balloon", "122.kayak", "129.leopards-101", "132.light-house",
        "145.motorbikes-101", "147.mushroom", "151.ostrich", "158.penguin", "167.pyramid",
        "178.school-bus", "200.stained-glass", "207.swan", "245.windmill", "246.wine-bottle",
    ]

    _OXFORD_FLOWERS_CATEGORIES = [
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

    _FOOD101_CATEGORIES = [
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

    _EUROSAT_CATEGORIES = [
        "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
        "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
    ]

    _STANFORD_DOGS_CATEGORIES = [
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

    _UCSF_DOCUMENTS_CATEGORIES = [
        "Tobacco", "Food", "Drug", "Chemical", "Fossil Fuel", "Opioids",
    ]

    @property
    def demo_datasets(self) -> list:
        from vtsearch.datasets.downloader import (  # noqa: PLC0415
            CALTECH101_DOWNLOAD_SIZE_MB,
            CALTECH256_DOWNLOAD_SIZE_MB,
            EUROSAT_DOWNLOAD_SIZE_MB,
            FOOD101_DOWNLOAD_SIZE_MB,
            OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
            STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            UCSF_IDL_DOWNLOAD_SIZE_MB,
        )

        cats101 = self._DEMO_CATEGORIES_CALTECH101
        cats256 = self._DEMO_CATEGORIES_CALTECH256
        return [
            DemoDataset(
                id="caltech101_s", label="Caltech-101 (S)",
                description="Centered, well-lit object photos — a classic vision benchmark.",
                categories=cats101, source="caltech101",
                required_folder=DATA_DIR / "caltech-101" / "101_ObjectCategories",
                slice_start=0, slice_end=20, download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech101_m", label="Caltech-101 (M)",
                description="Centered, well-lit object photos — a classic vision benchmark.",
                categories=cats101, source="caltech101",
                required_folder=DATA_DIR / "caltech-101" / "101_ObjectCategories",
                slice_start=20, slice_end=60, download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech256_l", label="Caltech-256 (L)",
                description="Harder object photos with more varied, cluttered backgrounds than Caltech-101.",
                categories=cats256, source="caltech256",
                required_folder=DATA_DIR / "caltech-256" / "256_ObjectCategories",
                slice_start=0, slice_end=80, download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="oxford_flowers_102_a", label="Oxford Flowers 102 (A)",
                description="Close-up flower photography with fine-grained species variation.",
                categories=self._OXFORD_FLOWERS_CATEGORIES, source="oxford_flowers_102",
                required_folder=DATA_DIR / "oxford_flowers",
                slice_start=0, slice_end=80, download_size_mb=OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="food101_a", label="Food-101 (A)",
                description="Crowd-sourced food photos, some mislabeled — a deliberately noisy benchmark.",
                categories=self._FOOD101_CATEGORIES, source="food101",
                required_folder=DATA_DIR / "food-101" / "images",
                slice_start=0, slice_end=1000, download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="eurosat_a", label="EuroSAT (A)",
                description="Sentinel-2 satellite imagery classified by land use type.",
                categories=self._EUROSAT_CATEGORIES, source="eurosat",
                required_folder=DATA_DIR / "EuroSAT_RGB",
                slice_start=0, slice_end=2700, download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="stanford_dogs_a", label="Stanford Dogs (A)",
                description="Fine-grained dog breed photos — many visually similar breeds.",
                categories=self._STANFORD_DOGS_CATEGORIES, source="stanford_dogs",
                required_folder=DATA_DIR / "stanford_dogs" / "Images",
                slice_start=0, slice_end=171, download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucsf_documents_a", label="UCSF Documents (A)",
                description="Scanned industry document pages from the UCSF Industry Documents Library.",
                categories=self._UCSF_DOCUMENTS_CATEGORIES, source="ucsf_documents",
                required_folder=DATA_DIR / "ucsf_documents",
                slice_start=0, slice_end=25, download_size_mb=UCSF_IDL_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(self, source, categories, slice_start, slice_end, clips, on_progress=None, embedder=None):
        import hashlib  # noqa: PLC0415
        import io as _io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.utils import update_progress

            on_progress = update_progress

        if embedder is None:
            from vtsearch.media import embedders_for_type

            avail = embedders_for_type(self.type_id)
            if not avail:
                raise ValueError(f"No embedders registered for media type {self.type_id!r}")
            embedder = avail[0]

        from vtsearch.datasets.loader import load_image_metadata_from_folders  # noqa: PLC0415

        demo_origin: dict = {"importer": "demo", "params": {}}

        def _embed_file_images(selected):
            """Embed a list of (img_path, category) tuples."""
            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                original_cb = embedder._on_progress
                embedder._on_progress = on_progress
                try:
                    embedder.load_models()
                finally:
                    embedder._on_progress = original_cb

            clip_id = max(clips.keys(), default=0) + 1
            total = len(selected)
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (img_path, category) in enumerate(selected):
                on_progress("embedding", f"Embedding {category}/{img_path.name}", i + 1, total)
                embedding = embedder.embed_media(img_path)
                if embedding is None:
                    continue
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                try:
                    img = Image.open(img_path)
                    width, height = img.width, img.height
                except Exception:
                    width, height = None, None
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": f"{category}/{img_path.name}",
                    "category": category,
                    "width": width,
                    "height": height,
                    "origin": demo_origin,
                    "origin_name": f"{category}/{img_path.name}",
                }
                clip_id += 1

        if source in ("caltech101", "caltech256"):
            if source == "caltech101":
                from vtsearch.datasets.downloader import download_caltech101  # noqa: PLC0415

                img_dir = download_caltech101(on_progress=on_progress)
            else:
                from vtsearch.datasets.downloader import download_caltech256  # noqa: PLC0415

                img_dir = download_caltech256(on_progress=on_progress)

            metadata = load_image_metadata_from_folders(img_dir, categories)
            by_cat: dict[str, list[tuple[Path, str]]] = {}
            for fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected: list[tuple[Path, str]] = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "oxford_flowers_102":
            from vtsearch.datasets.downloader import download_oxford_flowers  # noqa: PLC0415
            from vtsearch.datasets.loader import load_oxford_flowers_metadata  # noqa: PLC0415

            flowers_dir = download_oxford_flowers(on_progress=on_progress)
            metadata = load_oxford_flowers_metadata(flowers_dir, self._OXFORD_FLOWERS_CATEGORIES)

            by_cat: dict[str, list[tuple[Path, str]]] = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                if cat in categories:
                    by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected: list[tuple[Path, str]] = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source in ("food101", "eurosat"):
            if source == "food101":
                from vtsearch.datasets.downloader import download_food101  # noqa: PLC0415

                img_dir = download_food101(on_progress=on_progress)
            else:
                from vtsearch.datasets.downloader import download_eurosat  # noqa: PLC0415

                img_dir = download_eurosat(on_progress=on_progress)

            metadata = load_image_metadata_from_folders(img_dir, categories)
            by_cat = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "stanford_dogs":
            from vtsearch.datasets.downloader import download_stanford_dogs  # noqa: PLC0415

            images_dir = download_stanford_dogs(on_progress=on_progress)

            breed_to_folder: dict[str, Path] = {}
            if images_dir.exists():
                for folder in images_dir.iterdir():
                    if folder.is_dir() and "-" in folder.name:
                        breed_name = folder.name.split("-", 1)[1]
                        breed_to_folder[breed_name] = folder

            by_cat = {}
            for cat in categories:
                folder = breed_to_folder.get(cat)
                if folder is None:
                    continue
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in sorted(folder.glob(ext)):
                        by_cat.setdefault(cat, []).append((img_path, cat))

            selected = []
            for cat in categories:
                selected.extend(by_cat.get(cat, [])[slice_start:slice_end])

            _embed_file_images(selected)
            return None

        elif source == "ucsf_documents":
            from vtsearch.datasets.downloader import download_ucsf_documents  # noqa: PLC0415
            from vtsearch.datasets.pdf import render_pdf_pages  # noqa: PLC0415

            docs_dir = download_ucsf_documents(categories, on_progress=on_progress)

            by_cat_pages: dict[str, list[tuple[str, "Image.Image"]]] = {}
            for cat in categories:
                cat_dir = docs_dir / cat
                if not cat_dir.is_dir():
                    continue
                for pdf_path in sorted(cat_dir.glob("*.pdf")):
                    try:
                        pages = render_pdf_pages(pdf_path, dpi=150)
                        if pages:
                            by_cat_pages.setdefault(cat, []).append(pages[0])
                    except Exception:
                        continue

            selected_pages: list[tuple[str, "Image.Image", str]] = []
            for cat in categories:
                for page_name, pil_image in by_cat_pages.get(cat, [])[slice_start:slice_end]:
                    selected_pages.append((page_name, pil_image, cat))

            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                original_cb = embedder._on_progress
                embedder._on_progress = on_progress
                try:
                    embedder.load_models()
                finally:
                    embedder._on_progress = original_cb

            clip_id = max(clips.keys(), default=0) + 1
            total = len(selected_pages)
            on_progress("embedding", f"Starting embedding for {total} document pages...", 0, total)

            for i, (page_name, pil_image, category) in enumerate(selected_pages):
                on_progress("embedding", f"Embedding {page_name}", i + 1, total)
                embedding = embedder.embed_pil_image(pil_image)
                if embedding is None:
                    continue
                img_buffer = _io.BytesIO()
                pil_image.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()
                rel_name = f"{category}/{page_name}"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": f"{rel_name}.png",
                    "category": category,
                    "width": pil_image.width,
                    "height": pil_image.height,
                    "origin": demo_origin,
                    "origin_name": rel_name,
                }
                clip_id += 1
            return None

        elif source == "cifar10_sample" or not source:
            from vtsearch.datasets.downloader import download_cifar10  # noqa: PLC0415
            from vtsearch.datasets.loader import load_cifar10_batch  # noqa: PLC0415

            cifar_dir = download_cifar10(on_progress=on_progress)
            batch_file = cifar_dir / "data_batch_1"
            images, labels, label_names = load_cifar10_batch(batch_file)
            category_indices = {label_names[i]: i for i in range(len(label_names))}

            selected_images = []
            selected_labels = []
            for cat in categories:
                if cat in category_indices:
                    cat_idx = category_indices[cat]
                    cat_mask = [i for i, lbl in enumerate(labels) if lbl == cat_idx]
                    for idx in cat_mask[slice_start:(slice_end or len(cat_mask))]:
                        selected_images.append(images[idx])
                        selected_labels.append(cat)

            if getattr(embedder, "_model", None) is None:
                on_progress("loading", "Loading image embedding model…", 0, 0)
                original_cb = embedder._on_progress
                embedder._on_progress = on_progress
                try:
                    embedder.load_models()
                finally:
                    embedder._on_progress = original_cb

            clip_id = max(clips.keys(), default=0) + 1
            total = len(selected_images)
            on_progress("embedding", f"Starting embedding for {total} images...", 0, total)

            for i, (image_array, category) in enumerate(zip(selected_images, selected_labels)):
                on_progress("embedding", f"Embedding {category}", i + 1, total)
                img = Image.fromarray(image_array.astype("uint8"), "RGB")
                img_buffer = _io.BytesIO()
                img.save(img_buffer, format="PNG")
                image_bytes = img_buffer.getvalue()
                embedding = embedder.embed_pil_image(img)
                if embedding is None:
                    continue
                fname = f"{category}/{category}_{clip_id}.png"
                clips[clip_id] = {
                    "id": clip_id,
                    "type": self.type_id,
                    "embedder": embedder.name,
                    "duration": 0,
                    "file_size": len(image_bytes),
                    "md5": hashlib.md5(image_bytes).hexdigest(),
                    "embedding": embedding,
                    "media_bytes": image_bytes,
                    "media_string": None,
                    "filename": fname,
                    "category": category,
                    "width": img.width,
                    "height": img.height,
                    "origin": demo_origin,
                    "origin_name": fname,
                }
                clip_id += 1
            return None

        else:
            raise ValueError(f"Unsupported image source: {source!r}")

    # ------------------------------------------------------------------
    # Clip data
    # ------------------------------------------------------------------

    def load_media_data(self, file_path: Path) -> dict:
        from PIL import Image  # noqa: PLC0415

        with open(file_path, "rb") as f:
            media_bytes = f.read()
        try:
            img = Image.open(file_path)
            width, height = img.width, img.height
        except Exception:
            width, height = None, None
        return {
            "media_bytes": media_bytes,
            "duration": 0,
            "width": width,
            "height": height,
        }

    # ------------------------------------------------------------------
    # HTTP serving
    # ------------------------------------------------------------------

    def media_response(self, media: dict) -> MediaResponse:
        filename = media.get("filename", "")
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        mimetype = _IMAGE_MIME_TYPES.get(ext, "image/jpeg")
        data = self._resolve_media_bytes(media)
        if data is None:
            return MediaResponse(data=b"", mimetype=mimetype, download_name=f"media_{media['id']}{ext}")
        return MediaResponse(
            data=data,
            mimetype=mimetype,
            download_name=f"media_{media['id']}{ext}",
        )
