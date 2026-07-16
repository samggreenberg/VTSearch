"""Eval dataset configurations built from demo datasets.

Each eval dataset wraps a demo dataset and adds per-category text
descriptions - the queries a user would type in the Text Sort box.

The ``EVAL_DATASETS`` dict is keyed by demo dataset ID.  Each value is
a dict with:

- ``"demo_dataset"``: the demo dataset ID to load
- ``"queries"``: list of :class:`EvalQuery`, one per category
"""

from __future__ import annotations

from dataclasses import dataclass

from vtscore.media.image._demo_categories import VGGFACE2_CATEGORIES


@dataclass
class EvalQuery:
    """One evaluation query: a text description targeting a single category."""

    text: str
    """The natural-language query to embed (what a user would type)."""

    target_category: str
    """The ground-truth category name that should rank highest."""


# ------------------------------------------------------------------
# Audio eval queries  (ESC-50 - all 50 categories)
# All S/M/L share the same categories; queries are identical.
# ------------------------------------------------------------------

_SOUNDS_QUERIES = [
    # Animals
    EvalQuery("a dog barking", "dog"),
    EvalQuery("a rooster crowing at dawn", "rooster"),
    EvalQuery("a pig oinking", "pig"),
    EvalQuery("a cow mooing", "cow"),
    EvalQuery("frogs croaking", "frog"),
    EvalQuery("a cat meowing", "cat"),
    EvalQuery("a hen clucking", "hen"),
    EvalQuery("buzzing insects", "insects"),
    EvalQuery("sheep bleating", "sheep"),
    EvalQuery("a crow cawing", "crow"),
    # Natural soundscapes
    EvalQuery("rain falling", "rain"),
    EvalQuery("ocean waves crashing on the shore", "sea_waves"),
    EvalQuery("crackling fire in a fireplace", "crackling_fire"),
    EvalQuery("crickets chirping at night", "crickets"),
    EvalQuery("birds singing and chirping", "chirping_birds"),
    EvalQuery("dripping water drops", "water_drops"),
    EvalQuery("wind howling", "wind"),
    EvalQuery("pouring water from a container", "pouring_water"),
    EvalQuery("a toilet flushing", "toilet_flush"),
    EvalQuery("thunderstorm with loud thunder", "thunderstorm"),
    # Human, non-speech
    EvalQuery("a baby crying", "crying_baby"),
    EvalQuery("someone sneezing", "sneezing"),
    EvalQuery("people clapping and applauding", "clapping"),
    EvalQuery("heavy breathing", "breathing"),
    EvalQuery("someone coughing", "coughing"),
    EvalQuery("footsteps walking", "footsteps"),
    EvalQuery("people laughing", "laughing"),
    EvalQuery("brushing teeth", "brushing_teeth"),
    EvalQuery("someone snoring loudly", "snoring"),
    EvalQuery("drinking and sipping", "drinking_sipping"),
    # Interior / domestic
    EvalQuery("knocking on a wooden door", "door_wood_knock"),
    EvalQuery("mouse clicking", "mouse_click"),
    EvalQuery("keyboard typing", "keyboard_typing"),
    EvalQuery("a creaking wooden door", "door_wood_creep"),
    EvalQuery("opening a can", "can_opening"),
    EvalQuery("a washing machine running", "washing_machine"),
    EvalQuery("a vacuum cleaner", "vacuum_cleaner"),
    EvalQuery("an alarm clock ringing", "clock_alarm"),
    EvalQuery("a clock ticking", "clock_tick"),
    EvalQuery("glass shattering", "glass_breaking"),
    # Exterior / urban
    EvalQuery("a helicopter flying", "helicopter"),
    EvalQuery("a chainsaw cutting wood", "chainsaw"),
    EvalQuery("an emergency siren", "siren"),
    EvalQuery("a car horn honking", "car_horn"),
    EvalQuery("an engine revving", "engine"),
    EvalQuery("a train passing by", "train"),
    EvalQuery("church bells ringing", "church_bells"),
    EvalQuery("an airplane flying overhead", "airplane"),
    EvalQuery("fireworks exploding", "fireworks"),
    EvalQuery("hand-sawing wood", "hand_saw"),
]

# ------------------------------------------------------------------
# Image eval queries  (Caltech-101 - 25 categories for S/M)
# ------------------------------------------------------------------

_IMAGES_QUERIES = [
    EvalQuery("a photograph of an airplane", "airplanes"),
    EvalQuery("a photograph of a bonsai tree", "bonsai"),
    EvalQuery("a photograph of a car from the side", "car_side"),
    EvalQuery("a photograph of a chandelier", "chandelier"),
    EvalQuery("a photograph of a cougar face", "cougar_face"),
    EvalQuery("a photograph of a crab", "crab"),
    EvalQuery("a photograph of a dalmatian dog", "dalmatian"),
    EvalQuery("a photograph of a dolphin", "dolphin"),
    EvalQuery("a photograph of an elephant", "elephant"),
    EvalQuery("a photograph of a ferry boat", "ferry"),
    EvalQuery("a photograph of a flamingo", "flamingo"),
    EvalQuery("a photograph of a grand piano", "grand_piano"),
    EvalQuery("a photograph of a hawksbill turtle", "hawksbill"),
    EvalQuery("a photograph of a helicopter", "helicopter"),
    EvalQuery("a photograph of an ibis bird", "ibis"),
    EvalQuery("a photograph of a kangaroo", "kangaroo"),
    EvalQuery("a photograph of a ketch sailing boat", "ketch"),
    EvalQuery("a photograph of a lamp", "lamp"),
    EvalQuery("a photograph of a laptop computer", "laptop"),
    EvalQuery("a photograph of a nautilus shell", "nautilus"),
    EvalQuery("a photograph of a starfish", "starfish"),
    EvalQuery("a photograph of a stop sign", "stop_sign"),
    EvalQuery("a photograph of a sunflower", "sunflower"),
    EvalQuery("a photograph of a trilobite fossil", "trilobite"),
    EvalQuery("a photograph of a wristwatch", "watch"),
]

# ------------------------------------------------------------------
# Image eval queries  (Caltech-256 - 25 categories for L)
# ------------------------------------------------------------------

_IMAGES_L_QUERIES = [
    EvalQuery("a photograph of a backpack", "003.backpack"),
    EvalQuery("a photograph of a butterfly", "024.butterfly"),
    EvalQuery("a photograph of a camel", "028.camel"),
    EvalQuery("a photograph of a cannon", "029.cannon"),
    EvalQuery("a photograph of a chimpanzee", "038.chimp"),
    EvalQuery("a photograph of a computer monitor", "046.computer-monitor"),
    EvalQuery("a photograph of a fighter jet", "069.fighter-jet"),
    EvalQuery("a photograph of a giraffe", "084.giraffe"),
    EvalQuery("a photograph of a goat", "085.goat"),
    EvalQuery("a photograph of the Golden Gate Bridge", "086.golden-gate-bridge"),
    EvalQuery("a photograph of a hammock", "096.hammock"),
    EvalQuery("a photograph of a hot air balloon", "107.hot-air-balloon"),
    EvalQuery("a photograph of a kayak", "122.kayak"),
    EvalQuery("a photograph of a leopard", "129.leopards-101"),
    EvalQuery("a photograph of a lighthouse", "132.light-house"),
    EvalQuery("a photograph of a motorbike", "145.motorbikes-101"),
    EvalQuery("a photograph of a mushroom", "147.mushroom"),
    EvalQuery("a photograph of an ostrich", "151.ostrich"),
    EvalQuery("a photograph of a penguin", "158.penguin"),
    EvalQuery("a photograph of a pyramid", "167.pyramid"),
    EvalQuery("a photograph of a school bus", "178.school-bus"),
    EvalQuery("a photograph of stained glass", "200.stained-glass"),
    EvalQuery("a photograph of a swan", "207.swan"),
    EvalQuery("a photograph of a windmill", "245.windmill"),
    EvalQuery("a photograph of a wine bottle", "246.wine-bottle"),
]

# ------------------------------------------------------------------
# Image eval queries  (Enrico - mobile UI screenshots by screen function)
# Born-digital screenshots, so the queries describe the *screen's function*
# ("a login screen", "a chat conversation") rather than a photographed subject.
# ``Bare`` and ``Other`` carry no describable subject and are omitted.
# ------------------------------------------------------------------

_ENRICO_QUERIES = [
    EvalQuery("a camera viewfinder screen", "Camera"),
    EvalQuery("a chat conversation screen with message bubbles", "Chat"),
    EvalQuery("a phone dialer with a numeric keypad", "Dialer"),
    EvalQuery("a text or document editor screen", "Editor"),
    EvalQuery("a form with input fields to fill in", "Form"),
    EvalQuery("a grid gallery of photos or thumbnails", "Gallery"),
    EvalQuery("a scrollable list of items", "List"),
    EvalQuery("a login or sign-in screen", "Login"),
    EvalQuery("a map view showing streets", "Maps"),
    EvalQuery("a media player with playback controls", "MediaPlayer"),
    EvalQuery("a navigation menu or drawer", "Menu"),
    EvalQuery("a modal dialog or popup overlay", "Modal"),
    EvalQuery("a news or article feed", "News"),
    EvalQuery("a user profile screen", "Profile"),
    EvalQuery("a search screen with a search bar and results", "Search"),
    EvalQuery("a settings or preferences screen", "Settings"),
    EvalQuery("a terms of service or legal text screen", "Terms"),
    EvalQuery("an onboarding or tutorial walkthrough screen", "Tutorial"),
]

# ------------------------------------------------------------------
# Image eval queries  (RICO-Screen2Words - mobile UI screenshots by app genre)
# The label is the app's Google Play category, so the query describes the kind
# of app the screen belongs to.  A harder signal than Enrico's screen function:
# two apps in different genres can share a screen layout, so this stresses
# whether the embedder captures app-domain semantics from a screenshot.
# ------------------------------------------------------------------

_RICO_SCREEN2WORDS_QUERIES = [
    EvalQuery("a screen from a books or reference reading app", "Books & Reference"),
    EvalQuery("a screen from a business app", "Business"),
    EvalQuery("a screen from a messaging or communication app", "Communication"),
    EvalQuery("a screen from a banking or finance app", "Finance"),
    EvalQuery("a screen from a food or drink ordering app", "Food & Drink"),
    EvalQuery("a screen from a health or fitness tracking app", "Health & Fitness"),
    EvalQuery("a screen from a maps or navigation app", "Maps & Navigation"),
    EvalQuery("a screen from a music or audio streaming app", "Music & Audio"),
    EvalQuery("a screen from a news or magazine app", "News & Magazines"),
    EvalQuery("a screen from a photography or camera app", "Photography"),
    EvalQuery("a screen from a productivity app", "Productivity"),
    EvalQuery("a screen from a shopping or e-commerce app", "Shopping"),
    EvalQuery("a screen from a social networking app", "Social"),
    EvalQuery("a screen from a sports app", "Sports"),
    EvalQuery("a screen from a travel or local guide app", "Travel & Local"),
    EvalQuery("a screen from a weather forecast app", "Weather"),
]

# ------------------------------------------------------------------
# Image eval queries  (RVL-CDIP - scanned document images by type)
# Digitally-native document images (not photos); the query describes the kind
# of document.  All 16 classes are describable, so every one gets a query.
# ------------------------------------------------------------------

_RVL_CDIP_QUERIES = [
    EvalQuery("a printed advertisement", "advertisement"),
    EvalQuery("a budget spreadsheet with numbers and totals", "budget"),
    EvalQuery("an email message with headers", "email"),
    EvalQuery("a file folder cover sheet", "file folder"),
    EvalQuery("a blank or filled paper form", "form"),
    EvalQuery("a handwritten note or letter", "handwritten"),
    EvalQuery("an invoice or bill", "invoice"),
    EvalQuery("a typed business letter", "letter"),
    EvalQuery("an office memo", "memo"),
    EvalQuery("a newspaper or news article page", "news article"),
    EvalQuery("presentation slides", "presentation"),
    EvalQuery("a questionnaire or survey", "questionnaire"),
    EvalQuery("a resume or CV", "resume"),
    EvalQuery("a scientific journal publication page", "scientific publication"),
    EvalQuery("a scientific or technical report", "scientific report"),
    EvalQuery("a technical specification document", "specification"),
]

# ------------------------------------------------------------------
# Text / paragraph eval queries  (20 Newsgroups - 15 categories)
# All S/M/L share the same categories; queries are identical.
# ------------------------------------------------------------------

_PARAGRAPHS_QUERIES = [
    EvalQuery("baseball games and athletic competition", "sports"),
    EvalQuery("outer space exploration and astronomy", "science"),
    EvalQuery("automobiles and car reviews", "cars"),
    EvalQuery("ice hockey games and NHL scores", "hockey"),
    EvalQuery("electronic circuits and components", "electronics"),
    EvalQuery("christian faith and religious practice", "religion"),
    EvalQuery("world politics and government affairs", "world"),
    EvalQuery("buying and selling merchandise", "business"),
    EvalQuery("computer graphics and image rendering", "technology"),
    EvalQuery("medical diseases and health treatments", "medicine"),
    EvalQuery("cryptography and data encryption", "crypto"),
    EvalQuery("atheism and arguments against religion", "atheism"),
    EvalQuery("motorcycle riding and maintenance", "motorcycles"),
    EvalQuery("middle east politics and conflict", "mideast"),
    EvalQuery("firearms and gun legislation", "guns"),
]

# ------------------------------------------------------------------
# Video eval queries  (UCF-101 - 10 shared categories for S/M/L)
# ------------------------------------------------------------------

_VIDEO_QUERIES = [
    EvalQuery("someone applying eye makeup", "ApplyEyeMakeup"),
    EvalQuery("someone applying lipstick", "ApplyLipstick"),
    EvalQuery("a person shooting a bow and arrow", "Archery"),
    EvalQuery("a baby crawling on the floor", "BabyCrawling"),
    EvalQuery("a gymnast on the balance beam", "BalanceBeam"),
    EvalQuery("a marching band performing", "BandMarching"),
    EvalQuery("a baseball pitcher throwing a pitch", "BaseballPitch"),
    EvalQuery("people playing basketball", "Basketball"),
    EvalQuery("a basketball dunk", "BasketballDunk"),
    EvalQuery("a person doing bench presses", "BenchPress"),
]

# ------------------------------------------------------------------
# Image (multi-label) eval queries  (Visual Genome)
# Unlike the single-label image datasets, a VG image can be a positive
# example of several of these targets at once; membership is decided by
# ``media["categories"]`` (see vtscore/eval/labels.py).  Each target below
# is one of the VG object vocab categories.
# ------------------------------------------------------------------

_VISUAL_GENOME_QUERIES = [
    EvalQuery("a person", "person"),
    EvalQuery("a man", "man"),
    EvalQuery("a woman", "woman"),
    EvalQuery("a car on the street", "car"),
    EvalQuery("a bus", "bus"),
    EvalQuery("a train", "train"),
    EvalQuery("a truck", "truck"),
    EvalQuery("a bicycle", "bike"),
    EvalQuery("a boat on the water", "boat"),
    EvalQuery("an airplane", "plane"),
    EvalQuery("a dog", "dog"),
    EvalQuery("a cat", "cat"),
    EvalQuery("a horse", "horse"),
    EvalQuery("an elephant", "elephant"),
    EvalQuery("a giraffe", "giraffe"),
    EvalQuery("a zebra", "zebra"),
    EvalQuery("a cow", "cow"),
    EvalQuery("a sheep", "sheep"),
    EvalQuery("a bird", "bird"),
    EvalQuery("a bear", "bear"),
    EvalQuery("a building", "building"),
    EvalQuery("a window", "window"),
    EvalQuery("a tree", "tree"),
    EvalQuery("a street sign", "sign"),
    EvalQuery("a clock", "clock"),
    EvalQuery("an umbrella", "umbrella"),
    EvalQuery("a chair", "chair"),
    EvalQuery("a table", "table"),
    EvalQuery("a bench", "bench"),
    EvalQuery("a bottle", "bottle"),
    EvalQuery("a plate of food", "plate"),
    EvalQuery("a pizza", "pizza"),
    EvalQuery("a banana", "banana"),
    EvalQuery("a laptop computer", "laptop"),
    EvalQuery("a kite in the sky", "kite"),
    EvalQuery("a skateboard", "skateboard"),
    EvalQuery("a surfboard", "surfboard"),
    EvalQuery("a person wearing a hat", "hat"),
    EvalQuery("a person wearing a jacket", "jacket"),
    EvalQuery("a snowy scene", "snow"),
]

# ------------------------------------------------------------------
# Faces eval queries  (VGGFace2 - one query per curated identity)
# ------------------------------------------------------------------
# Identity is the ground-truth category, so the meaningful mode here is
# learned sort: vote a few of a person's photos Good, train the MLP, and
# measure whether their held-out photos are recovered.  The text is the
# person's name so text-sort still has a (weaker) name->photo query to score.

_FACES_QUERIES = [EvalQuery(f"a photo of {name}", name) for name in VGGFACE2_CATEGORIES]


# ------------------------------------------------------------------
# Registry - keyed by demo dataset ID
# ------------------------------------------------------------------

EVAL_DATASETS: dict[str, dict] = {
    # Audio
    "esc50_s": {
        "demo_dataset": "esc50_s",
        "queries": _SOUNDS_QUERIES,
    },
    "esc50_m": {
        "demo_dataset": "esc50_m",
        "queries": _SOUNDS_QUERIES,
    },
    "esc50_l": {
        "demo_dataset": "esc50_l",
        "queries": _SOUNDS_QUERIES,
    },
    # Image
    "caltech101_s": {
        "demo_dataset": "caltech101_s",
        "queries": _IMAGES_QUERIES,
    },
    "caltech101_m": {
        "demo_dataset": "caltech101_m",
        "queries": _IMAGES_QUERIES,
    },
    "caltech256_a": {
        "demo_dataset": "caltech256_a",
        "queries": _IMAGES_L_QUERIES,
    },
    # Image (faces - same-person identity matching)
    "vggface2_faces_s": {
        "demo_dataset": "vggface2_faces_s",
        "queries": _FACES_QUERIES,
    },
    "vggface2_faces_m": {
        "demo_dataset": "vggface2_faces_m",
        "queries": _FACES_QUERIES,
    },
    # Image (born-digital mobile UI screenshots)
    "enrico_m": {
        "demo_dataset": "enrico_m",
        "queries": _ENRICO_QUERIES,
    },
    "enrico_a": {
        "demo_dataset": "enrico_a",
        "queries": _ENRICO_QUERIES,
    },
    "rico_screen2words_m": {
        "demo_dataset": "rico_screen2words_m",
        "queries": _RICO_SCREEN2WORDS_QUERIES,
    },
    "rico_screen2words_a": {
        "demo_dataset": "rico_screen2words_a",
        "queries": _RICO_SCREEN2WORDS_QUERIES,
    },
    # Image (scanned document images)
    "rvl_cdip_m": {
        "demo_dataset": "rvl_cdip_m",
        "queries": _RVL_CDIP_QUERIES,
    },
    "rvl_cdip_a": {
        "demo_dataset": "rvl_cdip_a",
        "queries": _RVL_CDIP_QUERIES,
    },
    # Image (multi-label)
    "visual_genome_s": {
        "demo_dataset": "visual_genome_s",
        "queries": _VISUAL_GENOME_QUERIES,
    },
    "visual_genome_m": {
        "demo_dataset": "visual_genome_m",
        "queries": _VISUAL_GENOME_QUERIES,
    },
    # Text
    "20newsgroups_s": {
        "demo_dataset": "20newsgroups_s",
        "queries": _PARAGRAPHS_QUERIES,
    },
    "20newsgroups_m": {
        "demo_dataset": "20newsgroups_m",
        "queries": _PARAGRAPHS_QUERIES,
    },
    "20newsgroups_l": {
        "demo_dataset": "20newsgroups_l",
        "queries": _PARAGRAPHS_QUERIES,
    },
    # Video
    "ucf101_s": {
        "demo_dataset": "ucf101_s",
        "queries": _VIDEO_QUERIES,
    },
    "ucf101_m": {
        "demo_dataset": "ucf101_m",
        "queries": _VIDEO_QUERIES,
    },
    "ucf101_l": {
        "demo_dataset": "ucf101_l",
        "queries": _VIDEO_QUERIES,
    },
}
