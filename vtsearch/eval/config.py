"""Eval dataset configurations built from demo datasets.

Each eval dataset wraps a demo dataset and adds per-category text
descriptions — the queries a user would type in the Text Sort box.

The ``EVAL_DATASETS`` dict is keyed by demo dataset ID.  Each value is
a dict with:

- ``"demo_dataset"``: the demo dataset ID to load
- ``"queries"``: list of :class:`EvalQuery`, one per category
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalQuery:
    """One evaluation query: a text description targeting a single category."""

    text: str
    """The natural-language query to embed (what a user would type)."""

    target_category: str
    """The ground-truth category name that should rank highest."""


# ------------------------------------------------------------------
# Audio eval queries  (ESC-50 — all 50 categories)
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
# Image eval queries  (Caltech-101 — 25 categories for S/M)
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
# Image eval queries  (Caltech-256 — 25 categories for L)
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
# Text / paragraph eval queries  (20 Newsgroups — 15 categories)
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
# Video eval queries  (UCF-101 — 10 shared categories for S/M/L)
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
# Registry — keyed by demo dataset ID
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
