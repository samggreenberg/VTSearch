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
# Audio eval datasets  (ESC-50)
# ------------------------------------------------------------------

_SOUNDS_S_QUERIES = [
    EvalQuery("a dog barking", "dog"),
    EvalQuery("a cat meowing", "cat"),
    EvalQuery("a rooster crowing at dawn", "rooster"),
    EvalQuery("church bells ringing", "church_bells"),
    EvalQuery("crackling fire in a fireplace", "crackling_fire"),
]

_SOUNDS_M_QUERIES = [
    EvalQuery("a baby crying", "crying_baby"),
    EvalQuery("people laughing", "laughing"),
    EvalQuery("hands clapping", "clapping"),
    EvalQuery("footsteps walking", "footsteps"),
    EvalQuery("someone sneezing", "sneezing"),
    EvalQuery("a chainsaw cutting wood", "chainsaw"),
    EvalQuery("an airplane flying overhead", "airplane"),
    EvalQuery("fireworks exploding", "fireworks"),
    EvalQuery("a pig oinking", "pig"),
    EvalQuery("a cow mooing", "cow"),
]

_SOUNDS_L_QUERIES = [
    EvalQuery("birds singing and chirping", "chirping_birds"),
    EvalQuery("a crow cawing", "crow"),
    EvalQuery("frogs croaking near a pond", "frog"),
    EvalQuery("buzzing insects", "insects"),
    EvalQuery("rain falling", "rain"),
    EvalQuery("ocean waves crashing on shore", "sea_waves"),
    EvalQuery("thunderstorm with loud thunder", "thunderstorm"),
    EvalQuery("strong wind blowing", "wind"),
    EvalQuery("dripping water drops", "water_drops"),
    EvalQuery("crickets chirping at night", "crickets"),
    EvalQuery("a car horn honking", "car_horn"),
    EvalQuery("emergency siren wailing", "siren"),
    EvalQuery("engine running and revving", "engine"),
    EvalQuery("a train passing by on rails", "train"),
    EvalQuery("helicopter flying overhead", "helicopter"),
    EvalQuery("vacuum cleaner running", "vacuum_cleaner"),
    EvalQuery("washing machine spinning", "washing_machine"),
    EvalQuery("an alarm clock ringing", "clock_alarm"),
    EvalQuery("someone typing on a keyboard", "keyboard_typing"),
    EvalQuery("knocking on a wooden door", "door_wood_knock"),
]

# ------------------------------------------------------------------
# Image eval datasets  (Caltech-101)
# ------------------------------------------------------------------

_IMAGES_S_QUERIES = [
    EvalQuery("a photograph of a butterfly", "butterfly"),
    EvalQuery("a photograph of a sunflower", "sunflower"),
    EvalQuery("a photograph of a starfish", "starfish"),
    EvalQuery("a photograph of a helicopter", "helicopter"),
]

_IMAGES_M_QUERIES = [
    EvalQuery("a photograph of a dolphin", "dolphin"),
    EvalQuery("a photograph of a grand piano", "grand_piano"),
    EvalQuery("a photograph of an elephant", "elephant"),
    EvalQuery("a photograph of a kangaroo", "kangaroo"),
    EvalQuery("a photograph of a laptop computer", "laptop"),
    EvalQuery("a photograph of a lobster", "lobster"),
    EvalQuery("a photograph of a wristwatch", "watch"),
    EvalQuery("a photograph of a flamingo", "flamingo"),
]

_IMAGES_L_QUERIES = [
    EvalQuery("a photograph of a scorpion", "scorpion"),
    EvalQuery("a photograph of a stop sign", "stop_sign"),
    EvalQuery("a photograph of a chandelier", "chandelier"),
    EvalQuery("a photograph of a rhinoceros", "rhino"),
    EvalQuery("a photograph of a rooster", "rooster"),
    EvalQuery("a photograph of a soccer ball", "soccer_ball"),
    EvalQuery("a photograph of a yin-yang symbol", "yin_yang"),
    EvalQuery("a photograph of a leopard", "leopards"),
    EvalQuery("a photograph of a hawksbill sea turtle", "hawksbill"),
    EvalQuery("a photograph of a revolver", "revolver"),
    EvalQuery("a photograph of a schooner sailing ship", "schooner"),
    EvalQuery("a photograph of an ibis bird", "ibis"),
    EvalQuery("a photograph of a trilobite fossil", "trilobite"),
    EvalQuery("a photograph of a ceiling fan", "ceiling_fan"),
    EvalQuery("a photograph of a dalmatian dog", "dalmatian"),
]

# ------------------------------------------------------------------
# Text / paragraph eval datasets  (20 Newsgroups)
# ------------------------------------------------------------------

_PARAGRAPHS_S_QUERIES = [
    EvalQuery("baseball games and athletic competition", "sports"),
    EvalQuery("outer space exploration and astronomy", "science"),
]

_PARAGRAPHS_M_QUERIES = [
    EvalQuery("international politics and world affairs", "world"),
    EvalQuery("buying and selling goods and products", "business"),
    EvalQuery("computer graphics and rendering", "technology"),
    EvalQuery("medical treatment and healthcare", "medicine"),
]

_PARAGRAPHS_L_QUERIES = [
    EvalQuery("automobiles and car reviews", "cars"),
    EvalQuery("ice hockey games and NHL scores", "hockey"),
    EvalQuery("electronic circuits and components", "electronics"),
    EvalQuery("encryption and computer security", "crypto"),
    EvalQuery("christian faith and religious practice", "religion"),
    EvalQuery("firearms and gun control debate", "guns"),
    EvalQuery("arguments about atheism and belief", "atheism"),
    EvalQuery("apple macintosh hardware and troubleshooting", "mac"),
]

# ------------------------------------------------------------------
# Video eval datasets  (UCF-101)
# ------------------------------------------------------------------

_ACTIVITIES_VIDEO_QUERIES = [
    EvalQuery("someone applying eye makeup", "ApplyEyeMakeup"),
    EvalQuery("someone applying lipstick", "ApplyLipstick"),
    EvalQuery("someone brushing their teeth", "BrushingTeeth"),
    EvalQuery("a person playing drums", "Drumming"),
    EvalQuery("a person doing yo-yo tricks", "YoYo"),
]

_SPORTS_VIDEO_QUERIES = [
    EvalQuery("a person diving off a cliff", "CliffDiving"),
    EvalQuery("someone walking on their hands", "HandstandWalking"),
    EvalQuery("a person jumping rope", "JumpRope"),
    EvalQuery("someone doing push-ups", "PushUps"),
    EvalQuery("a person practicing tai chi", "TaiChi"),
]

# ------------------------------------------------------------------
# Registry — keyed by demo dataset ID
# ------------------------------------------------------------------

EVAL_DATASETS: dict[str, dict] = {
    # Audio
    "sounds_s": {
        "demo_dataset": "sounds_s",
        "queries": _SOUNDS_S_QUERIES,
    },
    "sounds_m": {
        "demo_dataset": "sounds_m",
        "queries": _SOUNDS_M_QUERIES,
    },
    "sounds_l": {
        "demo_dataset": "sounds_l",
        "queries": _SOUNDS_L_QUERIES,
    },
    # Image
    "images_s": {
        "demo_dataset": "images_s",
        "queries": _IMAGES_S_QUERIES,
    },
    "images_m": {
        "demo_dataset": "images_m",
        "queries": _IMAGES_M_QUERIES,
    },
    "images_l": {
        "demo_dataset": "images_l",
        "queries": _IMAGES_L_QUERIES,
    },
    # Text
    "paragraphs_s": {
        "demo_dataset": "paragraphs_s",
        "queries": _PARAGRAPHS_S_QUERIES,
    },
    "paragraphs_m": {
        "demo_dataset": "paragraphs_m",
        "queries": _PARAGRAPHS_M_QUERIES,
    },
    "paragraphs_l": {
        "demo_dataset": "paragraphs_l",
        "queries": _PARAGRAPHS_L_QUERIES,
    },
    # Video
    "activities_video": {
        "demo_dataset": "activities_video",
        "queries": _ACTIVITIES_VIDEO_QUERIES,
    },
    "sports_video": {
        "demo_dataset": "sports_video",
        "queries": _SPORTS_VIDEO_QUERIES,
    },
}
