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
    demo_slice,
)


_IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


# Raw Places365 category list, copied verbatim from
# https://github.com/CSAILVision/places365/blob/master/categories_places365.txt
# Each line is "<path> <index>" where <path> is "/<letter>/<name>" with an
# optional subcategory.  The list is ordered by index 0..364 so a simple
# splitlines() yields the categories in label-index order.
_PLACES365_CATEGORIES_RAW = """\
/a/airfield 0
/a/airplane_cabin 1
/a/airport_terminal 2
/a/alcove 3
/a/alley 4
/a/amphitheater 5
/a/amusement_arcade 6
/a/amusement_park 7
/a/apartment_building/outdoor 8
/a/aquarium 9
/a/aqueduct 10
/a/arcade 11
/a/arch 12
/a/archaelogical_excavation 13
/a/archive 14
/a/arena/hockey 15
/a/arena/performance 16
/a/arena/rodeo 17
/a/army_base 18
/a/art_gallery 19
/a/art_school 20
/a/art_studio 21
/a/artists_loft 22
/a/assembly_line 23
/a/athletic_field/outdoor 24
/a/atrium/public 25
/a/attic 26
/a/auditorium 27
/a/auto_factory 28
/a/auto_showroom 29
/b/badlands 30
/b/bakery/shop 31
/b/balcony/exterior 32
/b/balcony/interior 33
/b/ball_pit 34
/b/ballroom 35
/b/bamboo_forest 36
/b/bank_vault 37
/b/banquet_hall 38
/b/bar 39
/b/barn 40
/b/barndoor 41
/b/baseball_field 42
/b/basement 43
/b/basketball_court/indoor 44
/b/bathroom 45
/b/bazaar/indoor 46
/b/bazaar/outdoor 47
/b/beach 48
/b/beach_house 49
/b/beauty_salon 50
/b/bedchamber 51
/b/bedroom 52
/b/beer_garden 53
/b/beer_hall 54
/b/berth 55
/b/biology_laboratory 56
/b/boardwalk 57
/b/boat_deck 58
/b/boathouse 59
/b/bookstore 60
/b/booth/indoor 61
/b/botanical_garden 62
/b/bow_window/indoor 63
/b/bowling_alley 64
/b/boxing_ring 65
/b/bridge 66
/b/building_facade 67
/b/bullring 68
/b/burial_chamber 69
/b/bus_interior 70
/b/bus_station/indoor 71
/b/butchers_shop 72
/b/butte 73
/c/cabin/outdoor 74
/c/cafeteria 75
/c/campsite 76
/c/campus 77
/c/canal/natural 78
/c/canal/urban 79
/c/candy_store 80
/c/canyon 81
/c/car_interior 82
/c/carrousel 83
/c/castle 84
/c/catacomb 85
/c/cemetery 86
/c/chalet 87
/c/chemistry_lab 88
/c/childs_room 89
/c/church/indoor 90
/c/church/outdoor 91
/c/classroom 92
/c/clean_room 93
/c/cliff 94
/c/closet 95
/c/clothing_store 96
/c/coast 97
/c/cockpit 98
/c/coffee_shop 99
/c/computer_room 100
/c/conference_center 101
/c/conference_room 102
/c/construction_site 103
/c/corn_field 104
/c/corral 105
/c/corridor 106
/c/cottage 107
/c/courthouse 108
/c/courtyard 109
/c/creek 110
/c/crevasse 111
/c/crosswalk 112
/d/dam 113
/d/delicatessen 114
/d/department_store 115
/d/desert/sand 116
/d/desert/vegetation 117
/d/desert_road 118
/d/diner/outdoor 119
/d/dining_hall 120
/d/dining_room 121
/d/discotheque 122
/d/doorway/outdoor 123
/d/dorm_room 124
/d/downtown 125
/d/dressing_room 126
/d/driveway 127
/d/drugstore 128
/e/elevator/door 129
/e/elevator_lobby 130
/e/elevator_shaft 131
/e/embassy 132
/e/engine_room 133
/e/entrance_hall 134
/e/escalator/indoor 135
/e/excavation 136
/f/fabric_store 137
/f/farm 138
/f/fastfood_restaurant 139
/f/field/cultivated 140
/f/field/wild 141
/f/field_road 142
/f/fire_escape 143
/f/fire_station 144
/f/fishpond 145
/f/flea_market/indoor 146
/f/florist_shop/indoor 147
/f/food_court 148
/f/football_field 149
/f/forest/broadleaf 150
/f/forest_path 151
/f/forest_road 152
/f/formal_garden 153
/f/fountain 154
/g/galley 155
/g/garage/indoor 156
/g/garage/outdoor 157
/g/gas_station 158
/g/gazebo/exterior 159
/g/general_store/indoor 160
/g/general_store/outdoor 161
/g/gift_shop 162
/g/glacier 163
/g/golf_course 164
/g/greenhouse/indoor 165
/g/greenhouse/outdoor 166
/g/grotto 167
/g/gymnasium/indoor 168
/h/hangar/indoor 169
/h/hangar/outdoor 170
/h/harbor 171
/h/hardware_store 172
/h/hayfield 173
/h/heliport 174
/h/highway 175
/h/home_office 176
/h/home_theater 177
/h/hospital 178
/h/hospital_room 179
/h/hot_spring 180
/h/hotel/outdoor 181
/h/hotel_room 182
/h/house 183
/h/hunting_lodge/outdoor 184
/i/ice_cream_parlor 185
/i/ice_floe 186
/i/ice_shelf 187
/i/ice_skating_rink/indoor 188
/i/ice_skating_rink/outdoor 189
/i/iceberg 190
/i/igloo 191
/i/industrial_area 192
/i/inn/outdoor 193
/i/islet 194
/j/jacuzzi/indoor 195
/j/jail_cell 196
/j/japanese_garden 197
/j/jewelry_shop 198
/j/junkyard 199
/k/kasbah 200
/k/kennel/outdoor 201
/k/kindergarden_classroom 202
/k/kitchen 203
/l/lagoon 204
/l/lake/natural 205
/l/landfill 206
/l/landing_deck 207
/l/laundromat 208
/l/lawn 209
/l/lecture_room 210
/l/legislative_chamber 211
/l/library/indoor 212
/l/library/outdoor 213
/l/lighthouse 214
/l/living_room 215
/l/loading_dock 216
/l/lobby 217
/l/lock_chamber 218
/l/locker_room 219
/m/mansion 220
/m/manufactured_home 221
/m/market/indoor 222
/m/market/outdoor 223
/m/marsh 224
/m/martial_arts_gym 225
/m/mausoleum 226
/m/medina 227
/m/mezzanine 228
/m/moat/water 229
/m/mosque/outdoor 230
/m/motel 231
/m/mountain 232
/m/mountain_path 233
/m/mountain_snowy 234
/m/movie_theater/indoor 235
/m/museum/indoor 236
/m/museum/outdoor 237
/m/music_studio 238
/n/natural_history_museum 239
/n/nursery 240
/n/nursing_home 241
/o/oast_house 242
/o/ocean 243
/o/office 244
/o/office_building 245
/o/office_cubicles 246
/o/oilrig 247
/o/operating_room 248
/o/orchard 249
/o/orchestra_pit 250
/p/pagoda 251
/p/palace 252
/p/pantry 253
/p/park 254
/p/parking_garage/indoor 255
/p/parking_garage/outdoor 256
/p/parking_lot 257
/p/pasture 258
/p/patio 259
/p/pavilion 260
/p/pet_shop 261
/p/pharmacy 262
/p/phone_booth 263
/p/physics_laboratory 264
/p/picnic_area 265
/p/pier 266
/p/pizzeria 267
/p/playground 268
/p/playroom 269
/p/plaza 270
/p/pond 271
/p/porch 272
/p/promenade 273
/p/pub/indoor 274
/r/racecourse 275
/r/raceway 276
/r/raft 277
/r/railroad_track 278
/r/rainforest 279
/r/reception 280
/r/recreation_room 281
/r/repair_shop 282
/r/residential_neighborhood 283
/r/restaurant 284
/r/restaurant_kitchen 285
/r/restaurant_patio 286
/r/rice_paddy 287
/r/river 288
/r/rock_arch 289
/r/roof_garden 290
/r/rope_bridge 291
/r/ruin 292
/r/runway 293
/s/sandbox 294
/s/sauna 295
/s/schoolhouse 296
/s/science_museum 297
/s/server_room 298
/s/shed 299
/s/shoe_shop 300
/s/shopfront 301
/s/shopping_mall/indoor 302
/s/shower 303
/s/ski_resort 304
/s/ski_slope 305
/s/sky 306
/s/skyscraper 307
/s/slum 308
/s/snowfield 309
/s/soccer_field 310
/s/stable 311
/s/stadium/baseball 312
/s/stadium/football 313
/s/stadium/soccer 314
/s/stage/indoor 315
/s/stage/outdoor 316
/s/staircase 317
/s/storage_room 318
/s/street 319
/s/subway_station/platform 320
/s/supermarket 321
/s/sushi_bar 322
/s/swamp 323
/s/swimming_hole 324
/s/swimming_pool/indoor 325
/s/swimming_pool/outdoor 326
/s/synagogue/outdoor 327
/t/television_room 328
/t/television_studio 329
/t/temple/asia 330
/t/throne_room 331
/t/ticket_booth 332
/t/topiary_garden 333
/t/tower 334
/t/toyshop 335
/t/train_interior 336
/t/train_station/platform 337
/t/tree_farm 338
/t/tree_house 339
/t/trench 340
/t/tundra 341
/u/underwater/ocean_deep 342
/u/utility_room 343
/v/valley 344
/v/vegetable_garden 345
/v/veterinarians_office 346
/v/viaduct 347
/v/village 348
/v/vineyard 349
/v/volcano 350
/v/volleyball_court/outdoor 351
/w/waiting_room 352
/w/water_park 353
/w/water_tower 354
/w/waterfall 355
/w/watering_hole 356
/w/wave 357
/w/wet_bar 358
/w/wheat_field 359
/w/wind_farm 360
/w/windmill 361
/y/yard 362
/y/youth_hostel 363
/z/zen_garden 364
"""


def _parse_places365_categories(raw: str) -> list[str]:
    """Convert raw Places365 category lines into a flat name list.

    Each input line is ``"<path> <index>"`` (e.g. ``"/b/bakery/shop 31"``).
    The leading ``/<letter>/`` prefix is stripped and remaining ``/`` are
    replaced with ``_`` so the result is a flat, filesystem-friendly name
    (``bakery_shop``).  Lines are kept in their original order so the list
    is indexable by the integer label from ``places365_val.txt``.
    """
    out: list[str] = []
    for line in raw.strip().splitlines():
        path = line.rsplit(" ", 1)[0]
        cleaned = path.split("/", 2)[2].replace("/", "_")
        out.append(cleaned)
    return out


_PLACES365_CATEGORIES_LIST = _parse_places365_categories(_PLACES365_CATEGORIES_RAW)


class ImageMediaType(MediaType):
    """Handles image medias — file import, HTTP serving, and demo datasets.

    Embedding is handled by :class:`~vtsearch.media.image.embedder_siglip.ImageSiglipEmbedder`.
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
        return "image"

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
        result.update({k: v for k, v in super().display_metadata(media).items() if k not in result})
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

    _OXFORD_FLOWERS_CATEGORIES = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe-flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink-yellow dahlia",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "ball moss",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]

    _FOOD101_CATEGORIES = [
        "apple_pie",
        "baby_back_ribs",
        "baklava",
        "beef_carpaccio",
        "beef_tartare",
        "beet_salad",
        "beignets",
        "bibimbap",
        "bread_pudding",
        "breakfast_burrito",
        "bruschetta",
        "caesar_salad",
        "cannoli",
        "caprese_salad",
        "carrot_cake",
        "ceviche",
        "cheesecake",
        "cheese_plate",
        "chicken_curry",
        "chicken_quesadilla",
        "chicken_wings",
        "chocolate_cake",
        "chocolate_mousse",
        "churros",
        "clam_chowder",
        "club_sandwich",
        "crab_cakes",
        "creme_brulee",
        "croque_madame",
        "cup_cakes",
        "deviled_eggs",
        "donuts",
        "dumplings",
        "edamame",
        "eggs_benedict",
        "escargots",
        "falafel",
        "filet_mignon",
        "fish_and_chips",
        "foie_gras",
        "french_fries",
        "french_onion_soup",
        "french_toast",
        "fried_calamari",
        "fried_rice",
        "frozen_yogurt",
        "garlic_bread",
        "gnocchi",
        "greek_salad",
        "grilled_cheese_sandwich",
        "grilled_salmon",
        "guacamole",
        "gyoza",
        "hamburger",
        "hot_and_sour_soup",
        "hot_dog",
        "huevos_rancheros",
        "hummus",
        "ice_cream",
        "lasagna",
        "lobster_bisque",
        "lobster_roll_sandwich",
        "macaroni_and_cheese",
        "macarons",
        "miso_soup",
        "mussels",
        "nachos",
        "omelette",
        "onion_rings",
        "oysters",
        "pad_thai",
        "paella",
        "pancakes",
        "panna_cotta",
        "peking_duck",
        "pho",
        "pizza",
        "pork_chop",
        "poutine",
        "prime_rib",
        "pulled_pork_sandwich",
        "ramen",
        "ravioli",
        "red_velvet_cake",
        "risotto",
        "samosa",
        "sashimi",
        "scallops",
        "seaweed_salad",
        "shrimp_and_grits",
        "spaghetti_bolognese",
        "spaghetti_carbonara",
        "spring_rolls",
        "steak",
        "strawberry_shortcake",
        "sushi",
        "tacos",
        "takoyaki",
        "tiramisu",
        "tuna_tartare",
        "waffles",
    ]

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

    _STANFORD_DOGS_CATEGORIES = [
        "Chihuahua",
        "Japanese_spaniel",
        "Maltese_dog",
        "Pekinese",
        "Shih-Tzu",
        "Blenheim_spaniel",
        "papillon",
        "toy_terrier",
        "Rhodesian_ridgeback",
        "Afghan_hound",
        "basset",
        "beagle",
        "bloodhound",
        "bluetick",
        "black-and-tan_coonhound",
        "Walker_hound",
        "English_foxhound",
        "redbone",
        "borzoi",
        "Irish_wolfhound",
        "Italian_greyhound",
        "whippet",
        "Ibizan_hound",
        "Norwegian_elkhound",
        "otterhound",
        "Saluki",
        "Scottish_deerhound",
        "Weimaraner",
        "Staffordshire_bullterrier",
        "American_Staffordshire_terrier",
        "Bedlington_terrier",
        "Border_terrier",
        "Kerry_blue_terrier",
        "Irish_terrier",
        "Norfolk_terrier",
        "Norwich_terrier",
        "Yorkshire_terrier",
        "wire-haired_fox_terrier",
        "Lakeland_terrier",
        "Sealyham_terrier",
        "Airedale",
        "cairn",
        "Australian_terrier",
        "Dandie_Dinmont",
        "Boston_bull",
        "miniature_schnauzer",
        "giant_schnauzer",
        "standard_schnauzer",
        "Scotch_terrier",
        "Tibetan_terrier",
        "silky_terrier",
        "soft-coated_wheaten_terrier",
        "West_Highland_white_terrier",
        "Lhasa",
        "flat-coated_retriever",
        "curly-coated_retriever",
        "golden_retriever",
        "Labrador_retriever",
        "Chesapeake_Bay_retriever",
        "German_short-haired_pointer",
        "vizsla",
        "English_setter",
        "Irish_setter",
        "Gordon_setter",
        "Brittany_spaniel",
        "clumber",
        "English_springer",
        "Welsh_springer_spaniel",
        "cocker_spaniel",
        "Sussex_spaniel",
        "Irish_water_spaniel",
        "kuvasz",
        "schipperke",
        "groenendael",
        "malinois",
        "briard",
        "kelpie",
        "komondor",
        "Old_English_sheepdog",
        "Shetland_sheepdog",
        "collie",
        "Border_collie",
        "Bouvier_des_Flandres",
        "Rottweiler",
        "German_shepherd",
        "Doberman",
        "miniature_pinscher",
        "Greater_Swiss_Mountain_dog",
        "Bernese_mountain_dog",
        "Appenzeller",
        "EntleBucher",
        "boxer",
        "bull_mastiff",
        "Tibetan_mastiff",
        "French_bulldog",
        "Great_Dane",
        "Saint_Bernard",
        "Eskimo_dog",
        "malamute",
        "Siberian_husky",
        "affenpinscher",
        "basenji",
        "pug",
        "Leonberg",
        "Newfoundland",
        "Great_Pyrenees",
        "Samoyed",
        "Pomeranian",
        "chow",
        "keeshond",
        "Brabancon_griffon",
        "Pembroke",
        "Cardigan",
        "toy_poodle",
        "miniature_poodle",
        "standard_poodle",
        "Mexican_hairless",
        "dingo",
        "dhole",
        "African_hunting_dog",
    ]

    _UCSF_DOCUMENTS_CATEGORIES = [
        "Tobacco",
        "Food",
        "Drug",
        "Chemical",
        "Fossil Fuel",
        "Opioids",
    ]

    # Built from the canonical 365-line categories file at module load time
    # (see ``_PLACES365_CATEGORIES_RAW`` near the top of this module).
    _PLACES365_CATEGORIES = _PLACES365_CATEGORIES_LIST

    @property
    def demo_datasets(self) -> list:
        from vtsearch.datasets.downloader import (  # noqa: PLC0415
            CALTECH101_DOWNLOAD_SIZE_MB,
            CALTECH256_DOWNLOAD_SIZE_MB,
            EUROSAT_DOWNLOAD_SIZE_MB,
            FOOD101_DOWNLOAD_SIZE_MB,
            OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
            PLACES365_DOWNLOAD_SIZE_MB,
            STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            UCSF_IDL_DOWNLOAD_SIZE_MB,
        )

        cats101 = self._DEMO_CATEGORIES_CALTECH101
        cats256 = self._DEMO_CATEGORIES_CALTECH256
        ct101_desc = "Centered object photos — a classic vision benchmark."
        ct101_folder = DATA_DIR / "caltech-101" / "101_ObjectCategories"
        food_desc = "Crowd-sourced food photos — a deliberately noisy benchmark."
        food_folder = DATA_DIR / "food-101" / "images"
        euro_desc = "Sentinel-2 satellite imagery classified by land use type."
        euro_folder = DATA_DIR / "EuroSAT_RGB"
        dogs_desc = "Fine-grained dog breeds with many visually similar classes."
        dogs_folder = DATA_DIR / "stanford_dogs" / "Images"
        places_desc = "Scene photos spanning indoor, outdoor natural, and outdoor man-made environments."
        places_folder = DATA_DIR / "places365" / "val_256"
        return [
            DemoDataset(
                id="caltech101_s",
                label="Caltech-101 (S)",
                description=ct101_desc,
                categories=cats101,
                source="caltech101",
                required_folder=ct101_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=86,
                download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech101_m",
                label="Caltech-101 (M)",
                description=ct101_desc,
                categories=cats101,
                source="caltech101",
                required_folder=ct101_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=86,
                download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech101_l",
                label="Caltech-101 (L)",
                description=ct101_desc,
                categories=cats101,
                source="caltech101",
                required_folder=ct101_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=86,
                download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech101_a",
                label="Caltech-101 (A)",
                description=ct101_desc,
                categories=cats101,
                source="caltech101",
                required_folder=ct101_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=86,
                download_size_mb=CALTECH101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="caltech256_a",
                label="Caltech-256 (A)",
                description="Harder object photos with cluttered backgrounds than Caltech-101.",
                categories=cats256,
                source="caltech256",
                required_folder=DATA_DIR / "caltech-256" / "256_ObjectCategories",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=119,
                download_size_mb=CALTECH256_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="oxford_flowers_102_a",
                label="Oxford Flowers 102 (A)",
                description="Close-up flower photography with fine-grained species variation.",
                categories=self._OXFORD_FLOWERS_CATEGORIES,
                source="oxford_flowers_102",
                required_folder=DATA_DIR / "oxford_flowers",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=80,
                download_size_mb=OXFORD_FLOWERS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="food101_s",
                label="Food-101 (S)",
                description=food_desc,
                categories=self._FOOD101_CATEGORIES,
                source="food101",
                required_folder=food_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=1000,
                download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="food101_m",
                label="Food-101 (M)",
                description=food_desc,
                categories=self._FOOD101_CATEGORIES,
                source="food101",
                required_folder=food_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=1000,
                download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="food101_l",
                label="Food-101 (L)",
                description=food_desc,
                categories=self._FOOD101_CATEGORIES,
                source="food101",
                required_folder=food_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=1000,
                download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="food101_a",
                label="Food-101 (A)",
                description=food_desc,
                categories=self._FOOD101_CATEGORIES,
                source="food101",
                required_folder=food_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=1000,
                download_size_mb=FOOD101_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="eurosat_s",
                label="EuroSAT (S)",
                description=euro_desc,
                categories=self._EUROSAT_CATEGORIES,
                source="eurosat",
                required_folder=euro_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=2700,
                download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="eurosat_m",
                label="EuroSAT (M)",
                description=euro_desc,
                categories=self._EUROSAT_CATEGORIES,
                source="eurosat",
                required_folder=euro_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=2700,
                download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="eurosat_l",
                label="EuroSAT (L)",
                description=euro_desc,
                categories=self._EUROSAT_CATEGORIES,
                source="eurosat",
                required_folder=euro_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=2700,
                download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="eurosat_a",
                label="EuroSAT (A)",
                description=euro_desc,
                categories=self._EUROSAT_CATEGORIES,
                source="eurosat",
                required_folder=euro_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=2700,
                download_size_mb=EUROSAT_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="stanford_dogs_s",
                label="Stanford Dogs (S)",
                description=dogs_desc,
                categories=self._STANFORD_DOGS_CATEGORIES,
                source="stanford_dogs",
                required_folder=dogs_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=171,
                download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="stanford_dogs_m",
                label="Stanford Dogs (M)",
                description=dogs_desc,
                categories=self._STANFORD_DOGS_CATEGORIES,
                source="stanford_dogs",
                required_folder=dogs_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=171,
                download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="stanford_dogs_l",
                label="Stanford Dogs (L)",
                description=dogs_desc,
                categories=self._STANFORD_DOGS_CATEGORIES,
                source="stanford_dogs",
                required_folder=dogs_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=171,
                download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="stanford_dogs_a",
                label="Stanford Dogs (A)",
                description=dogs_desc,
                categories=self._STANFORD_DOGS_CATEGORIES,
                source="stanford_dogs",
                required_folder=dogs_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=171,
                download_size_mb=STANFORD_DOGS_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="places365_s",
                label="Places365 (S)",
                description=places_desc,
                categories=self._PLACES365_CATEGORIES,
                source="places365",
                required_folder=places_folder,
                slice_frac_start=0.0,
                slice_frac_end=1 / 7,
                items_per_category=100,
                download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="places365_m",
                label="Places365 (M)",
                description=places_desc,
                categories=self._PLACES365_CATEGORIES,
                source="places365",
                required_folder=places_folder,
                slice_frac_start=1 / 7,
                slice_frac_end=3 / 7,
                items_per_category=100,
                download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="places365_l",
                label="Places365 (L)",
                description=places_desc,
                categories=self._PLACES365_CATEGORIES,
                source="places365",
                required_folder=places_folder,
                slice_frac_start=3 / 7,
                slice_frac_end=None,
                items_per_category=100,
                download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="places365_a",
                label="Places365 (A)",
                description=places_desc,
                categories=self._PLACES365_CATEGORIES,
                source="places365",
                required_folder=places_folder,
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=100,
                download_size_mb=PLACES365_DOWNLOAD_SIZE_MB,
            ),
            DemoDataset(
                id="ucsf_documents_a",
                label="UCSF Documents (A)",
                description="Scanned pages from the UCSF Industry Documents Library.",
                categories=self._UCSF_DOCUMENTS_CATEGORIES,
                source="ucsf_documents",
                required_folder=DATA_DIR / "ucsf_documents",
                slice_frac_start=0.0,
                slice_frac_end=None,
                items_per_category=25,
                download_size_mb=UCSF_IDL_DOWNLOAD_SIZE_MB,
            ),
        ]

    # ------------------------------------------------------------------
    # Demo dataset loading
    # ------------------------------------------------------------------

    def load_demo_source(
        self,
        source,
        categories,
        slice_start,
        slice_end,
        clips,
        on_progress=None,
        embedder=None,
        slice_frac_start=None,
        slice_frac_end=None,
        **kwargs,
    ):
        import hashlib  # noqa: PLC0415
        import io as _io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        if on_progress is None:
            from vtsearch.concurrency.progress import update_progress

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

            from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

            for i, (img_path, category) in enumerate(selected):
                on_progress("embedding", f"Embedding {category}/{img_path.name}", i + 1, total)
                embedding = embedder.embed_media(media_from_path(img_path))
                if embedding is None:
                    continue
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                try:
                    with Image.open(img_path) as img:
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
                selected.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )

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
                selected.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )

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
                selected.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )

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
                selected.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )

            _embed_file_images(selected)
            return None

        elif source == "places365":
            from vtsearch.datasets.downloader import download_places365  # noqa: PLC0415
            from vtsearch.datasets.loader import load_places365_metadata  # noqa: PLC0415

            places_dir = download_places365(on_progress=on_progress)
            metadata = load_places365_metadata(places_dir, self._PLACES365_CATEGORIES)

            by_cat: dict[str, list[tuple[Path, str]]] = {}
            for _fname, meta in sorted(metadata.items()):
                cat = meta["category"]
                if cat in categories:
                    by_cat.setdefault(cat, []).append((meta["path"], cat))

            selected: list[tuple[Path, str]] = []
            for cat in categories:
                selected.extend(
                    demo_slice(
                        by_cat.get(cat, []),
                        slice_start,
                        slice_end,
                        slice_frac_start,
                        slice_frac_end,
                    )
                )

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
                for page_name, pil_image in demo_slice(
                    by_cat_pages.get(cat, []),
                    slice_start,
                    slice_end,
                    slice_frac_start,
                    slice_frac_end,
                ):
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
                    for idx in demo_slice(
                        cat_mask,
                        slice_start,
                        slice_end or len(cat_mask),
                        slice_frac_start,
                        slice_frac_end,
                    ):
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

    def load_media_data(self, file_path: Path, media_bytes: bytes | None = None) -> dict:
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        if media_bytes is None:
            with open(file_path, "rb") as f:
                media_bytes = f.read()
        try:
            with Image.open(io.BytesIO(media_bytes)) as img:
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
