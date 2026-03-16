"""Dataset configurations — built from the demo dataset registry.

``DEMO_DATASETS`` is assembled at import time from the centralised demo
dataset definitions in :mod:`vtsearch.datasets.importers.demo.datasets`.
"""

from vtsearch.datasets.importers.demo.datasets import all_demo_datasets

DEMO_DATASETS: dict = all_demo_datasets()
