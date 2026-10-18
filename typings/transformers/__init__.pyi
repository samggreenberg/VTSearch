# pyright reads this stub in place of the installed `transformers` (#5077).
#
# `transformers/__init__.py` does `from .models import *` under TYPE_CHECKING,
# and `models/__init__.py` wildcard-imports each of its ~530 model packages, so
# the first checked file that touched a `transformers` name made pyright parse
# and bind ~2,600 library files, once per pyright thread. A full two-thread
# check spent about half its CPU there. Resolving `transformers` here means the
# real `__init__.py` is never bound.
#
# Every name is `Any`. Our code already holds transformers models and
# processors in `Any` fields and loads them through
# `load_pretrained_local_first(...) -> Any`, so the inferred types never
# reached it. Re-exporting the real classes from their own modules does not
# work: those modules reach the package root through relative imports
# (`from ... import is_torch_available`), which bind the real `__init__.py`
# and the whole cascade with it.
#
# Submodules are deliberately not stubbed. pyright resolves
# `transformers.utils.logging` and `transformers.modeling_utils` from the
# installed package, with their real types, without binding the root.
#
# Adding a name: a new `from transformers import X` fails pyright with
# '"X" is unknown import symbol' until X is declared below. There is no
# module `__getattr__`, so a misspelled name stays an error.
# `tests_lib/meta/test_transformers_stub.py` checks that every name here exists
# in the installed transformers, the check pyright used to make against the
# real package.

from typing import Any

__version__: str

ASTFeatureExtractor: Any
ASTModel: Any
AutoConfig: Any
AutoImageProcessor: Any
AutoModel: Any
AutoProcessor: Any
AutoTokenizer: Any
BertModel: Any
CLIPModel: Any
CLIPProcessor: Any
ClapModel: Any
ClapProcessor: Any
Qwen2_5_VLForConditionalGeneration: Any
SiglipModel: Any
SiglipProcessor: Any
SiglipTokenizer: Any
Wav2Vec2FeatureExtractor: Any
WhisperFeatureExtractor: Any
WhisperForConditionalGeneration: Any
WhisperModel: Any
WhisperTokenizer: Any
XCLIPModel: Any
XCLIPProcessor: Any
