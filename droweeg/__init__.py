from __future__ import annotations

import eegda as _eegda
from eegda import *  # noqa: F401,F403
from eegda.results import DrowEEGResults

__all__ = [*_eegda.__all__, "DrowEEGResults"]
__path__ = _eegda.__path__
