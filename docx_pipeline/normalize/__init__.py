"""Pure, replaceable semantic views derived from Source IR."""

from .blocks import build_normalized_views
from .captions import link_captions
from .lists import build_list_views
from .sections import detect_sections

__all__ = ["build_list_views", "build_normalized_views", "detect_sections", "link_captions"]
