"""Text to talking-head video, using your own face and a clone of your voice.

Built on a deliberately narrow model stack: every component's *weights* -- not
just its code -- permit commercial use, and every one is pulled from its
vendor's own repository. See ``config.TRUSTED_SOURCES`` for the list and
``config.REJECTED_MODELS`` for the popular options that had to be excluded.
"""

__version__ = "0.1.0"
