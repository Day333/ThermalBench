"""Vendored scOT (the Therm-FM backbone), copied verbatim from Therm-FM/scOT.

Modules inside the package import each other as `from scOT.x import y`. Rather than
rewrite every file, model/scOT is placed on PYTHONPATH so it is importable under its
original name. This keeps the source identical to upstream, which makes future syncs
straightforward.
"""
import sys as _sys

_sys.modules.setdefault('scOT', _sys.modules[__name__])
