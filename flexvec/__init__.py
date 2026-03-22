"""flexvec — numpy-backed semantic search for any SQLite database."""
from .vec_ops import (
    VectorCache, register_vec_ops, materialize_vec_ops,
)
from .score import parse_modifiers
from .keyword import materialize_keyword
from .execute import execute
from .embed import get_embed_fn
