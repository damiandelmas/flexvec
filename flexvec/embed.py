"""Default embedding function — wraps the bundled ONNX model."""


def get_embed_fn(prefix='search_query: '):
    """Return an embedding function using the bundled Nomic ONNX model.

    Requires: pip install flexvec[embed]

    Usage:
        from flexvec import get_embed_fn
        embed_fn = get_embed_fn()
    """
    try:
        from .onnx.embed import get_model
        from .onnx.fetch import model_ready, download_model
    except ImportError:
        raise ImportError(
            "flexvec[embed] is required for get_embed_fn(). "
            "Install with: pip install flexvec[embed]"
        )

    if not model_ready():
        print("flexvec: downloading embedding model (~87MB)...")
        download_model()

    model = get_model()

    def embed(text):
        return model.encode([text], prefix=prefix)

    return embed
