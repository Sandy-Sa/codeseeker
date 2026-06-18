import pydantic
from qdrant_client import models as qdrm


class FactoryConfig(pydantic.BaseModel):
    """Configures how the Qdrant client connects.

    Two mutually exclusive modes:
      * Embedded/local: set ``local_path`` to a directory (on-disk index) or to
        ``":memory:"`` for an in-process index. No server required.
      * Server: leave ``local_path`` unset and use ``host``/``port``/``grpc_port``.
    """

    local_path: str | None = None
    host: str = "localhost"
    port: int = 6333
    grpc_port: None | int = 6334


class CollectionBody(pydantic.BaseModel):
    """Collection config."""

    vectors_config: dict[str, qdrm.VectorParams]
    sparse_vectors_config: dict[str, qdrm.SparseVectorParams]
    hnsw_config: qdrm.HnswConfigDiff | None = None
    quantization_config: qdrm.QuantizationConfig | None = None
