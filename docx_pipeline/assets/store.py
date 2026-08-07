"""Content-addressed storage for assets extracted from a DOCX package."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
import mimetypes

from ..logging import get_logger


_MIME_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "image/x-emf": ".emf",
    "image/x-wmf": ".wmf",
}


@dataclass(frozen=True, slots=True)
class StoredAsset:
    """One unique binary asset; multiple source nodes may reference it."""

    asset_id: str
    sha256: str
    original_name: str
    mime_type: str
    size_bytes: int
    path: str | None = None
    source_parts: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class AssetStore:
    """Deduplicate assets by bytes and optionally materialize them on disk."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self._assets: dict[str, StoredAsset] = {}

    @property
    def assets(self) -> list[StoredAsset]:
        return [self._assets[key] for key in sorted(self._assets)]

    def add(
        self,
        data: bytes,
        *,
        original_name: str,
        mime_type: str | None,
        source_part: str,
        metadata: dict[str, object] | None = None,
    ) -> StoredAsset:
        digest = sha256(data).hexdigest()
        asset_id = f"asset-{digest}"
        normalized_mime = mime_type or mimetypes.guess_type(original_name)[0] or "application/octet-stream"

        existing = self._assets.get(asset_id)
        if existing is not None:
            get_logger(component_area="assets").debug("Reused asset {} from part {}", asset_id, source_part)
            sources = tuple(sorted(set(existing.source_parts) | {source_part}))
            if sources != existing.source_parts:
                existing = StoredAsset(
                    asset_id=existing.asset_id,
                    sha256=existing.sha256,
                    original_name=existing.original_name,
                    mime_type=existing.mime_type,
                    size_bytes=existing.size_bytes,
                    path=existing.path,
                    source_parts=sources,
                    metadata=existing.metadata,
                )
                self._assets[asset_id] = existing
            return existing

        path: str | None = None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = _MIME_EXTENSIONS.get(normalized_mime)
            if suffix is None:
                suffix = Path(original_name).suffix.lower() or mimetypes.guess_extension(normalized_mime) or ".bin"
            destination = self.output_dir / f"{asset_id}{suffix}"
            if not destination.exists():
                destination.write_bytes(data)
            # Keep serialized IR independent of the caller's absolute output
            # directory; the CLI/API already supplies that directory context.
            path = destination.name

        asset = StoredAsset(
            asset_id=asset_id,
            sha256=digest,
            original_name=original_name,
            mime_type=normalized_mime,
            size_bytes=len(data),
            path=path,
            source_parts=(source_part,),
            metadata=metadata or {},
        )
        self._assets[asset_id] = asset
        get_logger(component_area="assets").debug("Stored asset {} ({} bytes, {})", asset_id, asset.size_bytes, asset.mime_type)
        return asset
