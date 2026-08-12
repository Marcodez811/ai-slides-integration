"""Independent package-level checks for rendered PowerPoint files.

The renderer's reopen check is useful, but this module deliberately does not
depend on python-pptx.  A generated deck is promotion-worthy only after the
OOXML package itself has passed these inexpensive structural checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from posixpath import normpath
import shutil
import subprocess
import tempfile
from typing import Literal
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile, is_zipfile


_PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True, slots=True)
class VerificationDiagnostic:
    severity: Literal["info", "warning", "error"]
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class PptxVerificationReport:
    """Serializable result of package-level validation."""

    output_path: str
    verified: bool
    slide_count: int | None
    used_soffice: bool
    diagnostics: tuple[VerificationDiagnostic, ...]
    verifier_mode: Literal["office", "ooxml_zip"] = "ooxml_zip"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_pptx(
    path: str | Path,
    *,
    expected_slide_count: int | None = None,
    try_soffice: bool = True,
    reported_path: str | Path | None = None,
) -> PptxVerificationReport:
    """Validate a PPTX ZIP package without requiring a desktop application.

    LibreOffice is attempted when installed, but its absence never weakens the
    mandatory ZIP/XML/relationship/slide-count checks.
    """
    target = Path(path)
    diagnostics: list[VerificationDiagnostic] = []
    slide_count: int | None = None
    if not target.is_file() or target.is_symlink():
        diagnostics.append(_error("PPTX_FILE_MISSING", "PPTX output is not a regular file."))
    elif not is_zipfile(target):
        diagnostics.append(_error("PPTX_NOT_ZIP", "PPTX output is not a valid ZIP package."))
    else:
        try:
            with ZipFile(target) as package:
                slide_count = _verify_ooxml(package, diagnostics)
        except (BadZipFile, ElementTree.ParseError, KeyError, OSError) as error:
            diagnostics.append(_error("PPTX_OOXML_INVALID", f"OOXML package validation failed: {type(error).__name__}."))
    if expected_slide_count is not None and slide_count != expected_slide_count:
        diagnostics.append(_error(
            "PPTX_SLIDE_COUNT_MISMATCH",
            f"Expected {expected_slide_count} slide(s); package contains {slide_count if slide_count is not None else 'none'}.",
        ))

    used_soffice = False
    verifier_mode: Literal["office", "ooxml_zip"] = "ooxml_zip"
    if not any(item.severity == "error" for item in diagnostics) and try_soffice:
        soffice = shutil.which("soffice")
        if soffice is None:
            diagnostics.append(VerificationDiagnostic("info", "SOFFICE_UNAVAILABLE", "LibreOffice is unavailable; OOXML validation was used."))
        else:
            verifier_mode = "office"
            used_soffice = _verify_with_soffice(target, soffice, diagnostics)

    return PptxVerificationReport(
        output_path=str(reported_path or target),
        verified=not any(item.severity == "error" for item in diagnostics),
        slide_count=slide_count,
        used_soffice=used_soffice,
        diagnostics=tuple(diagnostics),
        verifier_mode=verifier_mode,
    )


def _verify_ooxml(package: ZipFile, diagnostics: list[VerificationDiagnostic]) -> int | None:
    names = set(package.namelist())
    required = {"[Content_Types].xml", "_rels/.rels", "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"}
    absent = sorted(required - names)
    if absent:
        diagnostics.append(_error("PPTX_REQUIRED_PART_MISSING", f"Missing required OOXML part(s): {', '.join(absent)}."))
        return None

    corrupt_member = package.testzip()
    if corrupt_member is not None:
        diagnostics.append(_error("PPTX_CRC_INVALID", f"ZIP CRC validation failed for {corrupt_member}."))
        return None

    ElementTree.fromstring(package.read("[Content_Types].xml"))
    ElementTree.fromstring(package.read("_rels/.rels"))
    presentation = ElementTree.fromstring(package.read("ppt/presentation.xml"))
    relationships = ElementTree.fromstring(package.read("ppt/_rels/presentation.xml.rels"))
    relationship_targets = {
        item.attrib.get("Id"): item.attrib.get("Target", "")
        for item in relationships.findall(f"{{{_PACKAGE_REL}}}Relationship")
        if item.attrib.get("Type", "").endswith("/slide")
    }
    slide_ids = presentation.findall(f".//{{{_PML}}}sldId")
    for slide_id in slide_ids:
        relationship_id = slide_id.attrib.get(f"{{{_REL}}}id")
        relationship_target = relationship_targets.get(relationship_id)
        if not relationship_target:
            diagnostics.append(_error("PPTX_SLIDE_RELATIONSHIP_MISSING", "A presentation slide ID has no slide relationship."))
            continue
        part_name = normpath("ppt/" + relationship_target.lstrip("/"))
        if part_name not in names:
            diagnostics.append(_error("PPTX_SLIDE_PART_MISSING", "A slide relationship points to a missing slide XML part."))
            continue
        root = ElementTree.fromstring(package.read(part_name))
        if root.tag != f"{{{_PML}}}sld":
            diagnostics.append(_error("PPTX_SLIDE_XML_INVALID", "A referenced slide part does not contain a presentation slide root."))
    return len(slide_ids)


def _verify_with_soffice(path: Path, soffice: str, diagnostics: list[VerificationDiagnostic]) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="pptx-verify-") as output_dir:
            completed = subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", output_dir, str(path)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            pdf_path = Path(output_dir) / f"{path.stem}.pdf"
            if completed.returncode != 0 or not pdf_path.is_file():
                diagnostics.append(_error("SOFFICE_CONVERSION_FAILED", "LibreOffice could not convert the PPTX."))
                return False
    except (OSError, subprocess.TimeoutExpired):
        diagnostics.append(_error("SOFFICE_CONVERSION_FAILED", "LibreOffice could not complete validation."))
        return False
    diagnostics.append(VerificationDiagnostic("info", "SOFFICE_CONVERSION_PASSED", "LibreOffice successfully converted the PPTX."))
    return True


def _error(code: str, message: str) -> VerificationDiagnostic:
    return VerificationDiagnostic("error", code, message)
