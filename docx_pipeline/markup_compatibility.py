"""Minimal OOXML Markup Compatibility branch selection."""

from __future__ import annotations

from collections.abc import Iterator

from lxml import etree

MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
MC_ALTERNATE_CONTENT = f"{{{MC_NS}}}AlternateContent"
MC_CHOICE = f"{{{MC_NS}}}Choice"
MC_FALLBACK = f"{{{MC_NS}}}Fallback"

# The walker can represent WordprocessingShape textboxes through their
# w:txbxContent descendants. Additional namespaces should be added only with
# explicit extraction support for their content.
WORDPROCESSING_SHAPE_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
SUPPORTED_NAMESPACE_URIS = frozenset({WORDPROCESSING_SHAPE_NS})


def selected_branch(alternate_content: etree._Element) -> etree._Element | None:
    """Choose the first supported ``mc:Choice``, otherwise ``mc:Fallback``."""

    fallback: etree._Element | None = None
    for child in alternate_content:
        if child.tag == MC_CHOICE:
            required_prefixes = (child.get("Requires") or "").split()
            required_uris = {child.nsmap.get(prefix) for prefix in required_prefixes}
            if required_prefixes and None not in required_uris and required_uris <= SUPPORTED_NAMESPACE_URIS:
                return child
        elif child.tag == MC_FALLBACK:
            fallback = child
    return fallback


def iter_effective_elements(root: etree._Element) -> Iterator[etree._Element]:
    """Yield visible/effective elements while excluding unselected MC branches."""

    def visit(element: etree._Element) -> Iterator[etree._Element]:
        if element.tag == MC_ALTERNATE_CONTENT:
            branch = selected_branch(element)
            if branch is not None:
                for child in branch:
                    yield from visit(child)
            return
        yield element
        for child in element:
            if isinstance(child.tag, str):
                yield from visit(child)

    yield from visit(root)
