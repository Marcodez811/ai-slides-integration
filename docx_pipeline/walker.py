"""Ordered OOXML walker that emits a flat, auditable source-node ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable

from dwml.omml import oMath2Latex
from lxml import etree

from .assets import AssetStore, StoredAsset
from .extractors import extract_chart
from .ids import make_doc_id, make_node_id
from .inventory import canonical_xml_paths
from .logging import get_logger
from .markup_compatibility import MC_ALTERNATE_CONTENT, iter_effective_elements, selected_branch


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "v": "urn:schemas-microsoft-com:vml",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
}

W = f"{{{NS['w']}}}"
R = f"{{{NS['r']}}}"
M = f"{{{NS['m']}}}"


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname


@dataclass(slots=True)
class WalkOutput:
    """Unvalidated records produced by :class:`OrderedXmlWalker`."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    assets: list[StoredAsset] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)


class OrderedXmlWalker:
    """Walk configured DOCX XML parts without semantic section/list grouping."""

    def __init__(self, package: Any, *, asset_store: AssetStore) -> None:
        self.package = package
        self.asset_store = asset_store
        self.document_id = make_doc_id(package.original_bytes)
        self._nodes: list[dict[str, Any]] = []
        self._diagnostics: list[dict[str, Any]] = []
        self._sequence = 0
        self._child_counts: dict[str, int] = {}
        self._processed_textboxes: set[tuple[str, str]] = set()
        self._styles = self._load_paragraph_styles()
        self._numbering = self._load_numbering()
        self._path_maps: dict[int, dict[etree._Element, str]] = {}

    def walk(self, part_names: Iterable[str] | None = None) -> WalkOutput:
        """Walk parts in the supplied order, defaulting to the main document."""
        names = list(part_names or ["word/document.xml"])
        log = get_logger(component_area="walker")
        log.debug("Walking {} configured parts", len(names))
        for part_name in names:
            if not self.package.has_part(part_name):
                self._diagnose(
                    "MISSING_CONFIGURED_PART",
                    "warning",
                    f"Configured XML part is absent: {part_name}",
                    part_name=part_name,
                )
                continue
            self._walk_part(part_name)
        log.debug("Walk produced {} nodes, {} assets, and {} diagnostics", len(self._nodes), len(self.asset_store.assets), len(self._diagnostics))
        return WalkOutput(
            nodes=self._nodes,
            assets=self.asset_store.assets,
            diagnostics=self._diagnostics,
        )

    def _walk_part(self, part_name: str) -> None:
        get_logger(component_area="walker").debug("Walking part {}", part_name)
        try:
            root = etree.fromstring(self.package.read_part(part_name))
        except Exception as exc:
            self._diagnose(
                "INVALID_XML_PART",
                "error",
                f"Could not parse {part_name}: {exc}",
                part_name=part_name,
            )
            return

        tree = root.getroottree()
        self._path_maps[id(tree)] = canonical_xml_paths(root)
        kind = self._part_kind(part_name)
        root_node = self._emit(
            kind,
            root,
            part_name,
            parent_id=None,
            payload={"part_name": part_name},
            tree=tree,
        )

        if kind == "document_body":
            bodies = root.xpath("/w:document/w:body", namespaces=NS)
            if not bodies:
                self._diagnose(
                    "MISSING_DOCUMENT_BODY",
                    "error",
                    "word/document.xml has no w:body",
                    part_name=part_name,
                    node_id=root_node["node_id"],
                )
                return
            for child in bodies[0]:
                self._walk_block(child, part_name, root_node["node_id"], tree)
        else:
            for child in root:
                self._walk_block(child, part_name, root_node["node_id"], tree)

    @staticmethod
    def _part_kind(part_name: str) -> str:
        name = PurePosixPath(part_name).name
        if name == "document.xml":
            return "document_body"
        if name.startswith("header"):
            return "header"
        if name.startswith("footer"):
            return "footer"
        if name == "footnotes.xml":
            return "footnote"
        if name == "endnotes.xml":
            return "endnote"
        if name == "comments.xml":
            return "comment"
        return "part"

    def _walk_block(
        self,
        element: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        tag = _local_name(element)
        if element.tag == MC_ALTERNATE_CONTENT:
            branch = selected_branch(element)
            if branch is None:
                self._emit_unsupported_alternate_content(element, part_name, parent_id, tree)
                return
            for child in branch:
                self._walk_block(child, part_name, parent_id, tree)
        elif tag == "p":
            self._walk_paragraph(element, part_name, parent_id, tree)
        elif tag == "tbl":
            self._walk_table(element, part_name, parent_id, tree)
        elif tag == "sdt":
            self._walk_sdt(element, part_name, parent_id, tree)
        elif tag in {"footnote", "endnote", "comment"}:
            container = self._emit(
                tag,
                element,
                part_name,
                parent_id,
                {"reference_id": element.get(f"{W}id")},
                tree,
            )
            for child in element:
                self._walk_block(child, part_name, container["node_id"], tree)
        elif tag == "sectPr":
            self._emit("section_break", element, part_name, parent_id, {}, tree)
        elif tag in {"bookmarkStart", "bookmarkEnd"}:
            self._emit_bookmark(element, part_name, parent_id, tree)
        elif tag in {"proofErr", "permStart", "permEnd"}:
            return
        else:
            visible = element.xpath(".//w:t", namespaces=NS)
            node = self._emit(
                "unsupported",
                element,
                part_name,
                parent_id,
                {"feature_type": tag, "visible_text_nodes": len(visible)},
                tree,
                status="unsupported",
            )
            self._diagnose(
                "UNSUPPORTED_BLOCK_ELEMENT",
                "warning" if visible else "info",
                f"Unsupported block element: {tag}",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _walk_paragraph(
        self,
        paragraph: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        p_style = paragraph.find(f"./{W}pPr/{W}pStyle")
        num_id = paragraph.find(f"./{W}pPr/{W}numPr/{W}numId")
        ilvl = paragraph.find(f"./{W}pPr/{W}numPr/{W}ilvl")
        outline = paragraph.find(f"./{W}pPr/{W}outlineLvl")
        style_id = p_style.get(f"{W}val") if p_style is not None else None
        style_facts = self._styles.get(style_id or "", {})
        direct_outline = self._as_int(outline.get(f"{W}val")) if outline is not None else None
        effective_num_id = num_id.get(f"{W}val") if num_id is not None else style_facts.get("num_id")
        effective_ilvl = (
            self._as_int(ilvl.get(f"{W}val"))
            if ilvl is not None
            else style_facts.get("list_level")
        )
        numbering_facts = self._numbering.get(str(effective_num_id), {}).get(effective_ilvl or 0, {})
        payload = {
            "style_id": style_id,
            "style_name": style_facts.get("name"),
            "num_id": effective_num_id,
            "list_level": effective_ilvl,
            "outline_level": direct_outline if direct_outline is not None else style_facts.get("outline_level"),
            "numbering": {
                "num_id": effective_num_id,
                "ilvl": effective_ilvl or 0,
                **numbering_facts,
            }
            if effective_num_id not in {None, "0", 0}
            else None,
        }
        p_node = self._emit("paragraph", paragraph, part_name, parent_id, payload, tree)
        p_id = p_node["node_id"]

        for child in paragraph:
            tag = _local_name(child)
            if tag == "pPr":
                sect_pr = child.find(f"./{W}sectPr")
                if sect_pr is not None:
                    self._emit("section_break", sect_pr, part_name, p_id, {}, tree)
                continue
            self._walk_inline(child, part_name, p_id, tree, formatting={})

    def _walk_inline(
        self,
        element: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
        *,
        formatting: dict[str, Any],
    ) -> None:
        tag = _local_name(element)
        if tag == "r":
            run_format = self._run_formatting(element)
            for child in element:
                if _local_name(child) != "rPr":
                    self._walk_inline(
                        child,
                        part_name,
                        parent_id,
                        tree,
                        formatting=run_format,
                    )
            return

        if tag in {"t", "delText"}:
            self._emit(
                "text_run",
                element,
                part_name,
                parent_id,
                {
                    "text": element.text or "",
                    "formatting": formatting,
                    "deleted": tag == "delText",
                },
                tree,
            )
            return

        if tag == "hyperlink":
            rel_id = element.get(f"{R}id")
            target_part, external_url = self.package.resolve_relationship(part_name, rel_id) if rel_id else (None, None)
            link_node = self._emit(
                "hyperlink",
                element,
                part_name,
                parent_id,
                {
                    "relationship_id": rel_id,
                    "url": external_url,
                    "target_part": target_part,
                    "anchor": element.get(f"{W}anchor"),
                },
                tree,
            )
            for child in element:
                self._walk_inline(child, part_name, link_node["node_id"], tree, formatting={})
            return

        if tag in {"oMath", "oMathPara"}:
            self._emit_equation(element, part_name, parent_id, tree)
            return

        if tag == "drawing":
            self._walk_drawing(element, part_name, parent_id, tree)
            return

        if tag == "pict":
            self._walk_vml_picture(element, part_name, parent_id, tree)
            return

        if tag == "txbxContent":
            self._walk_textbox_content(element, part_name, parent_id, tree, {"placement": "drawingml"})
            return

        if tag in {"br", "cr"}:
            break_type = element.get(f"{W}type") or "line"
            kind = "page_break" if break_type == "page" else "line_break"
            self._emit(kind, element, part_name, parent_id, {"break_type": break_type}, tree)
            return

        if tag == "tab":
            self._emit("tab", element, part_name, parent_id, {}, tree)
            return

        if tag in {"instrText", "fldChar", "fldSimple"}:
            field_node = self._emit(
                "field",
                element,
                part_name,
                parent_id,
                {
                    "field_type": tag,
                    "instruction": (element.text or "").strip() or element.get(f"{W}instr"),
                    "marker": element.get(f"{W}fldCharType"),
                },
                tree,
            )
            if tag == "fldSimple":
                for child in element:
                    self._walk_inline(child, part_name, field_node["node_id"], tree, formatting=formatting)
            return

        if tag == "sdt":
            self._walk_sdt(element, part_name, parent_id, tree, inline=True)
            return

        if tag in {"bookmarkStart", "bookmarkEnd"}:
            self._emit_bookmark(element, part_name, parent_id, tree)
            return

        if tag in {"lastRenderedPageBreak"}:
            self._emit("page_break", element, part_name, parent_id, {"rendered": True}, tree)
            return

        if element.tag == MC_ALTERNATE_CONTENT:
            branch = selected_branch(element)
            if branch is None:
                self._emit_unsupported_alternate_content(element, part_name, parent_id, tree)
                return
            for child in branch:
                self._walk_inline(child, part_name, parent_id, tree, formatting=formatting)
            return

        if tag in {
            "smartTag",
            "customXml",
            "ins",
            "del",
            "moveFrom",
            "moveTo",
            "AlternateContent",
            "Choice",
            "Fallback",
        }:
            for child in element:
                self._walk_inline(child, part_name, parent_id, tree, formatting=formatting)
            return

        if tag in {"commentReference", "footnoteReference", "endnoteReference"}:
            self._emit(
                "field",
                element,
                part_name,
                parent_id,
                {"field_type": tag, "reference_id": element.get(f"{W}id")},
                tree,
            )
            return

        # Property and layout-only elements are represented on their owner.
        if tag.endswith("Pr") or tag in {"noBreakHyphen", "softHyphen", "sym"}:
            return

        if len(element):
            for child in element:
                self._walk_inline(child, part_name, parent_id, tree, formatting=formatting)
            return

        if (element.text or "").strip():
            node = self._emit(
                "unsupported",
                element,
                part_name,
                parent_id,
                {"feature_type": tag, "text": element.text},
                tree,
                status="unsupported",
            )
            self._diagnose(
                "UNSUPPORTED_INLINE_ELEMENT",
                "warning",
                f"Unsupported visible inline element: {tag}",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _walk_sdt(
        self,
        element: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
        *,
        inline: bool = False,
    ) -> None:
        alias = element.find(f"./{W}sdtPr/{W}alias")
        tag_node = element.find(f"./{W}sdtPr/{W}tag")
        gallery = element.find(f"./{W}sdtPr/{W}docPartObj/{W}docPartGallery")
        gallery_value = gallery.get(f"{W}val") if gallery is not None else None
        sdt_node = self._emit(
            "content_control",
            element,
            part_name,
            parent_id,
            {
                "alias": alias.get(f"{W}val") if alias is not None else None,
                "tag": tag_node.get(f"{W}val") if tag_node is not None else None,
                "is_toc": bool(gallery_value and "table of contents" in gallery_value.lower()),
                "inline": inline,
            },
            tree,
        )
        contents = element.findall(f"./{W}sdtContent")
        for content in contents:
            for child in content:
                if inline:
                    self._walk_inline(child, part_name, sdt_node["node_id"], tree, formatting={})
                else:
                    self._walk_block(child, part_name, sdt_node["node_id"], tree)

    def _walk_table(
        self,
        table: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        grid_cols = table.findall(f"./{W}tblGrid/{W}gridCol")
        tbl_node = self._emit(
            "table",
            table,
            part_name,
            parent_id,
            {"grid_column_count": len(grid_cols)},
            tree,
        )
        for row_index, row in enumerate(child for child in table if _local_name(child) == "tr"):
            row_node = self._emit(
                "table_row",
                row,
                part_name,
                tbl_node["node_id"],
                {"row_index": row_index, "is_header": row.find(f"./{W}trPr/{W}tblHeader") is not None},
                tree,
            )
            col_index = 0
            for cell in (child for child in row if _local_name(child) == "tc"):
                grid_span_node = cell.find(f"./{W}tcPr/{W}gridSpan")
                grid_span = self._as_int(grid_span_node.get(f"{W}val")) if grid_span_node is not None else 1
                vmerge = cell.find(f"./{W}tcPr/{W}vMerge")
                cell_node = self._emit(
                    "table_cell",
                    cell,
                    part_name,
                    row_node["node_id"],
                    {
                        "row_index": row_index,
                        "column_index": col_index,
                        "column_span": grid_span or 1,
                        "vertical_merge": (vmerge.get(f"{W}val") or "continue") if vmerge is not None else None,
                    },
                    tree,
                )
                for child in cell:
                    if _local_name(child) != "tcPr":
                        self._walk_block(child, part_name, cell_node["node_id"], tree)
                col_index += grid_span or 1

    def _walk_drawing(
        self,
        drawing: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        emitted = 0
        effective_elements = list(iter_effective_elements(drawing))
        placement = "anchor" if any(item.tag == f"{{{NS['wp']}}}anchor" for item in effective_elements) else "inline"
        extents = [item for item in effective_elements if item.tag == f"{{{NS['wp']}}}extent"]
        doc_props = [item for item in effective_elements if item.tag == f"{{{NS['wp']}}}docPr"]
        common_metadata: dict[str, Any] = {"placement": placement}
        if extents:
            common_metadata.update(
                width_emu=self._as_int(extents[0].get("cx")),
                height_emu=self._as_int(extents[0].get("cy")),
            )
        if doc_props:
            common_metadata.update(
                name=doc_props[0].get("name"),
                title=doc_props[0].get("title"),
                alt_text=doc_props[0].get("descr"),
            )

        for blip in (item for item in effective_elements if item.tag == f"{{{NS['a']}}}blip"):
            rel_id = blip.get(f"{R}embed") or blip.get(f"{R}link")
            self._emit_image_occurrence(
                blip,
                part_name,
                parent_id,
                tree,
                rel_id=rel_id,
                linked=blip.get(f"{R}link") is not None,
                metadata=common_metadata,
            )
            emitted += 1

        for chart in (item for item in effective_elements if item.tag == f"{{{NS['c']}}}chart"):
            rel_id = chart.get(f"{R}id")
            target_part, external_url = self.package.resolve_relationship(part_name, rel_id) if rel_id else (None, None)
            chart_metadata = extract_chart(self.package, part_name, rel_id) if rel_id else None
            payload = {
                "relationship_id": rel_id,
                "target_part": target_part,
                "external_url": external_url,
                **common_metadata,
            }
            if chart_metadata is not None:
                payload.update(
                    {
                        key: value
                        for key, value in chart_metadata.to_dict().items()
                        if key != "diagnostics"
                    }
                )
            chart_node = self._emit(
                "chart",
                chart,
                part_name,
                parent_id,
                payload,
                tree,
            )
            if chart_metadata is not None:
                for diagnostic in chart_metadata.diagnostics:
                    self._diagnose(
                        diagnostic.code,
                        diagnostic.severity,
                        diagnostic.message,
                        part_name=diagnostic.part_name or part_name,
                        node_id=chart_node["node_id"],
                        xml_path=chart_node["source"]["xml_path"],
                    )
            emitted += 1

        for textbox in (item for item in effective_elements if item.tag == f"{W}txbxContent"):
            emitted += self._walk_textbox_content(textbox, part_name, parent_id, tree, common_metadata)

        if emitted == 0:
            node = self._emit(
                "unsupported",
                drawing,
                part_name,
                parent_id,
                {"feature_type": "drawing", **common_metadata},
                tree,
                status="unsupported",
            )
            self._diagnose(
                "UNSUPPORTED_DRAWING",
                "warning",
                "Drawing contained no supported image, chart, or text box",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _walk_vml_picture(
        self,
        pict: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        emitted = 0
        for element in pict.iter():
            if element.tag == f"{{{NS['v']}}}imagedata":
                self._emit_image_occurrence(
                    element,
                    part_name,
                    parent_id,
                    tree,
                    rel_id=element.get(f"{R}id"),
                    linked=False,
                    metadata={"placement": "vml"},
                )
                emitted += 1
            elif element.tag == f"{W}txbxContent":
                emitted += self._walk_textbox_content(
                    element, part_name, parent_id, tree, {"placement": "vml"}
                )
        if emitted == 0:
            node = self._emit(
                "unsupported",
                pict,
                part_name,
                parent_id,
                {"feature_type": "vml_picture"},
                tree,
                status="unsupported",
            )
            self._diagnose(
                "UNSUPPORTED_VML_CONTENT",
                "warning",
                "VML content contains no supported image or textbox",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _walk_textbox_content(
        self,
        textbox: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
        metadata: dict[str, Any],
    ) -> int:
        """Emit a textbox once and walk its block children in document order."""

        key = (part_name, self._xml_path(tree, textbox))
        if key in self._processed_textboxes:
            return 0
        self._processed_textboxes.add(key)
        box_node = self._emit("text_box", textbox, part_name, parent_id, metadata, tree)
        for child in textbox:
            self._walk_block(child, part_name, box_node["node_id"], tree)
        return 1

    def _emit_unsupported_alternate_content(
        self,
        alternate_content: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        """Retain an MC container when no branch can be represented."""

        node = self._emit(
            "unsupported",
            alternate_content,
            part_name,
            parent_id,
            {"feature_type": "markup_compatibility_alternate_content"},
            tree,
            status="unsupported",
        )
        self._diagnose(
            "UNSUPPORTED_MARKUP_COMPATIBILITY_BRANCH",
            "warning",
            "Markup Compatibility content has no supported Choice or Fallback branch",
            part_name=part_name,
            node_id=node["node_id"],
            xml_path=node["source"]["xml_path"],
        )

    def _emit_image_occurrence(
        self,
        element: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
        *,
        rel_id: str | None,
        linked: bool,
        metadata: dict[str, Any],
    ) -> None:
        target_part, external_url = self.package.resolve_relationship(part_name, rel_id) if rel_id else (None, None)
        get_logger(component_area="walker").debug("Image relationship decision: relation_present={}, internal_target={}, external={}", bool(rel_id), bool(target_part), bool(external_url))
        asset_id: str | None = None
        status = "extracted"
        if target_part and self.package.has_part(target_part):
            data = self.package.read_part(target_part)
            asset = self.asset_store.add(
                data,
                original_name=PurePosixPath(target_part).name,
                mime_type=self.package.content_type_for(target_part),
                source_part=target_part,
                metadata={"relationship_source_part": part_name},
            )
            asset_id = asset.asset_id
        else:
            status = "unsupported" if linked or external_url else "failed"

        node = self._emit(
            "image",
            element,
            part_name,
            parent_id,
            {
                "relationship_id": rel_id,
                "target_part": target_part,
                "external_url": external_url,
                "linked": linked,
                "asset_id": asset_id,
                **metadata,
            },
            tree,
            status=status,
            relationship_id=rel_id,
        )
        if asset_id is None:
            get_logger(component_area="walker").warning("Image relationship could not be resolved (linked={})", linked)
            self._diagnose(
                "UNRESOLVED_IMAGE_RELATIONSHIP",
                "warning" if linked else "error",
                f"Could not resolve image relationship {rel_id!r}",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _emit_equation(
        self,
        equation: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        latex: str | None = None
        error: str | None = None
        candidates = [equation]
        if _local_name(equation) == "oMathPara":
            candidates = equation.xpath("./m:oMath", namespaces=NS)
        values: list[str] = []
        for candidate in candidates:
            try:
                value = str(oMath2Latex(candidate)).strip()
                if value:
                    values.append(value)
            except Exception as exc:
                error = str(exc)
        if values:
            latex = " ".join(values)
        node = self._emit(
            "equation",
            equation,
            part_name,
            parent_id,
            {
                "latex": latex,
                "display": _local_name(equation) == "oMathPara",
                "raw_omml": etree.tostring(equation, encoding="unicode"),
            },
            tree,
            status="extracted" if latex else "failed",
        )
        if latex is None:
            self._diagnose(
                "EQUATION_CONVERSION_FAILED",
                "error",
                error or "OMML equation produced no LaTeX",
                part_name=part_name,
                node_id=node["node_id"],
                xml_path=node["source"]["xml_path"],
            )

    def _emit_bookmark(
        self,
        element: etree._Element,
        part_name: str,
        parent_id: str,
        tree: etree._ElementTree,
    ) -> None:
        self._emit(
            "bookmark",
            element,
            part_name,
            parent_id,
            {
                "marker": _local_name(element),
                "bookmark_id": element.get(f"{W}id"),
                "name": element.get(f"{W}name"),
            },
            tree,
        )

    @staticmethod
    def _run_formatting(run: etree._Element) -> dict[str, Any]:
        props = run.find(f"./{W}rPr")
        if props is None:
            return {}
        vertical = props.find(f"./{W}vertAlign")
        color = props.find(f"./{W}color")
        size = props.find(f"./{W}sz")
        lang = props.find(f"./{W}lang")
        return {
            "bold": props.find(f"./{W}b") is not None,
            "italic": props.find(f"./{W}i") is not None,
            "underline": props.find(f"./{W}u") is not None,
            "strikethrough": props.find(f"./{W}strike") is not None,
            "vertical_align": vertical.get(f"{W}val") if vertical is not None else None,
            "color": color.get(f"{W}val") if color is not None else None,
            "size_half_points": OrderedXmlWalker._as_int(size.get(f"{W}val")) if size is not None else None,
            "language": lang.get(f"{W}val") if lang is not None else None,
        }

    def _load_paragraph_styles(self) -> dict[str, dict[str, Any]]:
        """Resolve paragraph style names and inherited outline levels."""
        if not self.package.has_part("word/styles.xml"):
            return {}
        try:
            root = etree.fromstring(self.package.read_part("word/styles.xml"))
        except Exception:
            return {}
        raw: dict[str, dict[str, Any]] = {}
        for style in root.xpath("//w:style[@w:type='paragraph']", namespaces=NS):
            style_id = style.get(f"{W}styleId")
            if not style_id:
                continue
            name = style.find(f"./{W}name")
            based_on = style.find(f"./{W}basedOn")
            outline = style.find(f"./{W}pPr/{W}outlineLvl")
            raw[style_id] = {
                "name": name.get(f"{W}val") if name is not None else None,
                "based_on": based_on.get(f"{W}val") if based_on is not None else None,
                "outline_level": self._as_int(outline.get(f"{W}val")) if outline is not None else None,
                "num_id": self._style_property(style, "numId"),
                "list_level": self._as_int(self._style_property(style, "ilvl")),
            }

        resolved: dict[str, dict[str, Any]] = {}

        def resolve(style_id: str, trail: set[str]) -> dict[str, Any]:
            if style_id in resolved:
                return resolved[style_id]
            facts = raw.get(style_id, {})
            if style_id in trail:
                return facts
            inherited: dict[str, Any] = {}
            parent = facts.get("based_on")
            if isinstance(parent, str) and parent:
                inherited = resolve(parent, trail | {style_id})
            result = {
                "name": facts.get("name") or inherited.get("name"),
                "outline_level": (
                    facts.get("outline_level")
                    if facts.get("outline_level") is not None
                    else inherited.get("outline_level")
                ),
                "num_id": facts.get("num_id") if facts.get("num_id") is not None else inherited.get("num_id"),
                "list_level": (
                    facts.get("list_level")
                    if facts.get("list_level") is not None
                    else inherited.get("list_level")
                ),
            }
            resolved[style_id] = result
            return result

        for style_id in raw:
            resolve(style_id, set())
        return resolved

    @staticmethod
    def _style_property(style: etree._Element, property_name: str) -> str | None:
        element = style.find(f"./{W}pPr/{W}numPr/{W}{property_name}")
        return element.get(f"{W}val") if element is not None else None

    def _load_numbering(self) -> dict[str, dict[int, dict[str, Any]]]:
        """Resolve numId/ilvl facts from numbering.xml without grouping lists."""
        if not self.package.has_part("word/numbering.xml"):
            return {}
        try:
            root = etree.fromstring(self.package.read_part("word/numbering.xml"))
        except Exception:
            return {}
        abstracts: dict[str, dict[int, dict[str, Any]]] = {}
        for abstract in root.findall(f"./{W}abstractNum"):
            abstract_id = abstract.get(f"{W}abstractNumId")
            if abstract_id is None:
                continue
            levels: dict[int, dict[str, Any]] = {}
            for level in abstract.findall(f"./{W}lvl"):
                ilvl_value = self._as_int(level.get(f"{W}ilvl"))
                if ilvl_value is None:
                    continue
                num_fmt = level.find(f"./{W}numFmt")
                lvl_text = level.find(f"./{W}lvlText")
                start = level.find(f"./{W}start")
                fmt = num_fmt.get(f"{W}val") if num_fmt is not None else None
                levels[ilvl_value] = {
                    "num_fmt": fmt,
                    "ordered": fmt not in {None, "bullet", "none"},
                    "marker_template": lvl_text.get(f"{W}val") if lvl_text is not None else None,
                    "start": self._as_int(start.get(f"{W}val")) if start is not None else 1,
                }
            abstracts[abstract_id] = levels

        resolved: dict[str, dict[int, dict[str, Any]]] = {}
        for numbering in root.findall(f"./{W}num"):
            num_id = numbering.get(f"{W}numId")
            abstract_ref = numbering.find(f"./{W}abstractNumId")
            abstract_id = abstract_ref.get(f"{W}val") if abstract_ref is not None else None
            if num_id is None or abstract_id is None:
                continue
            levels = {level: dict(facts) for level, facts in abstracts.get(abstract_id, {}).items()}
            for override in numbering.findall(f"./{W}lvlOverride"):
                override_level = self._as_int(override.get(f"{W}ilvl"))
                if override_level is None:
                    continue
                start_override = override.find(f"./{W}startOverride")
                if start_override is not None:
                    levels.setdefault(override_level, {})["start"] = self._as_int(
                        start_override.get(f"{W}val")
                    )
            resolved[num_id] = levels
        return resolved

    def _emit(
        self,
        kind: str,
        element: etree._Element,
        part_name: str,
        parent_id: str | None,
        payload: dict[str, Any],
        tree: etree._ElementTree,
        *,
        status: str = "extracted",
        relationship_id: str | None = None,
    ) -> dict[str, Any]:
        xml_path = self._xml_path(tree, element)
        node_id = make_node_id(self.document_id, part_name, xml_path, kind)
        sibling_index = self._child_counts.get(parent_id or "<root>", 0)
        self._child_counts[parent_id or "<root>"] = sibling_index + 1
        node = {
            "node_id": node_id,
            "sequence": self._sequence,
            "parent_id": parent_id,
            "sibling_index": sibling_index,
            "kind": kind,
            "source": {
                "part_name": part_name,
                "xml_path": xml_path,
                "relationship_id": relationship_id,
            },
            "payload": payload,
            "status": status,
        }
        self._sequence += 1
        self._nodes.append(node)
        return node

    def _xml_path(self, tree: etree._ElementTree, element: etree._Element) -> str:
        paths = self._path_maps.get(id(tree))
        if paths is None:
            paths = canonical_xml_paths(tree.getroot())
            self._path_maps[id(tree)] = paths
        return paths[element]

    def _diagnose(
        self,
        code: str,
        severity: str,
        message: str,
        *,
        part_name: str | None = None,
        node_id: str | None = None,
        xml_path: str | None = None,
    ) -> None:
        log = get_logger(component_area="walker")
        if severity == "error":
            log.error("Walker diagnostic emitted: {}", code)
        elif severity == "warning":
            log.warning("Walker diagnostic emitted: {}", code)
        else:
            log.debug("Walker diagnostic emitted: {}", code)
        self._diagnostics.append(
            {
                "code": code,
                "severity": severity,
                "message": message,
                "source_node_id": node_id,
                "source": {
                    "part_name": part_name,
                    "xml_path": xml_path,
                }
                if part_name or xml_path
                else None,
            }
        )

    @staticmethod
    def _as_int(value: str | None) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
