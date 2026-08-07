"""Tests for safe native-chart package traversal."""

from __future__ import annotations

from io import BytesIO
import zipfile

from docx_pipeline.extractors.chart import extract_chart
from docx_pipeline.package import DocxPackage


def _package_bytes() -> bytes:
    members = {
        "[Content_Types].xml": b'''<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/charts/chart1.xml" ContentType="application/vnd.openxmlformats-officedocument.drawingml.chart+xml"/><Override PartName="/word/embeddings/data.xlsx" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"/></Types>''',
        "word/document.xml": b"<w:document xmlns:w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'/>",
        "word/_rels/document.xml.rels": b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdChart" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart" Target="charts/chart1.xml"/></Relationships>''',
        "word/charts/chart1.xml": b'''<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><c:chart><c:title><c:tx><c:rich><c:p><c:r><c:t>Quarterly results</c:t></c:r></c:p></c:rich></c:tx></c:title><c:plotArea><c:barChart><c:ser><c:idx val="0"/><c:order val="0"/><c:tx><c:v>Revenue</c:v></c:tx><c:cat><c:strRef><c:f>Sheet1!$A$2:$A$4</c:f></c:strRef></c:cat><c:val><c:numRef><c:f>Sheet1!$B$2:$B$4</c:f></c:numRef></c:val></c:ser></c:barChart></c:plotArea></c:chart><c:externalData r:id="rIdWorkbook"/></c:chartSpace>''',
        "word/charts/_rels/chart1.xml.rels": b'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdWorkbook" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/package" Target="../embeddings/data.xlsx"/></Relationships>''',
        "word/embeddings/data.xlsx": b"not-opened-by-extractor",
    }
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return output.getvalue()


def test_extract_chart_reads_metadata_and_never_opens_workbook() -> None:
    result = extract_chart(DocxPackage.from_bytes(_package_bytes()), "word/document.xml", "rIdChart")

    assert result.chart_part_name == "word/charts/chart1.xml"
    assert len(result.raw_xml_sha256 or "") == 64
    assert result.chart_types == ("barChart",)
    assert result.title == "Quarterly results"
    assert result.series[0].to_dict() == {
        "index": 0,
        "order": 0,
        "name": "Revenue",
        "category_formula": "Sheet1!$A$2:$A$4",
        "value_formula": "Sheet1!$B$2:$B$4",
    }
    assert result.embedded_workbook_parts == ("word/embeddings/data.xlsx",)
    assert not result.diagnostics


def test_extract_chart_reports_missing_relation() -> None:
    result = extract_chart(DocxPackage.from_bytes(_package_bytes()), "word/document.xml", "rIdMissing")

    assert result.chart_part_name is None
    assert result.diagnostics[0].code == "MISSING_CHART_RELATIONSHIP"
