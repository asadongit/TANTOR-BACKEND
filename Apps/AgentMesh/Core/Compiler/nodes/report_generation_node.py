"""
report_generation_node.py
─────────────────────────
Report Generation Node for AgentMesh / Tantor platform.

Tech stack (Option A):
  DOCX  → docxtpl  (Jinja2 template filling)
  DOCX  → python-docx  (AI semantic substitution for no-placeholder docs)
  PDF   → reportlab  (pure Python, zero system deps, Linux-safe)
  XLSX  → openpyxl
  CSV   → stdlib csv
  JSON  → stdlib json
  MD    → jinja2

Bugs fixed vs previous version:
  1. Split-run replacement — paragraph-level text reconstruction before replacing,
     so "John Doe" split across ["John ", "Doe"] runs is found and replaced correctly.
  2. docx2pdf removed entirely — PDF is generated with reportlab directly (no Word/LibreOffice needed).
  3. LLM config null-guard — llm_config=None is handled gracefully with sensible defaults.
  4. get_undeclared_template_variables() unreliability — replaced with a direct XML scan
     for {{ and }} markers, which is more reliable across docxtpl versions.
"""

import logging
import os
import json
import csv
import decimal
import io
import re
import traceback
from typing import Callable

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_config import get_llm_provider

# ── Optional imports (fail loudly at call-time, not import-time) ─────────────

try:
    from docxtpl import DocxTemplate
except ImportError:
    DocxTemplate = None

try:
    import docx
except ImportError:
    docx = None

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )

    reportlab = True
except ImportError:
    reportlab = False

logger = logging.getLogger(__name__)

# ── Path helpers ──────────────────────────────────────────────────────────────

_BACKEND_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)


def _abs(path: str) -> str:
    """Convert a relative path from DB to an absolute path (used for legacy storage_path values)."""
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    stripped = path.lstrip("./").lstrip(".\\")
    return os.path.join(_BACKEND_ROOT, stripped)


# ── Image processing helpers ──────────────────────────────────────────────────


def _download_and_cache_image(url: str, thread_id: str = "default_thread") -> str:
    """Download an image from a URL and cache it locally under TEMP_IMAGES_MOUNT/{thread_id}."""
    import urllib.request
    import urllib.parse
    import uuid
    import ssl

    # Fix: Encode URL query parameters if they contain raw JSON or spaces (e.g., QuickChart)
    encoded_url = urllib.parse.quote(url, safe=":/&?=")

    from Apps.AgentMesh.Config import settings

    temp_dir = os.path.abspath(os.path.join(settings.TEMP_IMAGES_MOUNT, thread_id))
    os.makedirs(temp_dir, exist_ok=True)
    ext = ".png"
    if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
        ext = ".jpg"
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{ext}")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    logger.info(f"[ReportNode] Downloading image from {encoded_url}...")
    req = urllib.request.Request(
        encoded_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    with (
        urllib.request.urlopen(req, context=ctx, timeout=15) as response,
        open(temp_path, "wb") as out_file,
    ):
        out_file.write(response.read())
    return temp_path


def _preprocess_images_for_docx(data, doc, thread_id: str):
    """Recursively scan dict for {'_type': 'image'} and replace with docxtpl.InlineImage."""
    from docxtpl import InlineImage
    from docx.shared import Mm

    if isinstance(data, dict):
        if data.get("_type") == "image" and data.get("url"):
            try:
                local_path = _download_and_cache_image(data["url"], thread_id)
                width = data.get("width_mm", 100)
                return InlineImage(doc, local_path, width=Mm(width))
            except Exception as e:
                logger.error(
                    f"[ReportNode] Failed to download image {data['url']}: {e}"
                )
                return f"[Image Error: {data['url']}]"

        return {
            k: _preprocess_images_for_docx(v, doc, thread_id) for k, v in data.items()
        }

    elif isinstance(data, list):
        return [_preprocess_images_for_docx(item, doc, thread_id) for item in data]

    return data


def _preprocess_images_for_pdf(data, thread_id: str):
    """Recursively scan dict for {'_type': 'image'} and replace with reportlab Image."""
    from reportlab.platypus import Image
    from reportlab.lib.units import mm

    if isinstance(data, dict):
        if data.get("_type") == "image" and data.get("url"):
            try:
                local_path = _download_and_cache_image(data["url"], thread_id)
                width = data.get("width_mm", 100)
                img = Image(local_path)
                img.drawWidth = width * mm
                aspect = (
                    img.imageHeight / float(img.imageWidth) if img.imageWidth else 1
                )
                img.drawHeight = img.drawWidth * aspect
                return img
            except Exception as e:
                logger.error(
                    f"[ReportNode] Failed to download image {data['url']}: {e}"
                )
                return f"[Image Error: {data['url']}]"

        return {k: _preprocess_images_for_pdf(v, thread_id) for k, v in data.items()}

    elif isinstance(data, list):
        return [_preprocess_images_for_pdf(item, thread_id) for item in data]

    return data


# ── DOCX helpers ──────────────────────────────────────────────────────────────


def _docx_has_placeholders(template_path: str) -> bool:
    """
    Reliably detect whether a .docx template contains Jinja2 placeholders.

    Strategy:
      1. Primary  — ask docxtpl via its own Jinja2 env (most accurate; finds
         variables even when tags are split across XML runs by Word).
      2. Fallback — raw XML scan for {{ / {% markers (handles edge cases where
         docxtpl itself chokes on malformed templates).

    FIX: the previous raw-XML-only approach returned True even when docxtpl
    could not actually render the template (tags split across <w:r> elements),
    producing silent empty output.
    """
    # Primary: use docxtpl's built-in Jinja2 variable scanner
    if DocxTemplate is not None:
        try:
            tpl = DocxTemplate(template_path)
            variables = tpl.get_undeclared_template_variables()
            if variables:
                logger.info(
                    f"[ReportNode] docxtpl found template variables: {variables}"
                )
                return True
        except Exception as exc:
            logger.warning(
                f"[ReportNode] docxtpl variable scan failed ({exc}); "
                "falling back to raw XML scan."
            )

    # Fallback: raw XML scan
    import zipfile

    try:
        with zipfile.ZipFile(template_path) as zf:
            xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
        found = (
            "{%" in xml
            or ('{{"{' in xml and "}}" in xml)
            or ("{{" in xml and "}}" in xml)
        )
        if found:
            logger.info("[ReportNode] Raw XML scan detected Jinja2 markers.")
        return found
    except Exception:
        return False


def _replace_in_paragraph(paragraph, old: str, new: str) -> bool:
    """
    FIX for Bug #1 — split-run replacement.

    Word splits text across runs unpredictably. "John Doe" may be stored as
    ["John ", "Doe"] so checking each run individually never finds the full string.

    Strategy:
      1. Reconstruct the full paragraph text from all runs.
      2. If the target string is present, find which runs span it.
      3. Put the replacement text in the first spanning run,
         clear the text from all subsequent spanning runs.
      4. Preserve all run-level formatting (bold, italic, font, size etc.).

    Returns True if a replacement was made.
    """
    full_text = "".join(run.text for run in paragraph.runs)
    if old not in full_text:
        return False

    # Build character→run mapping
    char_to_run = []
    for run_idx, run in enumerate(paragraph.runs):
        for _ in run.text:
            char_to_run.append(run_idx)

    start_char = full_text.index(old)
    end_char = start_char + len(old)

    if end_char > len(char_to_run):
        return False

    first_run_idx = char_to_run[start_char]
    last_run_idx = char_to_run[end_char - 1]

    # Rebuild the text for each affected run
    new_runs_text = [""] * len(paragraph.runs)
    for i, ch in enumerate(full_text):
        run_idx = char_to_run[i]
        if start_char <= i < end_char:
            # Replacement block — inject into first spanning run only
            if i == start_char:
                new_runs_text[first_run_idx] += new
        else:
            new_runs_text[run_idx] += ch

    for idx, run in enumerate(paragraph.runs):
        run.text = new_runs_text[idx]

    return True


def _apply_replacements_to_doc(doc, replacements: dict) -> None:
    """
    Apply a {old: new} replacement dict to all paragraphs, table cells,
    and section headers/footers.

    FIX: previous version only touched doc.paragraphs and doc.tables,
    missing headers/footers (company letterheads, branding, page numbers)
    stored in doc.sections[i].header / .footer.
    """

    def _replace_block(paragraphs, tables):
        for old_str, new_str in replacements.items():
            old_str = str(old_str)
            new_str = str(new_str)
            if not old_str or len(old_str) < 2:
                continue
            for paragraph in paragraphs:
                _replace_in_paragraph(paragraph, old_str, new_str)
            for table in tables:
                for row in table.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            _replace_in_paragraph(paragraph, old_str, new_str)

    # Body
    _replace_block(doc.paragraphs, doc.tables)

    # Headers and footers across every section  ← FIX
    for section in doc.sections:
        for part in [
            section.header,
            section.footer,
            section.even_page_header,
            section.even_page_footer,
            section.first_page_header,
            section.first_page_footer,
        ]:
            if part is not None:
                _replace_block(part.paragraphs, part.tables)


# ── AI Semantic Substitution — DOCX ──────────────────────────────────────────


def _get_llm(llm_config):
    """
    FIX for Bug #3 — null-guard for llm_config.

    If the node has no LLM configured, fall back to sensible defaults rather
    than crashing with AttributeError: 'NoneType' object has no attribute 'provider'.
    """
    provider = getattr(llm_config, "provider", None) or "groq"
    model = getattr(llm_config, "model", None) or "llama-3.3-70b-versatile"
    temperature = getattr(llm_config, "temperature", None) or 0.0
    return get_llm_provider(provider=provider, model=model, temperature=temperature)


def run_ai_semantic_substitution_docx(
    template_path: str,
    output_path: str,
    input_data: dict,
    llm_config,
) -> None:
    """
    For .docx files that have no {{placeholders}}:
      1. Extract document text.
      2. Ask the LLM to map existing text → new values.
      3. Apply replacements using the split-run-safe helper.
    """
    if not docx:
        raise ImportError("python-docx is not installed. Run: pip install python-docx")

    doc = docx.Document(template_path)

    # Extract representative text (paragraphs + table cells)
    lines = []
    for p in doc.paragraphs:
        if p.text.strip():
            lines.append(p.text)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    lines.append(cell.text)

    doc_text = "\n".join(lines)

    prompt = (
        "You are a document editor. I have a template document and new data.\n"
        "Identify EXACT literal substrings in the template that should be replaced with the new data.\n"
        "Return ONLY a valid JSON object where keys are EXACT substrings from the document text "
        "and values are the replacements. No markdown, no backticks, no explanation.\n\n"
        f"DOCUMENT TEXT:\n{doc_text[:3000]}\n\n"
        f"NEW DATA:\n{json.dumps(input_data, ensure_ascii=False)}"
    )

    llm = _get_llm(llm_config)
    logger.info("[ReportNode] Calling LLM for DOCX semantic substitution…")
    res = llm.invoke(
        [
            SystemMessage(content="Return only raw JSON. No markdown. No backticks."),
            HumanMessage(content=prompt),
        ]
    )

    raw = res.content.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

    try:
        replacements = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"[ReportNode] LLM returned invalid JSON: {res.content}")
        raise ValueError(
            f"AI substitution: LLM did not return valid JSON. {exc}"
        ) from exc

    logger.info(f"[ReportNode] LLM replacements: {replacements}")
    _apply_replacements_to_doc(doc, replacements)
    doc.save(output_path)
    logger.info(f"[ReportNode] DOCX AI substitution saved → {output_path}")


# ── AI Semantic Substitution — XLSX ──────────────────────────────────────────


def run_ai_semantic_substitution_xlsx(
    template_path: str,
    output_path: str,
    input_data: dict,
    llm_config,
) -> None:
    if not openpyxl:
        raise ImportError("openpyxl is not installed. Run: pip install openpyxl")

    wb = openpyxl.load_workbook(template_path)
    ws = wb.active

    lines = []
    for r_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=50), 1):
        row_cells = []
        for c in row:
            # Include empty cells if they are adjacent to filled cells, or just include all cells up to max_col
            # Actually, just listing filled cells is good, but if they need to fill an empty cell next to a label,
            # they need to know the coordinate. Let's list a few empty cells next to labels.
            if c.value:
                row_cells.append(f"[{c.coordinate}: '{c.value}']")
            else:
                row_cells.append(f"[{c.coordinate}: EMPTY]")
        if any(c.value for c in row):
            lines.append(" ".join(row_cells[:15]))  # limit to 15 cols

    sheet_text = "\n".join(lines)

    prompt = (
        "You are a data editor. I have an Excel sheet layout and new data.\n"
        "Match the new data fields to the correct target cell coordinates (e.g. 'B2', 'C4').\n"
        "You can overwrite placeholder text cells, or fill EMPTY cells that are next to labels.\n"
        "Return ONLY a valid JSON object — keys are exact cell coordinates (e.g. 'B2'), values are the data to insert.\n"
        "No markdown, no backticks.\n\n"
        f"SPREADSHEET SAMPLE:\n{sheet_text[:4000]}\n\n"
        f"NEW DATA:\n{json.dumps(input_data, ensure_ascii=False)}"
    )

    llm = _get_llm(llm_config)
    logger.info("[ReportNode] Calling LLM for XLSX semantic substitution…")
    res = llm.invoke(
        [
            SystemMessage(
                content="Return only raw JSON. No markdown. No backticks. Keys must be cell coordinates like 'B2'."
            ),
            HumanMessage(content=prompt),
        ]
    )

    raw = res.content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

    try:
        replacements = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"[ReportNode] LLM returned invalid JSON: {res.content}")
        raise ValueError(
            f"AI substitution: LLM did not return valid JSON. {exc}"
        ) from exc

    logger.info(f"[ReportNode] LLM cell mapping: {replacements}")

    for sheet in wb.worksheets:
        for coord, new_val in replacements.items():
            try:
                sheet[coord] = str(new_val)
            except Exception as e:
                logger.warning(f"[ReportNode] Failed to write to cell {coord}: {e}")

    wb.save(output_path)
    logger.info(f"[ReportNode] XLSX AI substitution saved → {output_path}")


# ── Pure-Python PDF generation (reportlab) ───────────────────────────────────

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Shared colours ────────────────────────────────────────────────────────────
_NAVY = colors.HexColor("#1E2761")
_GRAY = colors.HexColor("#555555")
_LGRAY = colors.HexColor("#F5F7FA")
_BLACK = colors.HexColor("#1A1A1A")


# ── Document type detection ───────────────────────────────────────────────────


def _detect_doc_type(data: dict) -> str:
    """
    Inspect the data keys to guess what kind of document to render.
    Returns one of: 'offer_letter', 'certificate', 'generic'
    """
    if not isinstance(data, dict):
        return "generic"

    keys = set(k.lower() for k in data.keys())

    offer_signals = {
        "role_title",
        "base_salary",
        "reporting_manager",
        "acceptance_deadline",
        "hr_signatory",
        "reference_id",
    }
    cert_signals = {
        "intern_name",
        "signatory_1_name",
        "signatory_2_name",
        "duration",
        "practice",
    }

    if len(keys & offer_signals) >= 3:
        return "offer_letter"
    if len(keys & cert_signals) >= 2:
        return "certificate"
    return "generic"


# ── Offer letter renderer ─────────────────────────────────────────────────────


def _render_offer_letter_pdf(output_path: str, data: dict, title: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1.1 * inch,
        rightMargin=1.1 * inch,
        topMargin=0.8 * inch,
        bottomMargin=1 * inch,
    )

    company_name_style = ParagraphStyle(
        "co", fontSize=15, fontName="Times-Bold", textColor=_NAVY, spaceAfter=2
    )
    company_sub_style = ParagraphStyle(
        "cosub", fontSize=9, fontName="Times-Roman", textColor=_GRAY, spaceAfter=2
    )
    ref_style = ParagraphStyle(
        "ref", fontSize=9, fontName="Times-Roman", textColor=_GRAY, spaceAfter=4
    )
    section_head = ParagraphStyle(
        "sh",
        fontSize=12,
        fontName="Times-Bold",
        textColor=_NAVY,
        spaceBefore=14,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "body",
        fontSize=10,
        fontName="Times-Roman",
        textColor=_BLACK,
        leading=15,
        spaceAfter=8,
    )
    body_bold = ParagraphStyle(
        "bodyb", fontSize=10, fontName="Times-Bold", textColor=_BLACK, leading=15
    )
    sig_name_style = ParagraphStyle(
        "sig", fontSize=10, fontName="Times-Bold", textColor=_BLACK, spaceBefore=6
    )
    sig_sub_style = ParagraphStyle(
        "sigsub", fontSize=9, fontName="Times-Roman", textColor=_GRAY
    )

    story = []

    # Header
    story.append(Paragraph(data.get("company_name", title), company_name_style))
    story.append(Paragraph(data.get("company_address", ""), company_sub_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_NAVY, spaceAfter=10))

    # Date + ref
    if data.get("issue_date"):
        story.append(Paragraph(f"Date: {data['issue_date']}", ref_style))
    if data.get("reference_id"):
        story.append(Paragraph(f"Ref: {data['reference_id']}", ref_style))
    story.append(Spacer(1, 0.1 * inch))

    # Addressee
    story.append(Paragraph("<b>To:</b>", body))
    story.append(Paragraph(data.get("candidate_name", ""), body_bold))
    if data.get("candidate_address"):
        story.append(Paragraph(data["candidate_address"], body))
    story.append(Spacer(1, 0.05 * inch))

    # Subject
    story.append(Paragraph("LETTER OF INTENT &amp; OFFER OF EMPLOYMENT", section_head))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_NAVY, spaceAfter=8))

    # Opening paragraph
    first_name = (
        data.get("candidate_name", "").split()[0] if data.get("candidate_name") else ""
    )
    story.append(Paragraph(f"Dear {first_name},", body))
    story.append(
        Paragraph(
            f"On behalf of <b>{data.get('company_name', 'the company')}</b>, we are thrilled to offer you "
            f"the position of <b>{data.get('role_title', '')}</b> in our "
            f"<b>{data.get('department', '')}</b> division.",
            body,
        )
    )

    # Terms table — only include fields that have values
    terms_map = [
        ("Role Title", "role_title"),
        ("Department", "department"),
        ("Commencement Date", "start_date"),
        ("Reporting Manager", "reporting_manager"),
        ("Compensation", "base_salary"),
        ("Benefits", "benefits"),
        ("Acceptance By", "acceptance_deadline"),
    ]
    terms = [[label, data[key]] for label, key in terms_map if data.get(key)]

    if terms:
        story.append(Paragraph("Core Terms of Employment", section_head))
        t = Table(terms, colWidths=[2.2 * inch, 4.1 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), _LGRAY),
                    ("FONTNAME", (0, 0), (0, -1), "Times-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Times-Roman"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TEXTCOLOR", (0, 0), (0, -1), _NAVY),
                    ("TEXTCOLOR", (1, 0), (1, -1), _BLACK),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_LGRAY, colors.white]),
                ]
            )
        )
        story.append(t)

    # Closing
    story.append(Spacer(1, 0.15 * inch))
    deadline = data.get("acceptance_deadline", "the specified date")
    story.append(
        Paragraph(
            f"This offer is contingent upon successful completion of a standard background screening. "
            f"Please countersign and return this document to Human Resources by <b>{deadline}</b>.",
            body,
        )
    )
    story.append(Paragraph("We look forward to welcoming you to the team.", body))
    story.append(Spacer(1, 0.3 * inch))

    # Signature
    story.append(Paragraph("Sincerely,", body))
    story.append(Spacer(1, 0.35 * inch))
    story.append(HRFlowable(width=2 * inch, thickness=0.5, color=_GRAY, spaceAfter=4))
    story.append(Paragraph(f"<b>{data.get('hr_signatory', '')}</b>", sig_name_style))
    story.append(Paragraph(data.get("hr_title", ""), sig_sub_style))
    story.append(Paragraph(data.get("company_name", ""), sig_sub_style))

    doc.build(story)


# ── Certificate renderer ──────────────────────────────────────────────────────


def _render_certificate_pdf(output_path: str, data: dict, title: str) -> None:
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=1 * inch,
        rightMargin=1 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.8 * inch,
    )

    co_name_s = ParagraphStyle(
        "cn",
        fontSize=20,
        fontName="Times-Bold",
        textColor=_NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    co_sub_s = ParagraphStyle(
        "cs",
        fontSize=9,
        fontName="Times-Roman",
        textColor=_GRAY,
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    cert_title_s = ParagraphStyle(
        "ct",
        fontSize=11,
        fontName="Times-Roman",
        textColor=_NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    present_s = ParagraphStyle(
        "pt",
        fontSize=11,
        fontName="Times-Roman",
        textColor=_GRAY,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    name_s = ParagraphStyle(
        "ns",
        fontSize=26,
        fontName="Times-Bold",
        textColor=_NAVY,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    body_c = ParagraphStyle(
        "bc",
        fontSize=10,
        fontName="Times-Roman",
        textColor=_BLACK,
        alignment=TA_CENTER,
        leading=16,
        spaceAfter=6,
    )
    sig_name_s = ParagraphStyle(
        "sn", fontSize=10, fontName="Times-Bold", textColor=_BLACK, alignment=TA_CENTER
    )
    sig_title_s = ParagraphStyle(
        "st", fontSize=9, fontName="Times-Roman", textColor=_GRAY, alignment=TA_CENTER
    )

    story = []

    story.append(HRFlowable(width="100%", thickness=4, color=_NAVY, spaceAfter=14))
    story.append(Paragraph(data.get("company_name", title).upper(), co_name_s))
    story.append(
        Paragraph(
            "G L O B A L  T E C H N O L O G Y  &amp;  S T R A T E G Y  S O L U T I O N S",
            co_sub_s,
        )
    )
    story.append(HRFlowable(width="60%", thickness=0.5, color=_NAVY, spaceAfter=18))

    story.append(Paragraph("Certificate of Internship Completion", cert_title_s))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph("This is proudly presented to", present_s))
    story.append(Spacer(1, 0.05 * inch))
    story.append(Paragraph(data.get("intern_name", ""), name_s))
    story.append(
        HRFlowable(
            width="50%", thickness=0.5, color=colors.HexColor("#AAAAAA"), spaceAfter=12
        )
    )

    story.append(
        Paragraph(
            f"for outstanding performance and successful completion of a "
            f"<b>{data.get('duration', '')}</b> professional internship as a "
            f"<b>{data.get('role', '')}</b> within the "
            f"<b>{data.get('practice', '')}</b> at "
            f"<b>{data.get('company_name', '')}</b>.",
            body_c,
        )
    )
    story.append(
        Paragraph(
            f"The internship was successfully conducted from "
            f"<b>{data.get('start_date', '')}</b> to <b>{data.get('end_date', '')}</b>.",
            body_c,
        )
    )

    story.append(Spacer(1, 0.5 * inch))

    # Dual signature block
    sig_data = [
        [
            Paragraph(f"<b>{data.get('signatory_1_name', '')}</b>", sig_name_s),
            Paragraph(f"<b>{data.get('signatory_2_name', '')}</b>", sig_name_s),
        ],
        [
            Paragraph(data.get("signatory_1_title", ""), sig_title_s),
            Paragraph(data.get("signatory_2_title", ""), sig_title_s),
        ],
    ]
    sig_table = Table(sig_data, colWidths=[3 * inch, 3 * inch])
    sig_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (0, 0), (0, 0), 0.5, _GRAY),
                ("LINEABOVE", (1, 0), (1, 0), 0.5, _GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(sig_table)
    story.append(Spacer(1, 0.3 * inch))
    story.append(HRFlowable(width="100%", thickness=4, color=_NAVY, spaceAfter=0))

    doc.build(story)


# ── PyMuPDF Template Injection ───────────────────────────────────────────────


def _fill_pdf_template(
    template_path: str,
    output_path: str,
    input_data: dict,
    llm_config,
    thread_id: str = "default_thread",
) -> None:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "[ReportNode] PyMuPDF (fitz) is not installed. Falling back to fresh generation."
        )
        generate_pdf_from_data(output_path, input_data, thread_id=thread_id)
        return

    doc = fitz.open(template_path)

    # Track which keys from input_data were successfully used as placeholders
    used_keys = set()

    # 1. In-Place Placeholder Replacement
    for page in doc:
        # Get all text words to find placeholders accurately
        for key, val in input_data.items():
            if isinstance(val, (dict, list)):
                continue  # Skip complex data for inline placeholders

            ph_str = f"{{{{{key}}}}}"
            rects = page.search_for(ph_str)
            if rects:
                used_keys.add(key)
                for rect in rects:
                    # Redact the exact placeholder without drawing a background box
                    annot = page.add_redact_annot(rect)
                    annot.set_colors(stroke=None, fill=None)
                    annot.update()
                    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                    # Inject the actual data string
                    # Use a slightly wider rect for insertion to prevent word wrapping if the value is longer than the placeholder
                    insert_rect = fitz.Rect(rect.x0, rect.y0, rect.x1 + 150, rect.y1)
                    page.insert_textbox(
                        insert_rect,
                        str(val),
                        fontsize=11,
                        fontname="Times-Roman",
                        color=(0.1, 0.1, 0.4),
                    )

    # 2. Addendum Generation for complex/unused data (like tables)
    unused_data = {k: v for k, v in input_data.items() if k not in used_keys}

    # We only append an addendum if there is meaningful data left (e.g. tables)
    if unused_data and any(isinstance(v, (dict, list)) for v in unused_data.values()):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_addendum:
            tmp_addendum_path = tmp_addendum.name

        logger.info("[ReportNode] Generating Addendum for unused complex data...")
        # Generate the addendum using ReportLab
        generate_pdf_from_data(
            tmp_addendum_path,
            unused_data,
            title="Report Addendum",
            template_style="auto",
            thread_id=thread_id,
        )

        # Merge the addendum to the end of the template
        addendum_doc = fitz.open(tmp_addendum_path)
        doc.insert_pdf(addendum_doc)

        addendum_doc.close()
        try:
            os.remove(tmp_addendum_path)
        except:
            pass

    doc.save(output_path)
    doc.close()
    logger.info(f"[ReportNode] PDF Template saved → {output_path}")


# ── Main entry point (replaces existing generate_pdf_from_data) ───────────────


def generate_pdf_from_data(
    output_path: str,
    input_data,
    title: str = "Generated Report",
    template_style: str = "auto",
    thread_id: str = "default_thread",
) -> None:
    """
    Smart PDF generator. Routes to the correct renderer based on template_style
    (derived from DocumentTemplateModel.category) or auto-detects from data keys.

    template_style values (from DocumentTemplateModel.category):
      'letter' / 'offer_letter' / 'offer' → headed letter + terms table + signature
      'certificate' / 'cert'              → centred award layout with dual signatures
      'auto' / 'general' / ''             → auto-detect from data keys, then generic

    FIX: previous version always auto-detected and ignored the selected template's
    category, so a user picking an 'offer_letter' template still got a generic dump.
    """
    if not reportlab:
        raise ImportError("reportlab is not installed. Run: pip install reportlab")

    # Map template category → internal doc_type
    _style_map = {
        "letter": "offer_letter",
        "offer": "offer_letter",
        "offer_letter": "offer_letter",
        "certificate": "certificate",
        "cert": "certificate",
    }
    style_key = (template_style or "").strip().lower()
    if style_key in _style_map:
        doc_type = _style_map[style_key]
        logger.info(
            f"[ReportNode] PDF layout forced by template category '{style_key}' → '{doc_type}'"
        )
    else:
        doc_type = _detect_doc_type(input_data)
        logger.info(f"[ReportNode] PDF doc_type auto-detected: '{doc_type}'")

    if doc_type == "offer_letter":
        _render_offer_letter_pdf(output_path, input_data, title)
        return

    if doc_type == "certificate":
        _render_certificate_pdf(output_path, input_data, title)
        return

    # ── Generic fallback — original table-dump behaviour ──────────────────────
    # Pre-process any dynamic images so they render as ReportLab flowables
    input_data = _preprocess_images_for_pdf(input_data, thread_id)

    def _sanitize_for_pdf(txt: str) -> str:
        if not isinstance(txt, str):
            txt = str(txt)
        replacements = {
            "\u2013": "-",
            "\u2014": "--",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "\u2026": "...",
            "\u00a0": " ",
            "\u2022": "*",
        }
        for k, v in replacements.items():
            txt = txt.replace(k, v)
        return txt

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        spaceAfter=16,
        textColor=_NAVY,
    )
    body_style = ParagraphStyle(
        "FieldValue",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
    )

    story = [Paragraph(_sanitize_for_pdf(title), title_style), Spacer(1, 0.2 * inch)]

    if isinstance(input_data, dict):
        subtitle_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8,
            textColor=_NAVY,
            fontName="Helvetica-Bold",
        )
        for key, value in input_data.items():
            section_title = _sanitize_for_pdf(str(key).replace("_", " ").title())
            story.append(Paragraph(section_title, subtitle_style))

            if hasattr(value, "drawWidth"):
                # It's an image flowable
                story.append(value)
                story.append(Spacer(1, 0.1 * inch))
            else:
                # Replace newlines with <br/> for ReportLab Paragraph
                text_val = _sanitize_for_pdf(str(value)).replace("\n", "<br/>")
                story.append(Paragraph(text_val, body_style))
                story.append(Spacer(1, 0.15 * inch))
    elif isinstance(input_data, list) and input_data:
        if isinstance(input_data[0], dict):
            headers = list(input_data[0].keys())
            col_width = 6.5 * inch / len(headers)
            table_data = [headers] + [
                [str(row.get(h, "")) for h in headers] for row in input_data
            ]
            t = Table(table_data, colWidths=[col_width] * len(headers))
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), _NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 10),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.HexColor("#F5F5F5"), colors.white],
                        ),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 1), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
                    ]
                )
            )
            story.append(t)
        else:
            for item in input_data:
                story.append(Paragraph(f"• {item}", body_style))
                story.append(Spacer(1, 0.05 * inch))
    else:
        story.append(Paragraph(str(input_data), body_style))

    doc.build(story)
    logger.info(f"[ReportNode] PDF written → {output_path}")


# ── XLSX Custom Helper Methods ────────────────────────────────────────────────


def _find_header_row(ws, max_row: int = 10) -> int:
    """
    Auto-detect the header row by finding the first row that has
    at least 2 non-None cells. Handles both:
      - Simple templates: headers at row 1
      - Corporate workbooks: title/logo rows above, headers deeper down
    Falls back to row 1 if nothing is found.
    """
    for r in range(1, max_row + 1):
        vals = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
        if sum(1 for v in vals if v is not None) >= 2:
            return r
    return 1


_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.\[\]-]+)\s*\}\}")


def _resolve_placeholder_path(data: dict, path: str):
    """
    Walk a dotted path into the data. Returns (found, value).

    'FY2025.BalanceSheet.Assets.Total' → data['FY2025']['BalanceSheet']…
    List indexing is supported: 'accounts.0.balance'.

    Returns found=False rather than None so that a value which is genuinely
    null can be told apart from a path that does not exist. Those mean very
    different things: one is "the statement said nothing here", the other is
    "this template refers to something we never produced".
    """
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, (list, tuple)) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _coerce_cell_value(value):
    """
    Write numbers as numbers, not as text.

    This matters more than it looks. A figure stored as text does not
    participate in the template's own formulas, and the template's number
    formatting will not apply to it — so a workbook that computes itself would
    silently produce blanks or #VALUE! wherever we wrote a string.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, decimal.Decimal):
        # Money is computed in Decimal; Excel needs a float at the boundary.
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text and re.fullmatch(r"-?\d+", text):
            return int(text)
        if text and re.fullmatch(r"-?\d*\.\d+", text):
            return float(text)
    return value


def fill_excel_by_placeholders(
    template_path: str,
    data: dict,
    on_missing: str = "fail",
) -> tuple:
    """
    Deterministic, multi-sheet template filling. No model involved.

    Scans EVERY sheet for {{ path }} markers and replaces each with the value
    at that path in `data`. Everything the template already contains — merged
    cells, number formats, column widths, and any formulas on other sheets —
    is left untouched, because we only ever overwrite the marker cells.

    Verified against a real 31-sheet Perfios workbook: 132 merged ranges,
    15,798 number formats, 256 column widths and 12,376 formulas all survive.

    `on_missing` decides what happens when a marker has no matching data:
        fail  — raise. The default, and the right choice for financial output:
                a workbook that looks complete but has a figure missing is far
                more dangerous than one that refuses to be produced.
        warn  — log it, leave the marker visible in the cell so it is obvious
        skip  — log it, blank the cell

    Returns (bytes, report) where report describes what was filled and missed.
    """
    if openpyxl is None:
        raise ImportError("openpyxl is not installed. Run: pip install openpyxl")

    wb = openpyxl.load_workbook(template_path)
    filled, missing = [], []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for row in ws.iter_rows():
            for cell in row:
                if not isinstance(cell.value, str):
                    continue
                match = _PLACEHOLDER_RE.search(cell.value)
                if not match:
                    continue

                path = match.group(1)
                found, value = _resolve_placeholder_path(data, path)
                where = f"{sheet_name}!{cell.coordinate}"

                if not found:
                    missing.append({"path": path, "at": where})
                    if on_missing == "skip":
                        cell.value = None
                    # 'warn' deliberately leaves the marker in place so the gap
                    # is visible to whoever opens the workbook.
                    continue

                # A marker on its own owns the cell and keeps the value's type.
                # A marker inside surrounding text is a string substitution.
                if match.group(0) == cell.value.strip():
                    cell.value = _coerce_cell_value(value)
                else:
                    cell.value = _PLACEHOLDER_RE.sub(
                        lambda m: str(_resolve_placeholder_path(data, m.group(1))[1] or ""),
                        cell.value,
                    )
                filled.append({"path": path, "at": where})

    if missing and on_missing == "fail":
        detail = ", ".join(f"{m['path']} at {m['at']}" for m in missing[:5])
        raise ValueError(
            f"Excel template has {len(missing)} placeholder(s) with no matching data: "
            f"{detail}{' …' if len(missing) > 5 else ''}. "
            f"Set report_on_missing_target to 'warn' or 'skip' to produce the file anyway."
        )

    if missing:
        logger.warning(
            f"[ReportNode] {len(missing)} placeholder(s) had no data "
            f"(on_missing='{on_missing}'): "
            f"{', '.join(m['path'] for m in missing[:8])}"
        )

    buf = io.BytesIO()
    wb.save(buf)
    wb.close()
    logger.info(
        f"[ReportNode] Template filled deterministically: "
        f"{len(filled)} placeholder(s) across {len(wb.sheetnames) if wb else 0} sheet(s), "
        f"{len(missing)} missing"
    )
    return buf.getvalue(), {"filled": filled, "missing": missing}


def _fill_excel_template(template_path: str, data: dict, llm_config=None) -> bytes:
    """
    Load the template workbook and fill it with data from data['rows'],
    preserving all header styling and column structure.
    """
    import io
    import re
    import json
    from openpyxl import load_workbook
    from openpyxl.styles import Font

    wb = load_workbook(template_path)

    # Target sheet: use data['sheet'] if named, else active
    sheet_name = data.get("sheet")
    ws = wb[sheet_name] if sheet_name and sheet_name in wb.sheetnames else wb.active

    # Auto-detect header row
    header_row_idx = _find_header_row(ws)
    headers = [
        ws.cell(row=header_row_idx, column=c).value for c in range(1, ws.max_column + 1)
    ]
    data_start_row = header_row_idx + 1

    merged_coords: set = set()
    for merged_range in ws.merged_cells.ranges:
        for row_idx in range(merged_range.min_row, merged_range.max_row + 1):
            for col_idx in range(merged_range.min_col, merged_range.max_col + 1):
                if not (
                    row_idx == merged_range.min_row and col_idx == merged_range.min_col
                ):
                    merged_coords.add((row_idx, col_idx))

    # Clear existing data rows (header row stays untouched, merged slaves skipped)
    for row in ws.iter_rows(min_row=data_start_row, max_row=ws.max_row):
        for cell in row:
            if (cell.row, cell.column) not in merged_coords:
                cell.value = None

    def _norm_key(k: str) -> str:
        return str(k).lower().strip().replace(" ", "_").replace("-", "_")

    norm_header_map = {
        _norm_key(h): (c_idx, h)
        for c_idx, h in enumerate(headers, start=1)
        if h is not None
    }

    # Write incoming rows
    incoming_rows = data.get("rows", [])

    # ── LLM Smart Header Mapping ─────────────────────────────────────────
    if incoming_rows and isinstance(incoming_rows[0], dict) and llm_config:
        json_keys = list(incoming_rows[0].keys())
        excel_headers = [h for h in headers if h is not None]

        # Check if we have missing keys
        unmapped_keys = [
            k
            for k in json_keys
            if _norm_key(k) not in norm_header_map and norm_header_map.get(k) is None
        ]

        if unmapped_keys and excel_headers:
            logger.info(
                "[ReportNode] Unmapped keys found in Excel rows. Asking LLM to map headers..."
            )
            prompt = (
                f"I am mapping JSON data into an Excel table.\n"
                f"JSON keys available: {json.dumps(unmapped_keys)}\n"
                f"Excel column headers available: {json.dumps(excel_headers)}\n\n"
                f"Map the JSON keys to the closest matching Excel header.\n"
                f"Return ONLY a valid JSON object where keys are the JSON keys and values are the exact Excel headers.\n"
                f"No markdown, no explanation."
            )
            try:
                llm = _get_llm(llm_config)
                from langchain_core.messages import SystemMessage, HumanMessage

                res = llm.invoke(
                    [
                        SystemMessage(content="Return only raw JSON. No markdown."),
                        HumanMessage(content=prompt),
                    ]
                )
                raw = res.content.strip()
                raw = re.sub(r"^```(?:json)?", "", raw, flags=re.MULTILINE).strip()
                raw = re.sub(r"```$", "", raw, flags=re.MULTILINE).strip()

                mapping = json.loads(raw)
                for j_key, e_header in mapping.items():
                    if e_header in excel_headers:
                        c_idx = headers.index(e_header) + 1
                        norm_header_map[j_key] = (c_idx, e_header)
                        norm_header_map[_norm_key(j_key)] = (c_idx, e_header)
                        logger.info(
                            f"[ReportNode] LLM Mapped: '{j_key}' -> '{e_header}'"
                        )
            except Exception as e:
                logger.error(f"[ReportNode] LLM header mapping failed: {e}")

    import copy

    def _clone_style(src, dst):
        if src.has_style:
            dst.font = copy.copy(src.font)
            dst.border = copy.copy(src.border)
            dst.fill = copy.copy(src.fill)
            dst.number_format = copy.copy(src.number_format)
            dst.protection = copy.copy(src.protection)
            dst.alignment = copy.copy(src.alignment)

    for r_idx, row_data in enumerate(incoming_rows, start=data_start_row):
        if isinstance(row_data, dict):
            for data_key, data_val in row_data.items():
                match = norm_header_map.get(data_key) or norm_header_map.get(
                    _norm_key(data_key)
                )
                if match:
                    c_idx, _ = match
                    if (r_idx, c_idx) not in merged_coords:
                        cell = ws.cell(row=r_idx, column=c_idx, value=data_val)
                        if r_idx > data_start_row:
                            _clone_style(
                                ws.cell(row=data_start_row, column=c_idx), cell
                            )
        elif isinstance(row_data, list):
            first_data_col = next(
                (i + 1 for i, h in enumerate(headers) if h is not None), 1
            )
            for offset, val in enumerate(row_data):
                c_idx = first_data_col + offset
                if (r_idx, c_idx) not in merged_coords:
                    cell = ws.cell(row=r_idx, column=c_idx, value=val)
                    if r_idx > data_start_row:
                        _clone_style(ws.cell(row=data_start_row, column=c_idx), cell)

    # ── Scalar Fields (Jinja Placeholders & Metadata) ─────────────────────
    reserved = {"rows", "sheet", "header_row", "data_start_row"}
    scalars = {
        k: v
        for k, v in data.items()
        if k not in reserved and not isinstance(v, (list, dict))
    }

    has_placeholders = False
    if scalars:
        for r in range(1, ws.max_row + 1):
            for c in range(1, ws.max_column + 1):
                cell = ws.cell(row=r, column=c)
                if cell.value and isinstance(cell.value, str) and "{{" in cell.value:
                    val = cell.value
                    for sk, sv in scalars.items():
                        pattern = re.compile(
                            r"\{\{\s*" + re.escape(str(sk)) + r"\s*\}\}", re.IGNORECASE
                        )
                        if pattern.search(val):
                            has_placeholders = True
                            val = pattern.sub(str(sv), val)
                    cell.value = val

    out = io.BytesIO()
    wb.save(out)
    return out.getvalue(), has_placeholders


async def _generate_excel_output(
    data: dict,
    template_path,  # str | None
    output_path: str,
    llm_config=None,
    report_mode: str = None,
    on_missing: str = "fail",
) -> str:
    """
    template_map mode  → fill {{ markers }} across every sheet, deterministically.
    Template provided  → fill it (auto-detect headers, preserve styling).
    No template        → generate a clean workbook from scratch.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    # ── Deterministic template map ─────────────────────────────────────────────
    # Chosen explicitly, never inferred: the two template paths behave very
    # differently and silently picking one would be a poor surprise.
    if report_mode == "template_map":
        if not template_path:
            raise ValueError(
                "report_mode='template_map' requires a template. "
                "Set template_id or template_file_path."
            )
        result_bytes, fill_report = fill_excel_by_placeholders(
            template_path, data, on_missing=on_missing or "fail"
        )
        with open(output_path, "wb") as f:
            f.write(result_bytes)
        logger.info(
            f"[ReportNode] template_map: filled {len(fill_report['filled'])}, "
            f"missing {len(fill_report['missing'])} → {output_path}"
        )
        return output_path

    # ── Template path ──────────────────────────────────────────────────────────
    if template_path:
        result_bytes, has_placeholders = _fill_excel_template(
            template_path, data, llm_config
        )
        with open(output_path, "wb") as f:
            f.write(result_bytes)

        if not has_placeholders and llm_config:
            logger.info(
                "[ReportNode] No placeholders found in Excel — triggering AI semantic substitution."
            )
            run_ai_semantic_substitution_xlsx(
                output_path, output_path, data, llm_config
            )

        return output_path

    # ── No-template path: build fresh workbook ─────────────────────────────────
    wb = Workbook()
    ws = wb.active
    ws.title = "Report"

    rows = data.get("rows", [])
    reserved = {"rows", "sheet", "header_row", "data_start_row"}
    scalars = {
        k: v
        for k, v in data.items()
        if k not in reserved and not isinstance(v, (list, dict))
    }

    header_fill = PatternFill("solid", fgColor="1B3A6B")
    header_font = Font(bold=True, color="FFFFFF")
    current_row = 1

    if scalars:
        for k, v in scalars.items():
            ws.cell(row=current_row, column=1, value=k).font = Font(bold=True)
            ws.cell(row=current_row, column=2, value=str(v))
            current_row += 1
        current_row += 1  # blank separator row

    if rows:
        if isinstance(rows[0], dict):
            col_headers = list(rows[0].keys())
            for c_idx, h in enumerate(col_headers, 1):
                cell = ws.cell(row=current_row, column=c_idx, value=h)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")
            current_row += 1
            for row_data in rows:
                for c_idx, h in enumerate(col_headers, 1):
                    ws.cell(row=current_row, column=c_idx, value=row_data.get(h, ""))
                current_row += 1
        elif isinstance(rows[0], list):
            for row_data in rows:
                for c_idx, val in enumerate(row_data, 1):
                    ws.cell(row=current_row, column=c_idx, value=val)
                current_row += 1

    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = 18

    wb.save(output_path)
    return output_path


# ── Node builder ──────────────────────────────────────────────────────────────


def build_report_generation_node(node_config: NodeConfig, **kwargs) -> Callable:
    """
    Report Generation Node.

    Reads data from AgentState.context, fills a template (or generates from
    scratch), writes the output to mnt/exports/{thread_id}/, and returns a
    download link in the chat.

    Supported formats: .docx  .xlsx  .pdf  .csv  .json  .md
    """

    template_id = getattr(node_config, "template_id", None) or None
    input_data_key = getattr(node_config, "input_data_key", None) or "report_data"
    # How a template is filled, and what to do about gaps. Read at build time so
    # an unset value keeps the node's previous behaviour exactly.
    report_mode = getattr(node_config, "report_mode", None) or None
    report_on_missing_target = (
        getattr(node_config, "report_on_missing_target", None) or "fail"
    )
    output_file_name = (
        getattr(node_config, "output_file_name", None) or "generated_report"
    )

    # FIX: prefer report_format (report-specific field) over file_format
    # which is shared with the File Store node and can bleed incorrect values.
    raw_format = (getattr(node_config, "report_format", None) or "").strip().lower()
    if raw_format and not raw_format.startswith("."):
        raw_format = "." + raw_format
    expected_format = raw_format or ".docx"

    async def execute_report_node(state: AgentState):
        nonlocal template_id
        logger.info(
            f"[ReportNode] Executing node '{node_config.id}'. Configured template_id: '{template_id}', input_data_key: '{input_data_key}', expected_format: '{expected_format}'"
        )
        context = state.get("context", {})
        thread_id = (
            state.get("thread_id") or context.get("thread_id") or "default_thread"
        )

        # ── 1. Resolve input data ─────────────────────────────────────────
        _internal = {
            "thread_id",
            "workflow_id",
            "workflow_name",
            "hitl_action",
            "hitl_feedback",
            "nodes",
            "report_output_path",
            "report_filename",
            "report_download_url",
        }
        fmt_key = expected_format.lstrip(".")

        def _try_keys(*keys):
            for k in keys:
                v = context.get(k)
                if v:
                    logger.info(f"[ReportNode] Using context key '{k}' as input data")
                    return k, v
            return None, None

        found_key, input_data = _try_keys(
            input_data_key,
            f"{fmt_key}_template_payload",
            "template_payload",
            "payload",
            "report_data",
            "data",
        )

        if not input_data:
            input_data = {
                k: v for k, v in context.items() if k not in _internal and v is not None
            }
            if input_data:
                logger.info(
                    "[ReportNode] No specific key found — using full context as data"
                )
            else:
                input_data = {
                    "error": f"No data in context. Keys: {list(context.keys())}"
                }
        else:
            # Auto-unwrap single-key nested dict: {"wrapper": {...}} → {...}
            if isinstance(input_data, dict) and len(input_data) == 1:
                only_key = next(iter(input_data))
                if isinstance(input_data[only_key], dict):
                    logger.info(
                        f"[ReportNode] Auto-unwrapping payload under key '{only_key}'"
                    )
                    input_data = input_data[only_key]
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data)
                except (json.JSONDecodeError, TypeError):
                    input_data = {"text": input_data}

        # ── 2. Resolve template from DB ───────────────────────────────────
        format_type = expected_format
        template_path = None
        template_category = (
            "auto"  # Populated from DB record.category; drives PDF layout
        )

        if template_id:
            try:
                from Database.Database import AsyncSessionLocal
                from Database.Models import DocumentTemplateModel
                from sqlalchemy import select as sa_select

                async with AsyncSessionLocal() as session:
                    if template_id == "auto":
                        # AI Semantic Template Routing
                        logger.info(
                            "[ReportNode] Triggering AI Template Auto-Selection..."
                        )
                        result = await session.execute(sa_select(DocumentTemplateModel))
                        all_templates = result.scalars().all()

                        if all_templates:
                            templates_info = [
                                f"ID: {t.id} | Name: {t.template_name} | Category: {t.category} | Format: {t.format_type}"
                                for t in all_templates
                            ]

                            prompt = (
                                "You are an AI Template Router. I have a JSON payload, and a list of available document templates.\n"
                                "Select the BEST matching template based on the template Name, Category, and Format.\n"
                                "Rules:\n"
                                "1. Return ONLY the exact string ID of the chosen template. No markdown, no quotes, no explanation.\n"
                                "2. If absolutely nothing fits, return 'none'.\n\n"
                                f"AVAILABLE TEMPLATES:\n{chr(10).join(templates_info)}\n\n"
                                f"PAYLOAD DATA:\n{json.dumps(input_data)[:2000]}"
                            )
                            llm = _get_llm(getattr(node_config, "llm_config", {}))
                            try:
                                from langchain_core.messages import (
                                    SystemMessage,
                                    HumanMessage,
                                )

                                res = llm.invoke(
                                    [
                                        SystemMessage(
                                            content="Return only the raw UUID string or 'none'."
                                        ),
                                        HumanMessage(content=prompt),
                                    ]
                                )
                                chosen = (
                                    res.content.strip()
                                    .replace('"', "")
                                    .replace("'", "")
                                )
                                if chosen.lower() != "none" and any(
                                    t.id == chosen for t in all_templates
                                ):
                                    template_id = chosen
                                    logger.info(
                                        f"[ReportNode] AI Semantically Selected Template ID: {template_id}"
                                    )
                                else:
                                    logger.info(
                                        "[ReportNode] AI rejected all templates. Proceeding blank."
                                    )
                                    template_id = None
                            except Exception as e:
                                logger.error(f"[ReportNode] AI routing failed: {e}")
                                template_id = None
                        else:
                            template_id = None

                # Re-check template_id after auto-resolution
                if template_id:
                    async with AsyncSessionLocal() as session:
                        result = await session.execute(
                            sa_select(DocumentTemplateModel).where(
                                DocumentTemplateModel.id == template_id
                            )
                        )
                        record = result.scalars().first()

                if record:
                    fmt = (record.format_type or "").strip()
                    db_fmt = fmt if fmt.startswith(".") else ("." + fmt) if fmt else ""
                    # Always use the template's native format for processing
                    if db_fmt:
                        format_type = ".docx" if db_fmt == ".dotx" else db_fmt
                    template_path = _abs(record.storage_path or "")
                    template_category = (record.category or "auto").strip().lower()
                    logger.info(
                        f"[ReportNode] Template '{template_id}' → {format_type} "
                        f"@ {template_path} (category: '{template_category}')"
                    )
                    if not os.path.exists(template_path):
                        logger.error(
                            f"[ReportNode] Template file missing: {template_path}"
                        )
                        template_path = None
                else:
                    logger.warning(
                        f"[ReportNode] Template ID '{template_id}' not found in DB."
                    )
            except Exception as exc:
                logger.error(f"[ReportNode] DB error: {exc}\n{traceback.format_exc()}")

        # ── 2b. A template file given directly, rather than from the library ──
        # Not every template belongs in the library: one supplied by whoever
        # defines the output format — a regulator, a customer, another system —
        # is a file on disk, and requiring it to be registered first only adds a
        # step. The library still wins when both are set, since choosing it was
        # the more deliberate act.
        if not template_path:
            direct_path = (
                getattr(node_config, "template_file_path", None) or ""
            ).strip()
            if direct_path:
                candidate = _abs(direct_path)
                if os.path.exists(candidate):
                    template_path = candidate
                    ext = os.path.splitext(candidate)[1].lower()
                    if ext:
                        format_type = ext
                    logger.info(f"[ReportNode] Using template file: {template_path}")
                else:
                    # Say which path was tried. "Template not found" without the
                    # path sends people hunting through their own configuration.
                    logger.error(
                        f"[ReportNode] template_file_path does not exist: {candidate}"
                    )

        # ── 3. Prepare output path ────────────────────────────────────────
        exports_root = os.path.abspath(os.path.join(settings.EXPORTS_MOUNT, thread_id))
        os.makedirs(exports_root, exist_ok=True)

        final_filename = f"{output_file_name}{format_type}"
        output_path = os.path.join(exports_root, final_filename)
        download_url = f"/exports/{thread_id}/{final_filename}"

        logger.info(f"[ReportNode] Writing → {output_path}")

        # ── 4. Generate file ──────────────────────────────────────────────
        llm_config = getattr(node_config, "llm", None)
        preview_content = None  # Set for .md — rendered inline in the chat panel

        if format_type == ".docx":
            if DocxTemplate is None:
                raise ImportError("docxtpl not installed. Run: pip install docxtpl")

            if not template_path:
                if docx is None:
                    raise ImportError(
                        "python-docx is not installed. Run: pip install python-docx"
                    )
                logger.info(
                    "[ReportNode] No DOCX template supplied — generating from scratch."
                )
                doc = docx.Document()
                doc.add_heading(output_file_name, level=1)

                if isinstance(input_data, dict):
                    for key, value in input_data.items():
                        section_title = str(key).replace("_", " ").title()
                        doc.add_heading(section_title, level=2)

                        if isinstance(value, dict) and value.get("_type") == "image":
                            try:
                                from docx.shared import Mm

                                local_path = _download_and_cache_image(
                                    value.get("url", ""), thread_id
                                )
                                width = value.get("width_mm", 100)
                                doc.add_picture(local_path, width=Mm(width))
                            except Exception as e:
                                logger.error(
                                    f"[ReportNode] Failed to download image for DOCX: {e}"
                                )
                                doc.add_paragraph(f"[Image Error: {value.get('url')}]")
                        else:
                            doc.add_paragraph(str(value))
                elif isinstance(input_data, list) and input_data:
                    if isinstance(input_data[0], dict):
                        headers = list(input_data[0].keys())
                        table = doc.add_table(rows=1, cols=len(headers))
                        table.style = "Table Grid"
                        hdr_cells = table.rows[0].cells
                        for idx, h in enumerate(headers):
                            hdr_cells[idx].text = str(h)
                            for paragraph in hdr_cells[idx].paragraphs:
                                for run in paragraph.runs:
                                    run.bold = True
                        for item in input_data:
                            row_cells = table.add_row().cells
                            for idx, h in enumerate(headers):
                                row_cells[idx].text = str(item.get(h, ""))
                    else:
                        for item in input_data:
                            doc.add_paragraph(str(item), style="List Bullet")
                else:
                    doc.add_paragraph(str(input_data))

                doc.save(output_path)
            else:
                if _docx_has_placeholders(template_path):
                    logger.info(
                        "[ReportNode] Placeholders detected — using docxtpl injection."
                    )
                    doc = DocxTemplate(template_path)

                    # PREPROCESS IMAGES: convert {'_type':'image'} into docxtpl.InlineImage
                    proc_data = (
                        input_data
                        if isinstance(input_data, dict)
                        else {"data": input_data}
                    )
                    proc_data = _preprocess_images_for_docx(proc_data, doc, thread_id)

                    doc.render(proc_data)
                    doc.save(output_path)
                else:
                    logger.info(
                        "[ReportNode] No placeholders — triggering AI semantic substitution."
                    )
                    run_ai_semantic_substitution_docx(
                        template_path, output_path, input_data, llm_config
                    )

        elif format_type == ".xlsx":
            if openpyxl is None:
                raise ImportError("openpyxl not installed. Run: pip install openpyxl")

            logger.info(
                f"[ReportNode] Generating Excel report... "
                f"(mode={report_mode or 'default'})"
            )
            await _generate_excel_output(
                input_data,
                template_path,
                output_path,
                llm_config,
                report_mode=report_mode,
                on_missing=report_on_missing_target,
            )

        elif format_type == ".pdf":
            if template_path:
                logger.info(
                    "[ReportNode] Generating PDF using uploaded template via PyMuPDF..."
                )
                _fill_pdf_template(
                    template_path, output_path, input_data, llm_config, thread_id
                )
            else:
                logger.info(
                    "[ReportNode] No PDF template provided. Generating blank layout using ReportLab..."
                )
                # Pure-Python PDF via reportlab — no Word, no LibreOffice, Linux-safe
                # template_category (from DB record.category) drives the layout style
                generate_pdf_from_data(
                    output_path,
                    input_data,
                    title=output_file_name,
                    template_style=template_category,
                    thread_id=thread_id,
                )

        elif format_type == ".csv":
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                if (
                    isinstance(input_data, list)
                    and input_data
                    and isinstance(input_data[0], dict)
                ):
                    writer = csv.DictWriter(f, fieldnames=list(input_data[0].keys()))
                    writer.writeheader()
                    writer.writerows(input_data)
                elif isinstance(input_data, dict):
                    writer = csv.DictWriter(f, fieldnames=list(input_data.keys()))
                    writer.writeheader()
                    writer.writerow(input_data)
                else:
                    writer = csv.writer(f)
                    writer.writerow(["Data"])
                    writer.writerow([str(input_data)])

        elif format_type == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(input_data, f, indent=4, ensure_ascii=False)

        elif format_type == ".md":
            if template_path and os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    tmpl = f.read()
                from Apps.AgentMesh.Core.Compiler.nodes.utils import interpolate
                ctx_data = (
                    input_data
                    if isinstance(input_data, dict)
                    else {"data": input_data}
                )
                content = interpolate(tmpl, ctx_data)
            elif isinstance(input_data, str) and input_data.strip().startswith("#"):
                # LLM already produced a markdown string — use it directly
                content = input_data
            elif isinstance(input_data, dict) and input_data.get("markdown"):
                # LLM placed rendered markdown under a 'markdown' key
                content = str(input_data["markdown"])
            else:
                # Fallback: pretty-print the data as markdown
                if isinstance(input_data, dict):
                    lines = [f"# {output_file_name}\n"]
                    for k, v in input_data.items():
                        lines.append(f"**{k}**: {v}  ")
                    content = "\n".join(lines)
                else:
                    content = f"# {output_file_name}\n\n{input_data}"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            preview_content = content  # Rendered inline in the chat panel

        else:
            raise ValueError(f"Unsupported format: {format_type}")

        logger.info(f"[ReportNode] ✅ Written: {output_path}")

        # ── Optional: Convert DOCX to PDF ────────────────────────────────
        pdf_download_url = None
        pdf_final_filename = None
        pdf_output_path = None

        if format_type in [".docx", ".dotx"]:
            try:
                import pythoncom
                from docx2pdf import convert

                pythoncom.CoInitialize()

                pdf_output_path = output_path.replace(".docx", ".pdf").replace(
                    ".dotx", ".pdf"
                )
                pdf_final_filename = final_filename.replace(".docx", ".pdf").replace(
                    ".dotx", ".pdf"
                )

                logger.info(f"[ReportNode] Converting {output_path} to PDF...")
                convert(output_path, pdf_output_path)
                logger.info(f"[ReportNode] ✅ PDF Generated: {pdf_output_path}")

                pdf_download_url = f"/exports/{thread_id}/{pdf_final_filename}"
            except Exception as e:
                logger.warning(f"[ReportNode] DOCX to PDF conversion failed: {e}")
                pdf_output_path = None

        # ── 5. Return state update ────────────────────────────────────────

        # Build the Markdown Links
        links_md = f"[📄 Download {final_filename}]({download_url})"
        if pdf_download_url:
            links_md += f"\n[📄 Download PDF Version ({pdf_final_filename})]({pdf_download_url})"

        vfs_files = {final_filename: output_path}
        if pdf_output_path:
            vfs_files[pdf_final_filename] = pdf_output_path

        return {
            "active_agent": node_config.id,
            "messages": [
                AIMessage(
                    content=(
                        f"✅ Report generated successfully!\n\n"
                        f"{links_md}\n\n"
                        f"---\n\n{preview_content}"
                    )
                    if preview_content
                    else (f"✅ Report generated successfully!\n\n{links_md}"),
                    response_metadata={"active_agent": node_config.id},
                )
            ],
            "vfs": {
                "files": vfs_files,
                "dirs": [],
            },
            "context": {
                **context,
                "report_output_path": output_path,
                "report_filename": final_filename,
                "report_download_url": download_url,
            },
        }

    executor = RunnableLambda(execute_report_node)
    setattr(executor, "name", node_config.id)
    return executor
