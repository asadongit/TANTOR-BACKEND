import base64
import io
import json
import logging
import os
import shutil
from typing import Callable, Any, List, Dict

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from Apps.AgentMesh.Schema.WorkflowSchema import NodeConfig
from Apps.AgentMesh.Schema.StateSchema import AgentState
from Apps.AgentMesh.Utilities.LLM.llm_config import get_vision_llm
from Apps.AgentMesh.Core.Compiler.nodes.utils import resolve_temp_path

from Apps.AgentMesh.Config import settings
from Apps.AgentMesh.Core.Services.ArtifactService import ArtifactService
from Database.Database import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Resolution every page is rendered at before it reaches a vision model. Dense
# financial statements are unreadable at PyMuPDF's 72 dpi default.
_RENDER_DPI = 200


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

import re

def _strip_reasoning(text: str) -> str:
    """Remove a reasoning model's narration, keeping only the answer.

    A reasoning model writes its working inside <think>...</think> before it
    answers, and that working is not the document. When the tag closes, the
    answer is whatever follows it.

    When it never closes, the model spent its whole token budget thinking and
    no answer was ever written. There is nothing to salvage: returning the
    narration would hand the model's private working to the next node as if it
    were the customer's statement. Return nothing instead, so an empty page is
    visible as an empty page rather than passing for extracted text.
    """
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        return ""
    return re.sub(r"```(?:json|markdown|md|text)?", "", text,
                  flags=re.IGNORECASE).strip()


def _tidy_markdown_tables(text: str) -> str:
    """Make a transcribed table usable without discarding any of it.

    Vision models pad tables out with blank rows and lose column parity on
    long pages. Both confuse whatever reads the table next, so rows are padded
    to the header width and empty rows dropped.

    Rows are never truncated to the header width. A row wider than its header
    is normal on a statement that prints two accounts side by side, and
    trimming it would silently delete printed amounts.
    """
    out: List[str] = []
    width = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            if stripped:
                width = 0            # prose ends the table
            out.append(line)
            continue

        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not re.sub(r"[\s|\-:]", "", stripped):
            continue                 # a row carrying neither text nor digits
        if width and all(set(c) <= set("-: ") for c in cells):
            out.append(line)         # the header underline
            continue

        if not width:
            # An unlabelled first column leaves every description homeless.
            if cells and not cells[0]:
                cells[0] = "Particulars"
            width = len(cells)
        elif len(cells) < width:
            cells += [""] * (width - len(cells))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out).strip()


def _post_process_ocr(text: str) -> str:
    if not text or not isinstance(text, str):
        return text or ""
    return _tidy_markdown_tables(_strip_reasoning(text))


# Kept under the original name: other modules import it.
def _strip_think_tags(text: str) -> str:
    return _post_process_ocr(text)

# Tesseract needs this much detail to tell which way the letters face. At 200
# it mistook an upside-down page for an upright one; at 300 it was right on
# every page of the same document.
_OSD_DPI = 300
# An earlier version of this file refused to turn a page unless Tesseract
# reported at least 1.0 confidence. That threshold was set on a page judged by
# eye to be upright; it was in fact upside down, Tesseract had called it
# correctly at 0.62, and the gate threw the answer away. The page reached the
# model the wrong way up and gave back 4 figures out of 14.
#
# Measured since, on 43 pages: Tesseract's verdict was never wrong, only
# sometimes unsure. So the verdict is acted on whatever its confidence, and the
# work goes into checking the few pages where it is least sure.
#
# Below this, the answer is verified rather than taken on trust: the page is
# rendered both the way Tesseract suggests and a half-turn from it, Tesseract is
# asked to actually read each, and the one it reads more confidently wins.
# Upside-down text yields garbage words. On the 43 pages this changed no
# verdict, so it is insurance, not a correction - and at 1.0 it runs on about
# one page in forty rather than on all of them.
_OSD_VERIFY_BELOW = 1.0
# The verification runs at half the OSD resolution: it only has to tell words
# from gibberish.
_READBACK_SCALE = 0.5
# Below this many legible words there is not enough text to judge - a stamp, a
# signature sheet, a near-blank continuation page. Those are left as scanned.
_READBACK_MIN_WORDS = 12
# Where the engine installs on Windows; on Linux it is normally on PATH.
_TESSERACT_GUESSES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


_orient_warned = False


def _orient_unavailable(reason: str) -> None:
    """Say once, loudly, that straightening was asked for and cannot be done.

    Silence here would be the worst outcome: the setting is on, every page goes
    through unturned, and the output looks like an ordinary poor read. Warned
    once per process rather than once per page so a long document does not bury
    the message it is trying to deliver.
    """
    global _orient_warned
    if _orient_warned:
        return
    _orient_warned = True
    logger.warning(
        f"[OCR orient] Page straightening is switched on but unavailable: {reason}. "
        "Pages will be read exactly as they were scanned. Install the engine and "
        "its orientation data (on Debian: tesseract-ocr and tesseract-ocr-osd) "
        "to enable it."
    )


def _readback_confidence(image, pytesseract) -> tuple:
    """Try to read the page as it stands; report how well that went.

    Returns the mean per-word confidence and how many legible words carried it.
    Short fragments are ignored: punctuation and single characters score highly
    whichever way up the page is, and would drown the signal.
    """
    try:
        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT)
    except Exception:
        return 0.0, 0
    scores = [int(c) for c, word in zip(data["conf"], data["text"])
              if str(c).lstrip("-").isdigit() and int(c) > 0 and len(word.strip()) > 2]
    if not scores:
        return 0.0, 0
    return sum(scores) / len(scores), len(scores)


def _page_turn(png_bytes: bytes, label: str) -> int:
    """How far this page must be turned clockwise to bring its text upright.

    Returns 0 when the page is already upright, when the engine is unavailable,
    or when the page carries too little text to judge. A scan fed in the wrong
    way round carries nothing to say so - the file's own rotation flag stays at
    zero because the rotation is baked into the picture - so the only way to
    know is to look at the shape of the letters.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        _orient_unavailable("pytesseract is not installed")
        return 0

    if not shutil.which(pytesseract.pytesseract.tesseract_cmd or "tesseract"):
        found = next((p for p in _TESSERACT_GUESSES if os.path.exists(p)), None)
        if not found:
            _orient_unavailable("no tesseract engine on PATH")
            return 0
        pytesseract.pytesseract.tesseract_cmd = found

    try:
        page = Image.open(io.BytesIO(png_bytes))
        report = pytesseract.image_to_osd(page)
        turn = int(re.search(r"Rotate:\s*(\d+)", report).group(1)) % 360
        confidence = float(re.search(r"Orientation confidence:\s*([\d.]+)", report).group(1))
    except Exception as err:
        # Missing osd.traineddata is the usual cause and ships separately from
        # the engine on Debian, so say it once rather than per page.
        if "osd.traineddata" in str(err) or "Error opening data file" in str(err):
            _orient_unavailable(f"orientation data missing ({err})")
        else:
            logger.debug(f"[OCR orient] {label}: could not be judged ({err})")
        return 0

    if confidence >= _OSD_VERIFY_BELOW:
        if turn:
            logger.info(f"[OCR orient] {label}: turning {turn}° "
                        f"(confidence {confidence:.2f})")
        return turn

    # Least-sure band only: settle the half-turn by reading the page both ways.
    flipped = (turn + 180) % 360
    try:
        small = page.resize((max(1, int(page.width * _READBACK_SCALE)),
                             max(1, int(page.height * _READBACK_SCALE))))
        as_osd_said = small.rotate(-turn, expand=True) if turn else small
        the_other_way = small.rotate(-flipped, expand=True) if flipped else small
        score_osd, words_osd = _readback_confidence(as_osd_said, pytesseract)
        score_alt, words_alt = _readback_confidence(the_other_way, pytesseract)
    except Exception as err:
        logger.debug(f"[OCR orient] {label}: readback failed ({err}), "
                     f"falling back to the reported verdict")
        return turn

    if max(words_osd, words_alt) < _READBACK_MIN_WORDS:
        # A stamp, a signature page, a near-blank continuation sheet. Turning it
        # on this little evidence is worse than leaving it as it was scanned.
        logger.info(f"[OCR orient] {label}: too little text to judge "
                    f"({max(words_osd, words_alt)} words) — leaving it alone")
        return 0

    chosen = turn if score_osd >= score_alt else flipped
    if chosen != turn:
        logger.info(f"[OCR orient] {label}: reported {turn}° (confidence "
                    f"{confidence:.2f}) but reads better at {chosen}° "
                    f"({score_alt:.0f} against {score_osd:.0f}) — turning {chosen}°")
    elif chosen:
        logger.info(f"[OCR orient] {label}: turning {chosen}° "
                    f"(confidence {confidence:.2f}, readback "
                    f"{score_osd:.0f} against {score_alt:.0f})")
    return chosen


# A page laid out sideways carries roughly twice as many columns as an upright
# one, and those are the pages where figures get read into the wrong column or
# guessed outright. Sending a sharper picture does not help: the model's image
# budget is fixed - measured at about 2,500 tokens for a page whether it was
# rendered at 200, 300, 450 or 600 dpi - so a bigger picture is simply shrunk
# again on arrival. The only way to give it more detail per digit is to show it
# less paper at a time.
#
# Measured on the worst page in the sample, straightened: 7 figures of 14 read
# whole, 9 of 14 read in halves.
_WIDE_PAGE_RATIO = 1.15
# Never cut through the margins, and never cut a page into slivers.
_SPLIT_SEARCH_BAND = (0.35, 0.65)
# Resolution matters once - and only once - the page has been cut. A whole page
# is squeezed into the same fixed budget however finely it was rendered, so
# rendering it larger achieves nothing. A half page is a picture in its own
# right with its own budget, so the detail rendered into it survives. Measured
# on the two hardest pages, splitting from a 300 dpi render scored 8/14 and
# 24/39; from 500 dpi, 9/14 and 26/39.
_SPLIT_DPI = 500
# Keep the payload sane. Each half still carries far more detail than the whole
# page did, and an over-large image costs upload time for pixels the model
# discards anyway.
_SPLIT_MAX_EDGE = 3200


def _is_wide(png_bytes: bytes) -> bool:
    """Is this page markedly wider than it is tall, once the right way up."""
    try:
        from PIL import Image
        page = Image.open(io.BytesIO(png_bytes))
        return page.width > page.height * _WIDE_PAGE_RATIO
    except Exception:
        return False


def _split_wide_page(png_bytes: bytes, label: str) -> list:
    """Cut a wide page into two along the blank gutter between its two halves.

    Returns the page unchanged, as a single item, unless it is markedly wider
    than it is tall. The cut is found from the ink itself - the column of pixels
    carrying least - rather than assumed to be the middle, because a statement
    with three branches does not divide evenly.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        return [png_bytes]

    try:
        page = Image.open(io.BytesIO(png_bytes))
        if page.width <= page.height * _WIDE_PAGE_RATIO:
            return [png_bytes]

        ink = (np.array(page.convert("L")) < 200).sum(axis=0).astype(float)
        # a scan is never perfectly blank anywhere, so smooth away the speckle
        # and look for the quietest band rather than an empty one
        window = max(3, page.width // 200)
        smooth = np.convolve(ink, np.ones(window) / window, mode="same")
        lo = int(page.width * _SPLIT_SEARCH_BAND[0])
        hi = int(page.width * _SPLIT_SEARCH_BAND[1])
        if hi <= lo:
            return [png_bytes]
        cut = int(np.argmin(smooth[lo:hi])) + lo

        # The quietest band is between two AMOUNT columns, which on a sheet
        # printing two accounts side by side falls INSIDE the second account -
        # after its descriptions and before its figures. The right-hand half
        # then arrives as columns of bare numbers, and no instruction can make
        # a model label figures whose descriptions were never in the picture.
        #
        # So the right-hand half begins where the second account begins, not at
        # the cut. The two halves overlap by that much; nothing is lost at the
        # seam, and each half is still far smaller than the whole page, which is
        # what splitting was for. Measured on the worst page: 26 figures of 39
        # cutting at the gap, 30 of 39 overlapping, and identical across three
        # runs of each.
        start_of_second = _second_account_starts(page, label)
        right_edge = cut
        if start_of_second is not None and start_of_second < cut:
            right_edge = start_of_second

        halves = []
        for box in ((0, 0, cut, page.height),
                    (right_edge, 0, page.width, page.height)):
            half = page.crop(box)
            if max(half.size) > _SPLIT_MAX_EDGE:
                scale = _SPLIT_MAX_EDGE / max(half.size)
                half = half.resize((max(1, int(half.width * scale)),
                                    max(1, int(half.height * scale))))
            buf = io.BytesIO()
            half.save(buf, format="PNG")
            halves.append(buf.getvalue())
        if right_edge != cut:
            logger.info(f"[OCR split] {label}: wide page ({page.width}x{page.height}), "
                        f"left half ends at x={cut}, right half starts at "
                        f"x={right_edge} where the second account's descriptions "
                        f"begin - the halves overlap by {cut - right_edge}px so "
                        f"every figure keeps its label")
        else:
            logger.info(f"[OCR split] {label}: wide page ({page.width}x{page.height}), "
                        f"reading it in two halves either side of x={cut} "
                        f"(no second description column found to overlap)")
        return halves
    except Exception as err:
        logger.warning(f"[OCR split] {label}: could not be divided ({err}); "
                       f"reading it whole")
        return [png_bytes]


def _second_account_starts(page, label: str):
    """Where the second account's DESCRIPTION column begins, or None.

    A sheet printing two accounts side by side gives each one a description
    column and several amount columns. Descriptions are words and amounts are
    figures, so the description column is much the widest - and the second
    account starts where the second wide column starts. That is readable from
    the spacing of the printed rules alone, with nothing recognised and no
    model called.

    The rules are found with up to six degrees of tilt allowed. These scans lie
    askew - measured at 2.7 degrees on one page - and a strictly vertical search
    finds only whichever rule happens to be straight, which on one page was the
    wrong one entirely.
    """
    try:
        import numpy as np
        import cv2
    except ImportError:
        return None
    try:
        grey = np.array(page.convert("L"))
        height, width = grey.shape
        ink = cv2.threshold(grey, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        lines = cv2.HoughLinesP(ink, 1, np.pi / 720, threshold=200,
                                minLineLength=int(height * 0.30), maxLineGap=40)
        if lines is None:
            return None
        rules = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if y2 == y1:
                continue
            if abs(np.degrees(np.arctan2(x2 - x1, abs(y2 - y1)))) <= 6:
                rules.append(((x1 + x2) / 2) / width)
        rules.sort()
        kept = []
        for x in rules:
            if kept and x - kept[-1] < 0.012:
                continue                     # the same rule found twice
            kept.append(x)
        kept = [x for x in kept if 0.20 <= x <= 0.80]
        if len(kept) < 2:
            return None
        widest = max((kept[i + 1] - kept[i], kept[i]) for i in range(len(kept) - 1))
        return int(widest[1] * width)
    except Exception as err:
        logger.info(f"[OCR split] {label}: could not read the printed rules ({err})")
        return None


def _table_blocks(lines: List[str]) -> list:
    """The runs of table rows in a page, each with where it starts and ends.

    A table is a stretch of consecutive rows beginning with a pipe. Anything
    else - a heading, a signature block - ends it. Divider rows are skipped
    when reading a table's contents but stay in the text.
    """
    blocks, start, rows = [], None, []
    for i, raw in enumerate(lines + [""]):
        s = raw.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if start is None:
                start = i
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif start is not None:
            blocks.append({"from": start, "to": i, "rows": rows})
            start, rows = None, []
    return [b for b in blocks if b["rows"]]


def _describes_same_lines(a: dict, b: dict) -> bool:
    """Do two tables describe the same lines?

    Judged on the descriptions they have in common, not on their order or their
    length. The two pieces of an overlapping page transcribe the same rows with
    different numbers of columns, so one is usually shorter - but the wording
    down the first column is the same, and that is what identifies them.
    """
    def names(block):
        return {re.sub(r"[^a-z0-9]", "", (r[0] or "").lower())
                for r in block["rows"] if r and r[0] and re.search(r"[A-Za-z]", r[0])}
    x, y = names(a), names(b)
    if len(x) < 2 or len(y) < 2:
        return False
    return len(x & y) >= min(len(x), len(y)) * 0.6


def _figures_in(block: dict) -> int:
    return sum(1 for r in block["rows"] for c in r[1:] if _FIGURE.search(c))


def _disagreements(a: dict, b: dict) -> int:
    """Cells where both copies carry a figure and the two differ.

    The overlapping strip is transcribed twice, independently, so this is a
    second reading of the part of the page most likely to be misread - free,
    and worth reporting even though the fuller copy is the one kept.
    """
    def by_name(block):
        out = {}
        for r in block["rows"]:
            if r and r[0]:
                out.setdefault(re.sub(r"[^a-z0-9]", "", r[0].lower()), r)
        return out
    left, right, clashes = by_name(a), by_name(b), 0
    for name, ra in left.items():
        rb = right.get(name)
        if not rb:
            continue
        for c in range(1, min(len(ra), len(rb))):
            u, v = ra[c].strip(), rb[c].strip()
            if _FIGURE.search(u) and _FIGURE.search(v) and \
                    u.replace(",", "") != v.replace(",", ""):
                clashes += 1
    return clashes


def _join_pieces(texts: List[str], label: str) -> tuple:
    """Join the readings of one page's pieces, without repeating the overlap.

    A wide page is read in two pieces that deliberately overlap, so that the
    second account's descriptions reach both. The cost is that its rows are
    transcribed twice - once from each piece - usually with only the first
    amount column filled in one of them. Left in, whatever reads this next has
    to choose between them, and choosing the shorter copy silently discards the
    columns the overlap existed to recover.

    So where two tables describe the same lines, the one carrying more figures
    is kept and the other's rows are dropped. The fuller copy is not a better
    guess - it is the same rows with more of their columns, so nothing is lost:
    measured on three pages against readings taken by hand, keeping the fuller
    copy scored exactly what leaving both in scored.

    Everything that is not a table - headings, signature blocks - is left where
    it was. Nothing is interleaved and no row is rebuilt from two sources, so
    unlike merging by row position there is nothing here that can misalign.
    """
    if len(texts) < 2:
        return (texts[0] if texts else ""), 0
    lines = "\n".join(texts).splitlines()
    blocks = _table_blocks(lines)
    drop, clashes = set(), 0
    for i in range(len(blocks)):
        for j in range(i + 1, len(blocks)):
            if i in drop or j in drop:
                continue
            if not _describes_same_lines(blocks[i], blocks[j]):
                continue
            clashes += _disagreements(blocks[i], blocks[j])
            poorer = i if _figures_in(blocks[i]) < _figures_in(blocks[j]) else j
            drop.add(poorer)
    if not drop:
        return "\n".join(lines), clashes
    cut = set()
    for k in drop:
        cut.update(range(blocks[k]["from"], blocks[k]["to"]))
    kept = [ln for n, ln in enumerate(lines) if n not in cut]
    logger.info(
        f"[OCR split] {label}: the two pieces transcribed {len(drop)} table(s) "
        f"twice; kept the copy carrying more figures"
        + (f", and the two disagreed on {clashes} cell(s)" if clashes else "")
    )
    return "\n".join(kept), clashes


def _turned(png_bytes: bytes, turn: int, label: str) -> bytes:
    """Apply a turn already decided on. Separate from deciding it, so the same
    verdict can be applied to a second, finer rendering of the same page without
    paying for the decision twice."""
    if not turn:
        return png_bytes
    try:
        from PIL import Image
        out = io.BytesIO()
        # expand keeps the whole page when the turn is a quarter rather than a half
        Image.open(io.BytesIO(png_bytes)).rotate(-turn, expand=True).save(out, format="PNG")
        return out.getvalue()
    except Exception as err:
        logger.warning(f"[OCR orient] {label}: could not turn the page ({err})")
        return png_bytes


_SKEW_LIMIT = 0.30     # degrees; below this a page is level enough to leave alone
_SKEW_MAX = 8.0        # degrees; beyond this it is a turn, not a tilt
_SKEW_STEP = 0.25      # degrees; finer than a quarter degree changes nothing


def _page_skew(png_bytes: bytes, label: str) -> float:
    """How far this page is tilted, in degrees, positive clockwise.

    Orientation (`_page_turn`) answers "which way up"; this answers "how level".
    They are different faults with different consequences. A page a couple of
    degrees off reads perfectly well word by word, so nothing downstream looks
    wrong - but on a statement the description column and the amount column
    drift apart as the eye travels down the page, and each amount ends up
    printed beside the WRONG description. The transcription is then faithful to
    a page that lies.

    Measured on a real customer page: 2.7 degrees of tilt was enough to shift
    every amount in a 24-line expense schedule down by one row.

    The angle is found by trying each candidate tilt and keeping the one that
    makes the rows of the page sharpest. When a page is level every text line
    falls entirely within a few pixel rows, so the count of ink per row rises
    and falls steeply; when it is tilted the same line is smeared across many
    rows and those steps flatten out. Summing the squared difference between
    neighbouring rows measures that sharpness in one number.

    Finding text lines by joining up words was tried first and does not work
    here: statements are printed inside a ruled box, so the border links every
    line into a single shape and there is nothing left to measure.
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image
    except ImportError:
        return 0.0
    try:
        grey = np.array(Image.open(io.BytesIO(png_bytes)).convert("L"))
        # a third of the size is ample to find the angle and keeps the search
        # cheap; the correction is applied to the full-resolution page
        small = cv2.resize(grey, (0, 0), fx=0.35, fy=0.35,
                           interpolation=cv2.INTER_AREA)
        ink = cv2.threshold(small, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if int((ink > 0).sum()) < ink.size * 0.005:      # an almost blank page
            return 0.0
        h, w = ink.shape
        centre = (w / 2, h / 2)
        best_angle, best_score = 0.0, -1.0
        angle = -_SKEW_MAX
        while angle <= _SKEW_MAX + 1e-9:
            rot = cv2.warpAffine(
                ink, cv2.getRotationMatrix2D(centre, angle, 1.0), (w, h),
                flags=cv2.INTER_NEAREST, borderValue=0)
            profile = rot.sum(axis=1).astype("float64")
            score = float(((profile[1:] - profile[:-1]) ** 2).sum())
            if score > best_score:
                best_angle, best_score = float(angle), score
            angle += _SKEW_STEP
        return best_angle if abs(best_angle) >= _SKEW_LIMIT else 0.0
    except Exception as err:                            # noqa: BLE001
        logger.warning(f"[OCR deskew] {label}: could not measure tilt ({err})")
        return 0.0


def _levelled(png_bytes: bytes, angle: float, label: str) -> bytes:
    """Rotate a page flat by an angle already measured.

    The canvas is expanded and padded white so no printed figure is clipped at
    a corner - losing a digit would be a worse fault than the tilt.
    """
    if not angle:
        return png_bytes
    try:
        from PIL import Image
        out = io.BytesIO()
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        img.rotate(angle, expand=True, resample=Image.BICUBIC,
                   fillcolor=(255, 255, 255)).save(out, format="PNG")
        logger.info(f"[OCR deskew] {label}: page was {angle:+.2f}° off level; "
                    "straightened before reading")
        return out.getvalue()
    except Exception as err:                            # noqa: BLE001
        logger.warning(f"[OCR deskew] {label}: could not straighten ({err})")
        return png_bytes


def _upright(png_bytes: bytes, label: str, deskew: bool = True) -> bytes:
    """Return the page the right way up and level, or unchanged if it is both.

    Order matters: the page is turned upright first, because tilt can only be
    measured against text that is already the right way round.
    """
    turned = _turned(png_bytes, _page_turn(png_bytes, label), label)
    if not deskew:
        return turned
    return _levelled(turned, _page_skew(turned, label), label)


# ── Reading a page more than once ───────────────────────────────────────────
# A model asked the same question twice does not always give the same answer,
# and where it wavers it is usually wrong. Measured over three readings each of
# three pages: figures the readings agreed on were right 92% of the time, those
# they differed on only 59%. Page by page the split was cleaner still - the two
# pages that read well produced NO disagreement at all, the page that read
# badly produced 61.
#
# So a second reading is not used to pick a better answer. It is used to say
# which pages should not be trusted. Voting was tried and abandoned: when a
# figure is hard to read the wrong readings agree with each other, so a
# majority is not evidence. The first reading is always the one returned.
_FIGURE = re.compile(r"\d[\d,]{2,}(?:\.\d{1,2})?")


def _figures(text: str) -> set:
    """Every figure in a reading, thousands separators removed."""
    return {m.replace(",", "") for m in _FIGURE.findall(text or "")}


def _table_rows(text: str) -> int:
    return sum(1 for line in (text or "").splitlines() if line.strip().startswith("|"))


def _agreement(readings: List[str]) -> Dict[str, Any]:
    """How far several readings of one page agree about the figures on it.

    Reported as the share of figures common to every reading. One reading
    agrees with itself, so a single pass is always 1.0 - which keeps the
    default behaviour of the node unchanged and uncontroversial.
    """
    texts = [r for r in readings if r]
    if len(texts) < 2:
        return {"agreement": 1.0, "passes": len(texts), "figures": len(_figures(
            texts[0] if texts else "")), "agreed": None, "rows": None}
    sets = [_figures(t) for t in texts]
    common = set.intersection(*sets)
    union = set.union(*sets)
    rows = [_table_rows(t) for t in texts]
    return {
        "agreement": (len(common) / len(union)) if union else 1.0,
        "passes": len(texts),
        "figures": len(union),
        "agreed": len(common),
        # a dropped column changes the shape as well as the figures, and the
        # shape is worth reporting because it names the failure we actually saw
        "rows": {"min": min(rows), "max": max(rows)},
    }


def _record_page(report: Any, page_no: int, readings: List[str], source: str,
                 threshold: float, label: str = "",
                 overlap_disagreements: int = 0) -> None:
    """Note how a page read, into the report the node returns beside the text."""
    if report is None:
        return
    verdict = _agreement(readings)
    verdict["source"] = source
    verdict["stable"] = verdict["agreement"] >= threshold
    # A page read in overlapping pieces has part of itself transcribed twice,
    # which is a second reading of exactly the strip most likely to be misread.
    # Where those two transcriptions disagree the page is worth a look, however
    # steady the repeated whole-page readings were.
    if overlap_disagreements:
        verdict["overlap_disagreements"] = overlap_disagreements
        verdict["stable"] = False
    report.setdefault("pages", {})[str(page_no)] = verdict
    if not verdict["stable"]:
        # Say which of the two checks objected. A page read only once has
        # nothing to compare against, so quoting an agreement figure for it
        # reports None out of 54 at 100% - three contradictory numbers in one
        # breath, and no clue that the overlap was what actually complained.
        where = label or f"page {page_no}"
        if verdict.get("agreed") is None:
            reason = (f"the two overlapping halves disagreed about "
                      f"{overlap_disagreements} figure(s)")
        elif overlap_disagreements:
            reason = (f"{verdict['agreed']} of {verdict['figures']} figures "
                      f"survived all {verdict['passes']} readings "
                      f"({verdict['agreement']:.0%}), and the two overlapping "
                      f"halves disagreed about {overlap_disagreements}")
        else:
            reason = (f"only {verdict['agreed']} of {verdict['figures']} figures "
                      f"survived all {verdict['passes']} readings "
                      f"({verdict['agreement']:.0%})")
        logger.warning(f"[OCR confidence] {where}: {reason} — page marked unstable")


async def _perform_llm_ocr(llm: Any, file_path: str, prompt_text: str, file_name: str,
                           auto_orient: bool = False, passes: int = 1,
                           report: Any = None, page_no: int = 1,
                           threshold: float = 1.0,
                           max_pages: int = 20,
                           on_max: str = "truncate") -> str:
    """Internal helper for LLM-based vision OCR. Supports both images and PDFs.

    `max_pages` is the node's `ocr_max_pages` setting (0 = no limit). It used to
    be a constant 20 in here, which silently dropped everything past page 20 of
    a longer document.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # 1. Handle standard images
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        if auto_orient:
            with open(file_path, "rb") as fh:
                base64_image = base64.b64encode(
                    _upright(fh.read(), file_name)).decode("utf-8")
            mime_type = "image/png"
        else:
            base64_image = encode_image(file_path)
            mime_type = f"image/{ext[1:]}" if ext != ".jpg" else "image/jpeg"
        content = [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
        ]
        readings = []
        for _ in range(max(1, passes)):
            response = await llm.ainvoke([HumanMessage(content=content)])
            readings.append(_strip_think_tags(response.content))
        _record_page(report, page_no, readings, "vision", threshold, file_name)
        return readings[0]

    # 2. Handle PDFs (Iterative Page Processing for large documents)
    elif ext == ".pdf":
        try:
            import fitz # PyMuPDF
        except ImportError:
            return "[ERROR] PDF processing requires 'pymupdf'. Please run: pip install pymupdf"

        doc = fitz.open(file_path)
        full_text = []
        total_pages_actual = len(doc)

        # The page limit is a SETTING, not a constant. It was hard-coded at 20,
        # which silently discarded everything past page 20 of a longer document
        # - a 29-page pack whose accounts sat in the last nine pages produced
        # nothing at all, and said so only in a notice buried in the text.
        # `ocr_max_pages` already existed for the hybrid path; the vision path
        # now honours it too, so one setting governs both. 0 means no limit,
        # and the default stays 20 so nothing changes for anyone who has not
        # set it.
        MAX_PAGES = int(max_pages or 0) or total_pages_actual
        total_pages = min(total_pages_actual, MAX_PAGES)

        if total_pages_actual > MAX_PAGES:
            if (on_max or "truncate").lower() == "fail":
                raise ValueError(
                    f"OCR: {file_name} has {total_pages_actual} pages, over the "
                    f"ocr_max_pages limit of {MAX_PAGES} "
                    f"(ocr_on_max_pages='fail')."
                )
            logger.warning(
                f"OCR Node: PDF {file_name} has {total_pages_actual} pages. "
                f"Reading the first {MAX_PAGES} (ocr_max_pages={MAX_PAGES}). "
                f"Raise ocr_max_pages on the node to read more, or set it to 0 "
                f"for no limit."
            )
            full_text.append(
                f"⚠️ [SYSTEM NOTICE] This document has {total_pages_actual} "
                f"pages and only the first {MAX_PAGES} were read, because the "
                f"node's 'ocr_max_pages' is set to {MAX_PAGES}. Anything later "
                f"in the document is MISSING from this text. Raise that setting "
                f"to read the whole document.\n---"
            )

        logger.info(f"Processing PDF ({file_name}): {total_pages}/{total_pages_actual} pages...")

        for i in range(total_pages):
            page = doc.load_page(i)
            # PyMuPDF renders at 72 dpi unless told otherwise, which leaves the
            # figures on a dense financial statement unreadable. The hybrid
            # path already renders at 200; match it here.
            label = f"{file_name} p{i+1}"
            pix = page.get_pixmap(dpi=_OSD_DPI if auto_orient else _RENDER_DPI)
            img_data = pix.tobytes("png")
            pieces = [img_data]
            if auto_orient:
                turn = _page_turn(img_data, label)
                img_data = _turned(img_data, turn, label)
                # Levelling comes after turning, and before the page is cut or
                # sent: a tilted statement pairs each amount with the wrong
                # description, and no later stage can tell that it happened.
                img_data = _levelled(img_data, _page_skew(img_data, label), label)
                pieces = [img_data]
                # A wide page goes to the model in two halves, rendered afresh at
                # the finer resolution that only pays off once the page is cut.
                # Both halves belong to the same page and are joined back together
                # under one page heading, so nothing downstream need know.
                if _is_wide(img_data):
                    fine = _turned(page.get_pixmap(dpi=_SPLIT_DPI).tobytes("png"),
                                   turn, label)
                    pieces = _split_wide_page(fine, label)
            # A page cut in two is still one page: every piece is read on each
            # pass, and the pieces are joined before the passes are compared, so
            # the verdict is about the page rather than about half of it.
            readings = []
            overlap_clashes = 0
            for _ in range(max(1, passes)):
                read = []
                for part, piece in enumerate(pieces, 1):
                    where = "" if len(pieces) == 1 else f" (part {part} of {len(pieces)})"
                    base64_image = base64.b64encode(piece).decode("utf-8")
                    content = [
                        {"type": "text",
                         "text": f"OCR Page {i+1}/{total_pages}{where}: {prompt_text}"},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]
                    response = await llm.ainvoke([HumanMessage(content=content)])
                    read.append(_strip_think_tags(response.content))
                joined, clashes = _join_pieces(read, label)
                overlap_clashes = max(overlap_clashes, clashes)
                readings.append(joined)
            _record_page(report, i + 1, readings, "vision", threshold, label,
                         overlap_clashes)
            full_text.append(f"--- Page {i+1} ---\n" + readings[0])
            
            # Progress logging for large files
            if (i+1) % 5 == 0:
                logger.info(f"OCR Progress: {i+1}/{total_pages} pages completed.")

        doc.close()
        return "\n\n".join(full_text)

    return f"[ERROR] Unsupported file format: {ext}"

async def _perform_local_ocr(engine_type: str, file_path: str) -> str:
    """
    Local OCR engine execution. Handles both images and PDFs.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    # Placeholder logic for Tesseract/Chandra
    # For PDFs, local engines usually have built-in page loops.
    if engine_type == "tesseract":
        return f"[TESSERACT] Simulated OCR for {file_path}. (Install pytesseract to enable)"
    elif engine_type == "chandra":
        return f"[CHANDRA] Simulated OCR for {file_path}."
    elif engine_type == "surya":
        return f"[SURYA] Simulated OCR for {file_path}."
    
    return f"[LOCAL OCR: {engine_type}] Not implemented."


def _extract_pdf_text_layer(file_path: str, max_pages: int = 0) -> str:
    """Read a PDF's embedded text layer as one string. Returns "" if there is none.

    Kept for callers that want a quick whole-document read. The hybrid engine
    does NOT use this: a document can mix authored pages with scanned ones, and
    a single whole-document verdict silently discards whichever kind loses.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("[OCR hybrid] PyMuPDF unavailable — cannot read text layers")
        return ""

    try:
        with fitz.open(file_path) as doc:
            pages = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
            return "\n".join(doc[i].get_text() for i in range(pages)).strip()
    except Exception as e:
        logger.info(f"[OCR hybrid] Could not read text layer from {file_path}: {e}")
        return ""


def _classify_pdf_pages(file_path: str, min_chars: int, max_pages: int = 0):
    """Decide, page by page, whether each page can be read or must be OCR'd.

    Returns (pages, page_count) where `pages` is a list of
    (index, text_or_None) — None meaning "this page has no usable text layer".

    Per page, not per document. A document that mixes authored pages with
    scanned ones is common: of 39 documents in the reference corpus, 4 contained
    both, and 68 image-only pages sat inside otherwise-textual documents. A
    whole-document verdict on those returns the text layer and drops the scans
    entirely, with no error — the worst kind of failure, a confident wrong
    answer.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, 0

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.info(f"[OCR hybrid] Cannot open {file_path}: {e}")
        return None, 0

    try:
        limit = doc.page_count if max_pages <= 0 else min(doc.page_count, max_pages)
        pages = []
        for i in range(limit):
            text = doc[i].get_text().strip()
            pages.append((i, text if len(text) >= min_chars else None))
        return pages, doc.page_count
    finally:
        doc.close()


def _render_pdf_page(file_path: str, page_index: int, out_dir: str, dpi: int = _RENDER_DPI):
    """Render one PDF page to a PNG so a vision model can read it.

    A vision engine takes a file, not a page, so a page that needs OCR has to
    become an image first. Rendering only the pages that need it is what makes
    per-page handling affordable.
    """
    import fitz
    with fitz.open(file_path) as doc:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        out = os.path.join(out_dir, f"page_{page_index + 1:04d}.png")
        pix.save(out)
        return out


async def _perform_hybrid_ocr(node_config, llm, file_path: str, prompt_text: str,
                              file_name: str, standard_model: str,
                              auto_orient: bool = False, passes: int = 1,
                              report: Any = None, threshold: float = 1.0) -> str:
    """Read the text layer if the document has one; otherwise fall back.

    Most bank documents are digital PDFs whose text can simply be read. Sending
    those to a vision model is slow and costly for a worse result, because the
    embedded text is exact while OCR is a best guess. Only scans genuinely need
    a model.

    Non-PDFs (photographs, scans as images) have no text layer by definition and
    go straight to the fallback engine.
    """
    ext = os.path.splitext(file_path)[1].lower()
    min_chars = int(getattr(node_config, "ocr_text_layer_min_chars", None) or 100)
    max_pages = int(getattr(node_config, "ocr_max_pages", None) or 0)
    on_max = (getattr(node_config, "ocr_on_max_pages", None) or "truncate").lower()
    fallback = (getattr(node_config, "ocr_fallback_engine", None) or "vision").lower()

    if ext != ".pdf":
        # No text layer exists in an image by definition.
        logger.info(f"[OCR hybrid] {file_name}: not a PDF — using '{fallback}'")
        if fallback == "vision":
            return await _perform_llm_ocr(llm, file_path, prompt_text, file_name,
                                          auto_orient, passes, report, 1,
                                          threshold, max_pages or 20, on_max)
        return await _perform_local_ocr(standard_model, file_path)

    pages, page_count = _classify_pdf_pages(file_path, min_chars, max_pages)

    if pages is None:
        # Unreadable or PyMuPDF missing — fall back for the whole document,
        # which is exactly what would have happened without hybrid.
        logger.info(f"[OCR hybrid] {file_name}: cannot inspect pages — using '{fallback}'")
        if fallback == "vision":
            return await _perform_llm_ocr(llm, file_path, prompt_text, file_name,
                                          auto_orient, passes, report, 1,
                                          threshold, max_pages or 20, on_max)
        return await _perform_local_ocr(standard_model, file_path)

    if max_pages > 0 and page_count > max_pages:
        if on_max == "fail":
            raise ValueError(
                f"{file_name} has {page_count} pages, exceeding "
                f"ocr_max_pages={max_pages} (ocr_on_max_pages='fail')"
            )
        logger.warning(
            f"[OCR hybrid] {file_name}: {page_count} pages, processing the first "
            f"{max_pages} (ocr_on_max_pages='truncate')"
        )

    need_ocr = [i for i, text in pages if text is None]
    logger.info(
        f"[OCR hybrid] {file_name}: {len(pages)} page(s) — "
        f"{len(pages) - len(need_ocr)} from text layer, {len(need_ocr)} need '{fallback}'"
    )

    # A page read from the text layer is exact - the characters are the ones the
    # author typed, not a guess about ink - so it is recorded as certain rather
    # than left absent, which would otherwise read as "not assessed".
    for i, text in pages:
        if text is not None:
            _record_page(report, i + 1, [text], "text_layer", threshold, file_name)

    if not need_ocr:
        return "\n".join(t for _, t in pages).strip()

    # Render only the pages that need it, into a directory that is removed
    # afterwards regardless of outcome.
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix="ocr_hybrid_")
    ocr_text = {}
    try:
        for idx in need_ocr:
            try:
                # judging orientation needs more detail than reading does, and
                # the model's image budget saturates below either resolution
                png = _render_pdf_page(file_path, idx, tmp_dir,
                                       _OSD_DPI if auto_orient else _RENDER_DPI)
                page_label = f"{file_name} (page {idx + 1})"
                if fallback == "vision":
                    # the real page number travels with the call, so the report
                    # is keyed by the page the Bank would open, not by 1
                    ocr_text[idx] = await _perform_llm_ocr(
                        llm, png, prompt_text, page_label, auto_orient,
                        passes, report, idx + 1, threshold)
                else:
                    ocr_text[idx] = await _perform_local_ocr(standard_model, png)
            except Exception as page_err:
                # One unreadable page must not discard the pages that did work.
                logger.warning(f"[OCR hybrid] {file_name} page {idx + 1} failed: {page_err}")
                ocr_text[idx] = ""
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Reassemble in page order so the document still reads top to bottom.
    return "\n".join(
        (text if text is not None else ocr_text.get(i, "")) for i, text in pages
    ).strip()


def _get_nested(data: dict, path: str):
    """Get a value from a nested dict using dot notation (e.g., 'data.text')."""
    parts = path.split(".")
    for p in parts:
        if isinstance(data, dict):
            data = data.get(p)
        else:
            return None
    return data


async def _perform_external_ocr(
    file_path: str,
    filename: str,
    service_url: str,
    service_params: Dict[str, str],
    service_headers: Dict[str, str],
    response_type: str,
    file_field_name: str,
    text_path: str
) -> List[Dict[str, Any]]:
    """
    Send a file to an external OCR/extraction service via multipart/form-data.
    Supports SSE (Server-Sent Events) and standard JSON responses.

    Returns a list of per-page results:
        [{"filename": "...", "pagenumber": 1, "raw_text": "...", "file_path": "..."}]
    """
    results = []
    persistent_path = os.path.abspath(file_path)

    logger.info(
        f"[ExternalOCR] Starting upload → {service_url} | "
        f"file={filename} | response_type={response_type} | "
        f"params={list(service_params.keys())}"
    )

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0),
            verify=False,
            headers=service_headers
        ) as client:
            with open(file_path, "rb") as f:
                files = {file_field_name: (filename, f)}

                if response_type == "sse":
                    # ── SSE Streaming Response (DocBot pattern) ──
                    try:
                        from httpx_sse import aconnect_sse
                    except ImportError:
                        raise RuntimeError(
                            "SSE response type requires 'httpx-sse'. "
                            "Install it: pip install httpx-sse"
                        )

                    async with aconnect_sse(
                        client, "POST", service_url,
                        data=service_params, files=files
                    ) as event_source:
                        async for sse in event_source.aiter_sse():
                            try:
                                if not sse.data.strip():
                                    continue

                                event_data = json.loads(sse.data)

                                if event_data.get("status") == "page_extracted":
                                    page_info = event_data.get("data", {})
                                    raw_text = _get_nested(event_data, text_path) or page_info.get("text", "")
                                    page_num = page_info.get("page")

                                    if raw_text and len(raw_text.strip()) >= 20:
                                        results.append({
                                            "filename": filename,
                                            "pagenumber": page_num,
                                            "raw_text": raw_text,
                                            "img_base64": page_info.get("img_base64"),
                                            "file_path": persistent_path,
                                        })
                                        logger.info(
                                            f"[ExternalOCR] Page {page_num} extracted from "
                                            f"{filename} ({len(raw_text)} chars)"
                                        )
                                    else:
                                        logger.warning(
                                            f"[ExternalOCR] Page {page_num} of {filename} "
                                            f"returned blank/empty text"
                                        )

                                elif event_data.get("status") == "error":
                                    logger.error(
                                        f"[ExternalOCR] Platform returned error for {filename}: "
                                        f"{event_data.get('error')}"
                                    )

                            except json.JSONDecodeError:
                                logger.debug(f"[ExternalOCR] Skipping non-JSON SSE event: {sse.data}")
                                continue

                else:
                    # ── Standard JSON Response ──
                    response = await client.post(
                        service_url, data=service_params, files=files
                    )
                    response.raise_for_status()
                    response_data = response.json()

                    # Handle different JSON response shapes
                    if isinstance(response_data, list):
                        # List of page results
                        for idx, item in enumerate(response_data):
                            raw_text = _get_nested(item, text_path) if isinstance(item, dict) else str(item)
                            results.append({
                                "filename": filename,
                                "pagenumber": idx + 1,
                                "raw_text": raw_text or str(item),
                                "file_path": persistent_path,
                            })
                    elif isinstance(response_data, dict):
                        # Single result or nested pages
                        pages = response_data.get("pages") or response_data.get("results") or [response_data]
                        for idx, item in enumerate(pages if isinstance(pages, list) else [pages]):
                            raw_text = _get_nested(item, text_path) if isinstance(item, dict) else str(item)
                            results.append({
                                "filename": filename,
                                "pagenumber": idx + 1,
                                "raw_text": raw_text or str(item),
                                "file_path": persistent_path,
                            })

    except Exception as e:
        logger.error(
            f"[ExternalOCR] Failed to reach service at {service_url}: {e}"
        )
        raise

    if not results:
        logger.warning(f"[ExternalOCR] No text extracted from {filename}")

    # Ensure pages are in document order
    results.sort(key=lambda x: x.get("pagenumber", 0))
    return results


def build_ocr_node(node_config: NodeConfig) -> Callable:
    """
    Build an OCR node that supports multiple engines:
      - vision: Cloud LLM-based vision OCR
      - standard: Local OCR (Tesseract, Chandra, Surya)
      - external: External API service (DocBot, Google Vision, AWS Textract)
    """
    llm = None
    # 'hybrid' needs the vision model too: it is the fallback for any page whose
    # text layer is missing or too thin. Building it only for 'vision' would
    # leave hybrid without an engine at the moment it needs one.
    if getattr(node_config, "ocr_engine", "vision") in ("vision", "hybrid"):
        cfg = node_config.llm
        # Everything the user set on the node has to reach the model. Forwarding
        # only provider/model/temperature silently discarded the rest: the token
        # budget fell back to a default too small for a full page transcription,
        # a reasoning model could not be told to keep its working to itself, and
        # a per-node API key was ignored in favour of the global one.
        opts = {
            "provider": cfg.provider if cfg else None,
            "model": cfg.model if cfg else None,
            "temperature": cfg.temperature if cfg else 0.0,
        }
        for field, arg in (("max_tokens", "max_tokens"),
                           ("reasoning_effort", "reasoning_effort"),
                           ("reasoning_format", "reasoning_format"),
                           ("api_key_override", "api_key"),
                           ("base_url", "base_url")):
            value = getattr(cfg, field, None) if cfg else None
            if value:
                opts[arg] = value
        llm = get_vision_llm(**opts)

    async def execute_ocr(state: AgentState):
        context = state.get("context", {})
        thread_id = context.get("thread_id", "global")
        engine_type = getattr(node_config, "ocr_engine", "vision")

        destination_key = getattr(node_config, "destination_key", "latest_ocr_output") or "latest_ocr_output"

        # ── EXTERNAL SERVICE ENGINE ──
        if engine_type == "external":
            service_url = node_config.ocr_service_url
            if not service_url:
                error_msg = "OCR Node Error: External service URL is not configured."
                return {
                    "messages": [AIMessage(content=error_msg)],
                    "active_agent": node_config.id,
                    "error": {"message": error_msg, "type": "ConfigError"},
                    "context": {destination_key: None}
                }

            service_params = dict(node_config.ocr_service_params or {})
            service_headers = dict(node_config.ocr_service_headers or {})
            response_type = node_config.ocr_service_response_type or "sse"
            file_field_name = node_config.ocr_service_file_field or "files"
            text_path = node_config.ocr_service_text_path or "data.text"
            multi_file_key = node_config.ocr_multi_file_key
            
            # Smart auto-detection for webhook payloads
            if not multi_file_key and context.get("payload", {}).get("uploaded_files"):
                multi_file_key = "payload.uploaded_files"

            all_results = []

            try:
                if multi_file_key:
                    # ── Multi-file mode: iterate over uploaded files ──
                    from Apps.AgentMesh.Core.Compiler.nodes.utils import get_nested_val
                    file_list = get_nested_val(context, multi_file_key)
                    if not file_list or not isinstance(file_list, list):
                        error_msg = f"OCR Node Error: No files found at context key '{multi_file_key}'"
                        return {
                            "messages": [AIMessage(content=error_msg)],
                            "active_agent": node_config.id,
                            "error": {"message": error_msg, "type": "NoFilesFound"},
                            "context": {destination_key: None}
                        }

                    for file_info in file_list:
                        fpath = file_info.get("path") if isinstance(file_info, dict) else str(file_info)
                        fname = file_info.get("filename", os.path.basename(fpath)) if isinstance(file_info, dict) else os.path.basename(fpath)

                        if not fpath or not os.path.exists(fpath):
                            logger.warning(f"[ExternalOCR] Skipping missing file: {fpath}")
                            continue

                        results = await _perform_external_ocr(
                            file_path=fpath,
                            filename=fname,
                            service_url=service_url,
                            service_params=service_params,
                            service_headers=service_headers,
                            response_type=response_type,
                            file_field_name=file_field_name,
                            text_path=text_path
                        )
                        all_results.extend(results)
                else:
                    # ── Single file mode ──
                    input_key = node_config.ocr_input_key or "latest_file_path"
                    file_path = context.get(input_key) or context.get("latest_file_path")

                    # Auto-detect from webhook payload
                    if not file_path and context.get("payload", {}).get("uploaded_files"):
                        file_path = context["payload"]["uploaded_files"][0].get("path")

                    if not file_path or not os.path.exists(file_path):
                        error_msg = f"OCR Node Error: No file found at '{file_path}'"
                        return {
                            "messages": [AIMessage(content=error_msg)],
                            "active_agent": node_config.id,
                            "error": {"message": error_msg, "type": "FileNotFound"},
                            "context": {destination_key: None}
                        }

                    fname = os.path.basename(file_path)
                    all_results = await _perform_external_ocr(
                        file_path=file_path,
                        filename=fname,
                        service_url=service_url,
                        service_params=service_params,
                        service_headers=service_headers,
                        response_type=response_type,
                        file_field_name=file_field_name,
                        text_path=text_path
                    )

                total_pages = len(all_results)
                total_files = len(set(r["filename"] for r in all_results)) if all_results else 0
                summary = f"External OCR completed: {total_pages} pages from {total_files} file(s)."


                return {
                    "intent": "done",
                    "messages": [AIMessage(content=summary)],
                    "active_agent": node_config.id,
                    "context": {
                        f"ocr_output_{node_config.id}": all_results,
                        destination_key: all_results,
                        "ocr_page_count": total_pages,
                        "ocr_file_count": total_files,
                        "cd_customer_id": context.get("cd_customer_id")
                    }
                }

            except Exception as e:
                error_msg = f"External OCR failed: {str(e)}"
                logger.error(f"OCR Node {node_config.id}: {error_msg}")
                return {
                    "intent": "error",
                    "messages": [AIMessage(content=error_msg)],
                    "active_agent": node_config.id,
                    "error": {"message": error_msg, "type": type(e).__name__},
                    "context": {destination_key: None}
                }

        # ── VISION / STANDARD ENGINES (original logic) ──
        input_key = node_config.ocr_input_key or "latest_file_path"
        
        def _extract_path_or_data(val):
            if isinstance(val, dict):
                artifact_id = val.get("artifact_id")

                if artifact_id:
                    return f"artifact://{artifact_id}"

                return (
                    val.get("vfs_path")
                    or val.get("file_path")
                    or val.get("path")
                    or val.get("url")
                )

            return val
        file_path = _extract_path_or_data(context.get(input_key))
        
        # Dot-notation lookup (e.g. 'nodes.form_node_6vl8c.aadhar_id.url' or 'aadhar_doc.file_path')
        if not file_path and "." in input_key:
            curr = context
            for part in input_key.split("."):
                if isinstance(curr, dict):
                    curr = curr.get(part)
                else:
                    curr = None
                    break
            file_path = _extract_path_or_data(curr)

        # Check in user_input
        if not file_path and isinstance(context.get("user_input"), dict):
            file_path = _extract_path_or_data(context["user_input"].get(input_key))
            if not file_path and "." in input_key:
                curr = context["user_input"]
                for part in input_key.split("."):
                    if isinstance(curr, dict):
                        curr = curr.get(part)
                    else:
                        curr = None
                        break
                file_path = _extract_path_or_data(curr)

        if not file_path:
            file_path = context.get("latest_file_path")

        # Auto-detect from any dictionary key in context if file_path is still None
        if not file_path:
            for k, v in context.items():
                if isinstance(v, dict) and (v.get("vfs_path") or v.get("file_path") or v.get("path") or v.get("url")):
                    file_path = v.get("vfs_path") or v.get("file_path") or v.get("path") or v.get("url")
                    break
            
        # Auto-detect from item if available
        if not file_path and isinstance(context.get("item"), dict) and (context["item"].get("vfs_path") or context["item"].get("file_path")):
            file_path = context["item"].get("vfs_path") or context["item"].get("file_path")

        # Auto-detect from webhook payload
        if not file_path and context.get("payload", {}).get("uploaded_files"):
            file_path = context["payload"]["uploaded_files"][0].get("vfs_path") or context["payload"]["uploaded_files"][0].get("path")
        
        # Keep the original lightweight reference for LangGraph state.
        checkpoint_file_reference = file_path
        # Convert an artifact reference into its actual local/NFS path.
        if (
            isinstance(file_path, str)
            and file_path.startswith("artifact://")
        ):
            artifact_id = file_path.removeprefix("artifact://")

            checkpoint_file_reference = {
                "artifact_id": artifact_id
            }

            try:
                async with AsyncSessionLocal() as db:
                    resolved_artifact_path = (
                        await ArtifactService.resolve_path(
                            artifact_id=artifact_id,
                            db=db,
                        )
                    )

                file_path = str(resolved_artifact_path)

            except (ValueError, FileNotFoundError) as error:
                error_msg = (
                    f"OCR Node Error: Could not resolve "
                    f"artifact '{artifact_id}': {error}"
                )

                return {
                    "messages": [
                        AIMessage(
                            content=error_msg,
                            response_metadata={
                                "active_agent": node_config.id
                            },
                        )
                    ],
                    "active_agent": node_config.id,
                    "error": {
                        "message": error_msg,
                        "type": type(error).__name__,
                    },
                    "context": {
                        destination_key: None
                    },
                }

        # Safely resolve /temp/ URLs to physical disk paths (path-traversal protected)
        if file_path and isinstance(file_path, str) and not os.path.exists(file_path) and not file_path.startswith("data:"):
            resolved = resolve_temp_path(file_path)
            if resolved:
                file_path = resolved

        if not file_path or not isinstance(file_path, str) or (not os.path.exists(file_path) and not file_path.startswith("data:")):
            error_msg = f"OCR Node Error: No file found at '{file_path}'."
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": "FileNotFound"},
                "context": {destination_key: None}
            }

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        allowed_types = getattr(node_config, "allowed_file_types", "all") or "all"

        if allowed_types == "images" and ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            error_msg = f"OCR Node Error: File format '{ext}' is not allowed. This node only accepts image files (.jpg, .png, .webp)."
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": "InvalidFormatError"},
                "context": {destination_key: None}
            }
        elif allowed_types == "pdf" and ext != ".pdf":
            error_msg = f"OCR Node Error: File format '{ext}' is not allowed. This node only accepts PDF files (.pdf)."
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": "InvalidFormatError"},
                "context": {destination_key: None}
            }

        standard_model = getattr(node_config, "ocr_standard_model", "tesseract")
        prompt_text = node_config.ocr_prompt or "Please perform OCR on this document."
        auto_orient = bool(getattr(node_config, "ocr_auto_orient", False))
        # One pass is the default, so an existing workflow costs and behaves
        # exactly as it did. Two or more turns the node into its own checker.
        passes = max(1, int(getattr(node_config, "ocr_passes", None) or 1))
        threshold = float(getattr(node_config, "ocr_stability_threshold", None) or 1.0)
        report = {"passes": passes, "threshold": threshold, "pages": {}}

        logger.info(f"OCR Node {node_config.id}: Using engine '{engine_type}' ({standard_model if engine_type == 'standard' else ''}) for {file_name}")

        try:
            if engine_type == "hybrid":
                output_text = await _perform_hybrid_ocr(
                    node_config, llm, file_path, prompt_text, file_name, standard_model,
                    auto_orient, passes, report, threshold
                )
            elif engine_type == "vision":
                # the page limit is the node's setting on this path too, not a
                # constant buried in the helper
                output_text = await _perform_llm_ocr(
                    llm, file_path, prompt_text, file_name, auto_orient,
                    passes, report, 1, threshold,
                    int(getattr(node_config, "ocr_max_pages", 20) or 0),
                    getattr(node_config, "ocr_on_max_pages", None) or "truncate")
            else:
                output_text = await _perform_local_ocr(standard_model, file_path)

          
            # Which pages should not be trusted, named so that whatever reads
            # this next need not understand how it was decided. The summary is
            # written whether or not a second pass was asked for: one pass
            # reports every page certain, which is the honest default and keeps
            # the shape the same for anything downstream.
            unstable = sorted(
                (int(p) for p, v in report.get("pages", {}).items() if not v.get("stable")),
                key=int,
            )
            report["unstable_pages"] = unstable
            report["stable"] = not unstable
            report["file_name"] = file_name
            if unstable:
                logger.warning(
                    f"OCR Node {node_config.id}: {file_name} — "
                    f"{len(unstable)} page(s) read differently on a second pass: {unstable}"
                )

            return {
                "messages": [AIMessage(content=f"OCR ({engine_type}/{standard_model if engine_type == 'standard' else 'LLM'}) completed for {file_name}.", response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "ocr_file_name": file_name,
                "context": {
                    f"ocr_output_{node_config.id}": output_text,
                    destination_key: output_text,
                    f"{destination_key}_confidence": report,
                    "latest_ocr_confidence": report,
                    "latest_ocr_file_name": file_name,
                    "latest_file_path": checkpoint_file_reference,
                }
            }

        except Exception as e:
            raw_error = str(e)
            error_msg = f"OCR Node ({engine_type}) failed: {raw_error}"
            
            if engine_type == "vision" and ("content must be a string" in raw_error or "invalid_request_error" in raw_error):
                error_msg = f"❌ Vision Support Error: Model '{node_config.llm.model}' is not vision-capable."

            logger.error(f"OCR Error: {raw_error}")
            return {
                "messages": [AIMessage(content=error_msg, response_metadata={"active_agent": node_config.id})],
                "active_agent": node_config.id,
                "error": {"message": error_msg, "type": type(e).__name__},
                "context": {destination_key: None}
            }

    return RunnableLambda(execute_ocr).with_retry(stop_after_attempt=3)
