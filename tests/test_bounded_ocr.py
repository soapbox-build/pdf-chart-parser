"""OCR of scanned pages is bounded, limited to the selected pages, and disclosed as OCR.

Measured 2026-10-04: the image had no ocrmypdf, so a scan came back as page images with no text.
Upstream's add_text_layer, once ocrmypdf exists, OCRs EVERY image-only page of the file, selected
or not, with no limit. The PyMuPDF and ocrmypdf stacks are not installed in CI, so these tests
run against fakes; the real run (15 scanned pages, 12 OCR'd, 30.7 s) is in the PR body.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bounded_extract as be  # noqa: E402


class _Page:
    def __init__(self, text):
        self._text = text

    def get_text(self, mode):
        return self._text


class _Doc:
    def __init__(self, texts):
        self._pages = [_Page(t) for t in texts]
        self.page_count = len(texts)

    def __getitem__(self, i):
        return self._pages[i]


def _wire(monkeypatch, texts, ocr=None):
    doc = _Doc(texts)
    monkeypatch.setattr(be, "_open_pdf", lambda b: doc)
    calls = []

    def fake_ocr(pdf_bytes, pages):
        calls.append(pages)
        if ocr is not None:
            return ocr(pdf_bytes, pages)
        return b"OCRED"

    monkeypatch.setattr(be, "_run_ocr", fake_ocr)
    original = lambda b: (b"ORIGINAL-ALL-PAGES", True, "upstream ocr ran")  # noqa: E731
    return doc, calls, be.bound_add_text_layer(original), be.record_selected_pages(lambda d, idx=None: True)


SCAN = ""
TEXT = "a real text layer of more than sixteen characters"


def test_only_selected_scanned_pages_are_ocrd(monkeypatch):
    doc, calls, add, record = _wire(monkeypatch, [SCAN, TEXT, SCAN, SCAN])
    record(doc, [0, 1])  # the caller asked for pages 1 and 2
    out, applied, note = add(b"pdf")
    assert (out, applied) == (b"OCRED", True)
    assert calls == [[1]]  # page 3 and 4 are scans too, but were not selected, so not OCR'd
    assert "page(s) 1" in note and "recognised, not the document's own" in note


def test_a_text_page_alone_needs_no_ocr(monkeypatch):
    doc, calls, add, record = _wire(monkeypatch, [TEXT, SCAN])
    record(doc, [0])
    assert add(b"pdf") == (b"pdf", False, None)
    assert calls == []


def test_more_scans_than_the_limit_ocr_the_first_few_and_name_the_rest(monkeypatch):
    monkeypatch.setattr(be, "MAX_OCR_PAGES", 2)
    doc, calls, add, record = _wire(monkeypatch, [SCAN] * 5)
    record(doc)
    _, applied, note = add(b"pdf")
    assert applied and calls == [[1, 2]]
    assert "3 further scanned page(s) (3, 4, 5) were NOT OCR'd" in note
    assert "at most 2" in note


def test_an_ocr_failure_returns_the_original_bytes_and_says_why(monkeypatch):
    def boom(b, pages):
        raise RuntimeError("tesseract timed out")

    doc, _, add, record = _wire(monkeypatch, [SCAN], ocr=boom)
    record(doc)
    out, applied, note = add(b"pdf")
    assert (out, applied) == (b"pdf", False)
    assert "OCR failed or timed out" in note and "tesseract timed out" in note


def test_a_missing_ocrmypdf_is_reported_not_swallowed(monkeypatch):
    def missing(b, pages):
        raise ImportError("ocrmypdf")

    doc, _, add, record = _wire(monkeypatch, [SCAN], ocr=missing)
    record(doc)
    assert add(b"pdf")[2].startswith("scanned page(s) detected but OCRmyPDF is not installed")


def test_outside_an_extraction_upstream_behaviour_is_kept(monkeypatch):
    monkeypatch.setattr(be._call, "selected", None, raising=False)
    _, _, add, _ = _wire(monkeypatch, [SCAN])
    be._call.selected = None
    assert add(b"pdf")[0] == b"ORIGINAL-ALL-PAGES"


def _chunks(texts):
    return lambda doc, *a, pages=None, **k: [{"text": t} for t in texts]


def test_ocrd_pages_are_labelled_and_others_are_not():
    be._call.ocr_pages = {1}
    wrapped = be.disclose_ocr(_chunks(["native text, long enough here", "recognised text long enough"]))
    out = wrapped(_Doc(["x", "y"]), pages=[0, 1])
    assert out[0]["text"] == "native text, long enough here"
    assert out[1]["text"].startswith("[page 2: text recognised by OCR from a scan; it may contain errors]")
    assert out[1]["text"].endswith("recognised text long enough")


def test_the_label_never_turns_an_empty_ocr_page_into_a_text_page():
    # Upstream renders a page image when a page's text is under its threshold. A label on an
    # empty OCR result would lift it over that threshold and drop the image, so there is none.
    be._call.ocr_pages = {0}
    out = be.disclose_ocr(_chunks(["  ", ""]))(_Doc(["x", "y"]), pages=[0, 1])
    assert [c["text"] for c in out] == ["  ", ""]


def test_nothing_is_labelled_when_no_page_was_ocrd():
    be._call.ocr_pages = set()
    out = be.disclose_ocr(_chunks(["some long enough text here"]))(_Doc(["x"]), pages=[0])
    assert out == [{"text": "some long enough text here"}]
