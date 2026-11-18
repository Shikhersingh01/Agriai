"""Soil Health Report Parser for AgriAI.

Extracts core soil chemistry parameters (N, P, K, pH, Moisture, Organic Content)
from uploaded PDF documents and image files (JPG, PNG) using pdfplumber,
pytesseract OCR (optional), and regular expression pattern matching.
"""

import io
import os
import re
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


def _extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, List[str]]:
    """Extract raw text from a PDF file using pdfplumber.

    Args:
        file_bytes: Raw binary content of the PDF file.

    Returns:
        Tuple of (extracted_text, list_of_notes).
    """
    notes: List[str] = []
    text_chunks: List[str] = []

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                notes.append("PDF file contains 0 pages.")
                return "", notes

            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text_chunks.append(page_text)

                # Also extract table text in case parameters are formatted inside tables
                try:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            clean_cells = [str(cell).strip() for cell in row if cell is not None]
                            if clean_cells:
                                text_chunks.append(" | ".join(clean_cells))
                except Exception as table_err:
                    logger.debug("Table extraction skipped on page %d: %s", page_idx, table_err)

            full_text = "\n".join(text_chunks).strip()
            if full_text:
                notes.append(f"Successfully extracted text from {page_count} PDF page(s) via pdfplumber.")
            else:
                notes.append(
                    "PDF opened successfully but contained no selectable text. "
                    "This document might be a scanned image PDF."
                )
            return full_text, notes

    except ImportError:
        # Fallback to pypdf if pdfplumber is not available
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_chunks.append(t)
            full_text = "\n".join(text_chunks).strip()
            notes.append("Extracted text using pypdf fallback (pdfplumber not installed).")
            return full_text, notes
        except Exception:
            notes.append(
                "pdfplumber is not installed. To parse PDF soil reports, please install pdfplumber (`pip install pdfplumber`)."
            )
            return "", notes

    except Exception as e:
        error_msg = f"Failed to extract text from PDF: {str(e)}"
        logger.error(error_msg)
        notes.append(error_msg)
        return "", notes


def _extract_text_from_image(file_bytes: bytes, filename: str) -> Tuple[str, List[str]]:
    """Extract raw text from an image file using pytesseract OCR with PIL fallback.

    Args:
        file_bytes: Raw binary content of the image.
        filename: Name of the uploaded file.

    Returns:
        Tuple of (extracted_text, list_of_notes).
    """
    notes: List[str] = []

    # 1. Load image using PIL to verify format and integrity
    image = None
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # Verify image is not corrupted
        notes.append(f"Image '{filename}' opened successfully ({image.format}, {image.size[0]}x{image.size[1]}px).")
    except ImportError:
        notes.append("Pillow (PIL) is not installed. Cannot process image files.")
        return "", notes
    except Exception as img_err:
        notes.append(f"Invalid or corrupted image file: {str(img_err)}")
        return "", notes

    # 2. Attempt OCR via pytesseract
    try:
        import pytesseract

        # Check for system Tesseract executable
        try:
            # Convert to RGB if needed
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB")

            extracted_text = pytesseract.image_to_string(image)
            clean_text = extracted_text.strip()
            if clean_text:
                notes.append("Successfully performed OCR on image using pytesseract.")
            else:
                notes.append("OCR completed but no readable text was detected in the image.")
            return clean_text, notes

        except getattr(pytesseract, "TesseractNotFoundError", FileNotFoundError):
            notes.append(
                "Tesseract OCR executable was not found on the system PATH. "
                "To enable image OCR for soil cards, please install Tesseract OCR "
                "(e.g., from https://github.com/UB-Mannheim/tesseract/wiki) or upload a PDF report."
            )
            return "", notes
        except Exception as ocr_err:
            notes.append(
                f"Tesseract OCR execution error: {str(ocr_err)}. "
                "Ensure Tesseract OCR is properly installed and configured."
            )
            return "", notes

    except ImportError:
        notes.append(
            "pytesseract Python package is not installed. "
            "To enable image OCR, run: `pip install pytesseract` and install Tesseract OCR on your system. "
            "You can also upload your report in PDF format."
        )
        return "", notes


def _extract_metric(
    raw_text: str,
    patterns: List[str],
    min_val: float = 0.0,
    max_val: float = 10000.0,
) -> Optional[float]:
    """Search for a numerical metric across multiple regex patterns.

    Args:
        raw_text: Text corpus to search within.
        patterns: List of regex pattern strings with exactly one capture group for the value.
        min_val: Minimum biologically/chemically realistic threshold.
        max_val: Maximum realistic threshold.

    Returns:
        Extracted float value, or None if no match is found within range.
    """
    for pattern in patterns:
        matches = re.finditer(pattern, raw_text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            val_str = match.group(1).strip()
            try:
                val = float(val_str)
                if min_val <= val <= max_val:
                    return round(val, 2)
            except ValueError:
                continue
    return None


def _extract_soil_metrics(raw_text: str) -> Tuple[Dict[str, Optional[float]], List[str]]:
    """Parse soil chemistry metrics from extracted text using domain regex patterns.

    Recognizes:
    - N (Nitrogen)
    - P (Phosphorus)
    - K (Potassium)
    - pH
    - Moisture
    - Organic_Content (Organic Carbon)
    """
    notes: List[str] = []
    extracted: Dict[str, Optional[float]] = {
        "N": None,
        "P": None,
        "K": None,
        "pH": None,
        "Moisture": None,
        "Organic_Content": None,
    }

    if not raw_text.strip():
        notes.append("No text available for parameter extraction.")
        return extracted, notes

    # 1. Nitrogen (N) patterns
    nitrogen_patterns = [
        # Explicit labels: "Available Nitrogen: 240 kg/ha", "Nitrogen (N): 45"
        r"(?:available\s+)?nitrogen(?:\s*\([nN]\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Explicit available N: "Available N: 280", "Avail N = 45"
        r"(?:avail(?:able)?\s+)?\bN\b[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "Nitrogen    45.0  kg/ha"
        r"(?:^|\n)\s*Nitrogen\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["N"] = _extract_metric(raw_text, nitrogen_patterns, min_val=0.0, max_val=2000.0)
    if extracted["N"] is not None:
        notes.append(f"Found Nitrogen (N): {extracted['N']}")
    else:
        notes.append("Nitrogen (N) not detected.")

    # 2. Phosphorus (P) patterns
    phosphorus_patterns = [
        # "Available Phosphorus (P): 18.5 kg/ha", "Phosphorus: 30"
        r"(?:available\s+)?phosphor(?:us|ic)?(?:\s*\([pP]\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # "P2O5: 22", "Available P: 30", "P = 30"
        r"(?:available\s+)?(?:P2O5|\bP\b)[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "Phosphorus    18.5  kg/ha"
        r"(?:^|\n)\s*Phosphorus\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["P"] = _extract_metric(raw_text, phosphorus_patterns, min_val=0.0, max_val=1000.0)
    if extracted["P"] is not None:
        notes.append(f"Found Phosphorus (P): {extracted['P']}")
    else:
        notes.append("Phosphorus (P) not detected.")

    # 3. Potassium (K) patterns
    potassium_patterns = [
        # "Available Potassium (K): 180 kg/ha", "Potassium: 200"
        r"(?:available\s+)?potassium(?:\s*\([kK]\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # "Potash (K2O): 160", "Potash: 180"
        r"(?:available\s+)?potash(?:\s*\([kK]2[oO]\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # "K2O = 180", "Available K: 150", "K: 200", "K = 200"
        r"(?:available\s+)?(?:K2O|\bK\b)[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "Potassium    160.0  kg/ha"
        r"(?:^|\n)\s*(?:Potassium|Potash)\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["K"] = _extract_metric(raw_text, potassium_patterns, min_val=0.0, max_val=3000.0)
    if extracted["K"] is not None:
        notes.append(f"Found Potassium (K): {extracted['K']}")
    else:
        notes.append("Potassium (K) not detected.")

    # 4. pH patterns
    ph_patterns = [
        # "Soil Reaction (pH): 7.2", "Reaction (pH) = 7.5"
        r"(?:soil\s+)?reaction\s*\(\s*pH\s*\)[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # "pH: 6.5", "Soil pH: 7.2", "pH (1:2.5): 8.1", "pH value: 6.8", "pH = 6.5"
        r"(?:soil\s+)?\bpH\b(?:\s*value)?(?:\s*\([^)]*\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "pH    7.2"
        r"(?:^|\n)\s*pH\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["pH"] = _extract_metric(raw_text, ph_patterns, min_val=3.0, max_val=11.0)
    if extracted["pH"] is not None:
        notes.append(f"Found pH: {extracted['pH']}")
    else:
        notes.append("pH not detected.")

    # 5. Organic Content / Organic Carbon patterns
    organic_patterns = [
        # "Organic Carbon: 0.54%", "Organic Carbon (OC): 0.65", "Organic Content: 0.5"
        r"organic\s+(?:carbon|content|matter)(?:\s*\([oO]\.?[cC]\.?\))?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # "OC: 0.45%", "O.C. = 0.55", "SOC: 0.6%"
        r"\b(?:SOC|[oO]\.?[cC]\.?)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "Organic Carbon    0.55  %"
        r"(?:^|\n)\s*Organic\s+Carbon\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["Organic_Content"] = _extract_metric(raw_text, organic_patterns, min_val=0.01, max_val=20.0)
    if extracted["Organic_Content"] is not None:
        notes.append(f"Found Organic Content (OC): {extracted['Organic_Content']}%")
    else:
        notes.append("Organic Content (OC) not detected.")

    # 6. Moisture patterns
    moisture_patterns = [
        # "Moisture: 18%", "Moisture content: 15.5%", "Soil Moisture: 14%"
        r"(?:soil\s+)?moisture(?:\s+content)?(?:\s*\([^)]*\))?(?:\s*%)?[\s:=]+([0-9]+(?:\.[0-9]+)?)",
        # Tabular line: "Moisture    15.0  %"
        r"(?:^|\n)\s*Moisture\b[^\d\n]*\b([0-9]+(?:\.[0-9]+)?)\b",
    ]
    extracted["Moisture"] = _extract_metric(raw_text, moisture_patterns, min_val=0.0, max_val=100.0)
    if extracted["Moisture"] is not None:
        notes.append(f"Found Moisture: {extracted['Moisture']}%")
    else:
        notes.append("Moisture not detected.")

    return extracted, notes


def parse_soil_report(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """Extract soil test values from uploaded PDF or image documents.

    Handles PDF documents via pdfplumber and images (JPG, PNG) via pytesseract OCR.
    Extracts values for Nitrogen (N), Phosphorus (P), Potassium (K), pH, Moisture,
    and Organic Content using regular expressions. Gracefully handles partial
    extractions and missing OCR dependencies.

    Args:
        file_bytes: Raw binary content of the uploaded report file.
        filename: Original filename (used to identify extension: .pdf, .jpg, .png, etc.).

    Returns:
        Dictionary containing:
        - 'extracted_values': Dict[str, Optional[float]] of N, P, K, pH, Moisture, Organic_Content
        - 'raw_text': String of extracted text from document
        - 'confidence': Float score from 0.0 to 1.0 (proportion of target metrics found)
        - 'parsing_notes': List of informative processing notes and warnings
    """
    total_metrics = 6.0
    all_notes: List[str] = []
    raw_text = ""

    if not file_bytes:
        return {
            "extracted_values": {
                "N": None,
                "P": None,
                "K": None,
                "pH": None,
                "Moisture": None,
                "Organic_Content": None,
            },
            "raw_text": "",
            "confidence": 0.0,
            "parsing_notes": ["Empty file uploaded. Please provide a valid soil test report."],
        }

    ext = os.path.splitext(filename.lower())[1]

    # 1. Text extraction by file type
    if ext == ".pdf":
        raw_text, notes = _extract_text_from_pdf(file_bytes)
        all_notes.extend(notes)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        raw_text, notes = _extract_text_from_image(file_bytes, filename)
        all_notes.extend(notes)
    elif ext in (".txt", ".csv"):
        try:
            raw_text = file_bytes.decode("utf-8")
            all_notes.append("Read plain text/CSV file directly.")
        except UnicodeDecodeError:
            raw_text = file_bytes.decode("latin-1", errors="ignore")
            all_notes.append("Read plain text file with latin-1 fallback encoding.")
    else:
        all_notes.append(
            f"Unsupported or unrecognized file extension '{ext}'. "
            "Supported formats: PDF (.pdf) and images (.jpg, .jpeg, .png, .bmp, .tiff, .webp)."
        )

    # 2. Parameter extraction from raw text
    extracted_values, metric_notes = _extract_soil_metrics(raw_text)
    all_notes.extend(metric_notes)

    # 3. Calculate extraction confidence score
    found_count = sum(1 for val in extracted_values.values() if val is not None)
    confidence = round(found_count / total_metrics, 2)

    all_notes.append(
        f"Extraction Summary: {found_count} of 6 key soil metrics extracted (Confidence: {confidence * 100:.0f}%)."
    )

    return {
        "extracted_values": extracted_values,
        "raw_text": raw_text,
        "confidence": confidence,
        "parsing_notes": all_notes,
    }
