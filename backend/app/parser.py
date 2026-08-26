from __future__ import annotations

import hashlib
import io
import json as _json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


# ============================================================================
# EXTENSIONS
# ============================================================================

_PLAIN_EXTS = {
    ".txt",
    ".csv",
    ".rst",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".env",
}

_CODE_EXTS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".cpp",
    ".c",
    ".rs",
    ".go",
    ".rb",
    ".php",
    ".swift",
    ".kt",
}


# ============================================================================
# DOCUMENT
# ============================================================================

@dataclass
class Document:
    file_path: str
    filename: str
    content: str
    file_type: str
    last_modified: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def document_id(self) -> str:
        """
        Stable hash berdasarkan absolute file path.
        """
        return hashlib.sha256(
            self.file_path.encode("utf-8")
        ).hexdigest()


# ============================================================================
# DOCUMENT PARSER
# ============================================================================

class DocumentParser:
    """
    Parser dokumen untuk Qvarn-RAG.

    Fokus PDF:
      1. Ekstraksi teks native menggunakan pdfplumber.
      2. Ekstraksi tabel native.
      3. Deteksi halaman yang membutuhkan OCR.
      4. OCR menggunakan PyMuPDF + Tesseract.
      5. Normalisasi hasil OCR.
      6. Normalisasi tabel agar informasi label dan angka
         tetap mudah ditemukan oleh sistem RAG.
    """

    # ----------------------------------------------------------------------
    # PUBLIC
    # ----------------------------------------------------------------------

    def parse(
            self,
            file_path: str | Path,
    ) -> Document:

        path = Path(file_path).resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Path bukan file: {path}"
            )

        ext = path.suffix.lower()

        last_modified = path.stat().st_mtime

        content = self._extract(
            path,
            ext,
        )

        return Document(
            file_path=str(path),
            filename=path.name,
            content=content,
            file_type=ext.lstrip("."),
            last_modified=last_modified,
            metadata={
                "file_size": path.stat().st_size,
                "extension": ext,
            },
        )

    # =========================================================================
    # DISPATCH
    # =========================================================================

    def _extract(
            self,
            path: Path,
            ext: str,
    ) -> str:

        named: Dict[str, Any] = {
            ".pdf": self._pdf,
            ".docx": self._docx,
            ".md": self._markdown,
            ".xlsx": self._xlsx,
            ".xls": self._xls,
            ".html": self._html,
            ".json": self._json,
            ".xml": self._xml,
            ".eml": self._eml,
            ".epub": self._epub,
        }

        if ext in named:
            return named[ext](path)

        if ext in _PLAIN_EXTS:
            return self._plain(path)

        if ext in _CODE_EXTS:
            return self._code(path)

        raise ValueError(
            f"Unsupported extension: {ext}"
        )

    # =========================================================================
    # PDF
    # =========================================================================

    def _pdf(
            self,
            path: Path,
    ) -> str:
        """
        Ekstraksi PDF dengan kombinasi:

        - teks native
        - tabel native
        - OCR

        OCR tidak langsung menggantikan teks native.
        Semua sumber dipertahankan supaya informasi tidak hilang.
        """

        try:
            import pdfplumber  # type: ignore
        except ImportError:
            logger.warning(
                "pdfplumber tidak tersedia. "
                "Menggunakan PyPDF2 untuk %s",
                path,
            )

            return self._pdf_pypdf2(path)

        pages: list[str] = []

        try:
            with pdfplumber.open(str(path)) as pdf:

                for page_number, page in enumerate(
                        pdf.pages,
                        start=1,
                ):

                    page_parts: list[str] = []

                    # ======================================================
                    # TEXT NATIVE
                    # ======================================================

                    text = ""

                    try:
                        text = page.extract_text(
                            x_tolerance=2,
                            y_tolerance=3,
                        ) or ""

                    except Exception as exc:
                        logger.warning(
                            "Gagal extract text halaman %s: %s",
                            page_number,
                            exc,
                        )

                    normalized_text = ""

                    if text.strip():
                        normalized_text = (
                            self._normalize_pdf_text(
                                text
                            )
                        )

                        if normalized_text.strip():

                            page_parts.append(
                                "[TEXT - Halaman "
                                f"{page_number}]\n"
                                f"{normalized_text.strip()}"
                            )

                    # ======================================================
                    # TABLE NATIVE
                    # ======================================================

                    try:

                        tables = page.extract_tables()

                        for table_index, table in enumerate(
                                tables,
                                start=1,
                        ):

                            if not table:
                                continue

                            table_text = (
                                self._format_pdf_table(
                                    table,
                                    page_number,
                                    table_index,
                                )
                            )

                            if table_text.strip():
                                page_parts.append(
                                    table_text
                                )

                    except Exception as exc:
                        logger.warning(
                            "Gagal extract tabel halaman %s: %s",
                            page_number,
                            exc,
                        )

                    # ======================================================
                    # OCR DECISION
                    # ======================================================

                    try:

                        image_count = len(
                            page.images
                        )

                    except Exception:
                        image_count = 0

                    should_ocr = (
                        self._should_ocr_page(
                            text=normalized_text,
                            image_count=image_count,
                            page=page,
                        )
                    )

                    # ======================================================
                    # OCR
                    # ======================================================

                    if should_ocr:

                        try:

                            ocr_text = (
                                self._ocr_pdf_page(
                                    path,
                                    page_number,
                                )
                            )

                            if ocr_text.strip():

                                page_parts.append(
                                    "[OCR - Halaman "
                                    f"{page_number}]\n"
                                    f"{ocr_text.strip()}"
                                )

                        except Exception as exc:
                            logger.warning(
                                "OCR gagal halaman %s pada %s: %s",
                                page_number,
                                path,
                                exc,
                            )

                    # ======================================================
                    # PAGE RESULT
                    # ======================================================

                    if page_parts:

                        page_text = (
                            "\n\n".join(
                                part.strip()
                                for part in page_parts
                                if part.strip()
                            )
                        )

                        if page_text:
                            pages.append(
                                page_text
                            )

        except Exception as exc:
            logger.exception(
                "Gagal membaca PDF %s: %s",
                path,
                exc,
            )

            return self._pdf_pypdf2(path)

        return "\n\n".join(
            pages
        )

    # =========================================================================
    # OCR DECISION
    # =========================================================================

    def _should_ocr_page(
            self,
            text: str,
            image_count: int,
            page: Any,
    ) -> bool:
        """
        Menentukan apakah halaman perlu OCR.

        OCR tetap dijalankan pada halaman yang:
          - tidak memiliki text native
          - memiliki text sangat sedikit
          - memiliki gambar
          - terlihat seperti halaman scan/tabel gambar

        Tetapi halaman teks normal tanpa gambar tidak perlu OCR.
        """

        clean_text = (
            text.strip()
            if text
            else ""
        )

        # Tidak ada text sama sekali
        if not clean_text:
            return True

        # Text sangat sedikit
        if len(clean_text) < 150:
            return True

        # Ada gambar
        if image_count > 0:
            return True

        # Beberapa PDF scanner menghasilkan text yang
        # sangat buruk/acak. Coba deteksi secara sederhana.
        alpha_count = sum(
            char.isalpha()
            for char in clean_text
        )

        digit_count = sum(
            char.isdigit()
            for char in clean_text
        )

        if len(clean_text) > 0:

            alpha_ratio = (
                    alpha_count /
                    len(clean_text)
            )

            digit_ratio = (
                    digit_count /
                    len(clean_text)
            )

            # Halaman yang didominasi karakter non-alfanumerik
            # kemungkinan hasil extraction buruk.
            if (
                    alpha_ratio < 0.20
                    and digit_ratio < 0.20
            ):
                return True

        return False

    # =========================================================================
    # OCR PAGE
    # =========================================================================

    def _ocr_pdf_page(
            self,
            path: Path,
            page_number: int,
    ) -> str:
        """
        Render PDF page dengan PyMuPDF kemudian OCR menggunakan Tesseract.

        Resolusi 2x digunakan agar angka dan tabel lebih mudah dibaca.
        """

        # ------------------------------------------------------------------
        # PyMuPDF
        # ------------------------------------------------------------------

        try:
            import pymupdf  # type: ignore

        except ImportError:

            try:
                import fitz as pymupdf  # type: ignore

            except ImportError:

                raise ImportError(
                    "PyMuPDF diperlukan untuk OCR PDF. "
                    "Install dengan: pip install pymupdf"
                )

        # ------------------------------------------------------------------
        # Tesseract
        # ------------------------------------------------------------------

        try:
            import pytesseract  # type: ignore

        except ImportError:

            raise ImportError(
                "pytesseract diperlukan untuk OCR PDF. "
                "Install dengan: pip install pytesseract"
            )

        # ------------------------------------------------------------------
        # Pillow
        # ------------------------------------------------------------------

        try:
            from PIL import Image  # type: ignore

        except ImportError:

            raise ImportError(
                "Pillow diperlukan untuk OCR PDF. "
                "Install dengan: pip install pillow"
            )

        # ------------------------------------------------------------------
        # Render
        # ------------------------------------------------------------------

        with pymupdf.open(str(path)) as pdf:

            if (
                    page_number < 1
                    or page_number > len(pdf)
            ):
                raise ValueError(
                    f"Nomor halaman PDF tidak valid: "
                    f"{page_number}"
                )

            page = pdf[
                page_number - 1
                ]

            # 2x resolution
            matrix = pymupdf.Matrix(
                2.0,
                2.0,
            )

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=False,
            )

            image_bytes = pix.tobytes(
                "png"
            )

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

        # ------------------------------------------------------------------
        # OCR
        # ------------------------------------------------------------------

        config = (
            "--oem 3 "
            "--psm 6"
        )

        try:

            text = pytesseract.image_to_string(
                image,
                lang="ind+eng",
                config=config,
            )

        except Exception as exc:

            logger.warning(
                "Tesseract ind+eng gagal: %s. "
                "Mencoba eng.",
                exc,
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng",
                config=config,
            )

        return self._normalize_ocr_text(
            text
        )

    # =========================================================================
    # OCR NORMALIZATION
    # =========================================================================

    def _normalize_ocr_text(
            self,
            text: str,
    ) -> str:
        """
        Membersihkan artefak OCR tanpa menghilangkan informasi angka.

        Fokus:

        - normalisasi whitespace
        - normalisasi Rp
        - memperbaiki angka yang terpisah
        - memperbaiki karakter OCR 0
        - mempertahankan baris
        - memperbaiki nama bulan yang rusak
        - memperbaiki pola tabel sederhana
        """

        if not text:
            return ""

        lines: list[str] = []

        for raw_line in text.splitlines():

            line = raw_line.strip()

            if not line:
                continue

            # --------------------------------------------------------------
            # 1. Whitespace
            # --------------------------------------------------------------

            line = re.sub(
                r"[ \t]+",
                " ",
                line,
            )

            # --------------------------------------------------------------
            # 2. Rupiah
            # --------------------------------------------------------------

            line = re.sub(
                r"\bRP\s*\.?\s*",
                "Rp ",
                line,
                flags=re.IGNORECASE,
            )

            # --------------------------------------------------------------
            # 3. OCR zero artifacts
            # --------------------------------------------------------------

            line = re.sub(
                r"(Rp\s*)"
                r"(?:A°|AÂ°|Ã‚Â°|Aº|Â°)",
                r"\g<1>0",
                line,
                flags=re.IGNORECASE,
            )

            # --------------------------------------------------------------
            # 4. Normalize currency numbers
            # --------------------------------------------------------------

            line = self._normalize_currency_numbers(
                line
            )

            # --------------------------------------------------------------
            # 5. Normalize common OCR month errors
            # --------------------------------------------------------------

            line = self._normalize_month_names(
                line
            )

            # --------------------------------------------------------------
            # 6. Normalize separators
            # --------------------------------------------------------------

            line = line.replace(
                "â€“",
                "-",
            )

            line = line.replace(
                "â€”",
                "-",
            )

            # OCR kadang menghasilkan vertical bar
            # yang menempel ke angka.
            line = re.sub(
                r"\s*\|\s*",
                " | ",
                line,
            )

            line = re.sub(
                r" {2,}",
                " ",
                line,
            )

            lines.append(
                line.strip()
            )

        # --------------------------------------------------------------
        # 7. Post-process tabel
        # --------------------------------------------------------------

        result = self._normalize_monthly_table(
            lines
        )

        return "\n".join(
            result
        )

    # =========================================================================
    # CURRENCY NORMALIZATION
    # =========================================================================

    def _normalize_currency_numbers(
            self,
            value: str,
    ) -> str:
        """
        Memperbaiki angka mata uang yang dipisahkan OCR.

        Contoh:

            Rp 2 8,400,934,740
            ->
            Rp 28,400,934,740

            Rp 2 ,840,093,474
            ->
            Rp 2,840,093,474

            Rp 54,829,980, 121
            ->
            Rp 54,829,980,121
        """

        def fix_currency(
                match: re.Match[str],
        ) -> str:

            prefix = match.group(1)
            number = match.group(2)

            # Digit + spasi + digit
            number = re.sub(
                r"(?<=\d)\s+(?=\d)",
                "",
                number,
            )

            # Digit + spasi + comma
            number = re.sub(
                r"(?<=\d)\s+(?=,)",
                "",
                number,
            )

            # Comma + spasi + digit
            number = re.sub(
                r"(?<=,)\s+(?=\d)",
                "",
                number,
            )

            return (
                    prefix
                    + number
            )

        value = re.sub(
            r"(Rp\s*)([\d\s,.\-()]+)",
            fix_currency,
            value,
        )

        # ------------------------------------------------------------------
        # Angka biasa yang terpisah OCR.
        #
        # Hanya diterapkan pada sequence yang jelas numerik.
        # ------------------------------------------------------------------

        def normalize_number(
                match: re.Match[str],
        ) -> str:

            number = match.group(0)

            number = re.sub(
                r"(?<=\d)\s+(?=\d)",
                "",
                number,
            )

            number = re.sub(
                r"(?<=\d)\s+(?=,)",
                "",
                number,
            )

            number = re.sub(
                r"(?<=,)\s+(?=\d)",
                "",
                number,
            )

            return number

        value = re.sub(
            r"\b\d[\d\s,.\-()]*\d\b",
            normalize_number,
            value,
        )

        return value

    # =========================================================================
    # MONTH NORMALIZATION
    # =========================================================================

    def _normalize_month_names(
            self,
            value: str,
    ) -> str:
        """
        Memperbaiki beberapa artefak OCR umum pada nama bulan.

        Tidak mencoba mengubah semua kemungkinan OCR secara agresif,
        karena perubahan terlalu agresif dapat merusak teks asli.
        """

        replacements = {
            "JANUAR": "JANUARI",
            "JANUARl": "JANUARI",
            "JANUAR|": "JANUARI",

            "FEBRUAR": "FEBRUARI",
            "FEBRUARl": "FEBRUARI",

            "MARET": "MARET",

            "APRIL": "APRIL",
            "APRlL": "APRIL",

            "MEI": "MEI",

            "JUNI": "JUNI",
            "JUNl": "JUNI",

            "JULI": "JULI",
            "JULl": "JULI",

            "AGUSTUS": "AGUSTUS",
            "AGUSTUS|": "AGUSTUS",

            "SEPTEMBER": "SEPTEMBER",
            "SEPTEM8ER": "SEPTEMBER",

            "OKTOBER": "OKTOBER",
            "0KTOBER": "OKTOBER",

            "NOPEMBER": "NOPEMBER",
            "NOVEMBER": "NOVEMBER",

            "DESEMBER": "DESEMBER",
            "DESEM8ER": "DESEMBER",
        }

        result = value

        for old, new in replacements.items():

            result = re.sub(
                rf"\b{re.escape(old)}\b",
                new,
                result,
                flags=re.IGNORECASE,
            )

        return result

    # =========================================================================
    # MONTHLY TABLE NORMALIZATION
    # =========================================================================

    def _normalize_monthly_table(
            self,
            lines: list[str],
    ) -> list[str]:
        """
        Memperbaiki pola tabel bulanan yang sering rusak oleh OCR.

        Contoh OCR:

            6GIJUNI Rp 314.656.500 | Rp 8.338.397

        menjadi:

            6 | JUNI | Rp 314.656.500 | Rp 8.338.397

        Jika nomor bulan menempel dengan nama bulan, nomor dipisahkan.
        """

        result: list[str] = []

        months = (
            "JANUARI",
            "FEBRUARI",
            "MARET",
            "APRIL",
            "MEI",
            "JUNI",
            "JULI",
            "AGUSTUS",
            "SEPTEMBER",
            "OKTOBER",
            "NOPEMBER",
            "NOVEMBER",
            "DESEMBER",
        )

        month_pattern = "|".join(
            re.escape(month)
            for month in months
        )

        for line in lines:

            normalized = line

            # --------------------------------------------------------------
            # OCR:
            #
            # 6GIJUNI
            # 7WULI
            # 10JOKTOBER
            #
            # Pertahankan angka bulan dan nama bulan.
            # --------------------------------------------------------------

            match = re.search(
                rf"(?<!\d)(\d{{1,2}})\s*[A-Z0-9Il|]*?"
                rf"({month_pattern})\b",
                normalized,
                flags=re.IGNORECASE,
            )

            if match:

                month_number = (
                    match.group(1)
                )

                month_name = (
                    match.group(2).upper()
                )

                prefix = normalized[
                    :match.start()
                ]

                suffix = normalized[
                    match.end():
                ]

                # Hilangkan artefak huruf di antara
                # nomor bulan dan nama bulan.
                prefix = prefix.rstrip()

                normalized = (
                        prefix
                        + month_number
                        + " | "
                        + month_name
                        + suffix
                )

            result.append(
                normalized
            )

        return result

    # =========================================================================
    # PDF FALLBACK
    # =========================================================================

    def _pdf_pypdf2(
            self,
            path: Path,
    ) -> str:
        """
        Fallback PDF extraction menggunakan PyPDF2.
        """

        try:
            import PyPDF2  # type: ignore

        except ImportError:

            raise ImportError(
                "pdfplumber atau PyPDF2 diperlukan "
                "untuk parsing PDF."
            )

        text_parts: list[str] = []

        with open(
                path,
                "rb",
        ) as fh:

            reader = PyPDF2.PdfReader(
                fh
            )

            for page_number, page in enumerate(
                    reader.pages,
                    start=1,
            ):

                try:

                    text = (
                            page.extract_text()
                            or ""
                    )

                    if text.strip():

                        text = (
                            self._normalize_pdf_text(
                                text
                            )
                        )

                        text_parts.append(
                            "[TEXT - Halaman "
                            f"{page_number}]\n"
                            f"{text.strip()}"
                        )

                except Exception as exc:

                    logger.warning(
                        "Gagal membaca halaman %s "
                        "dari %s: %s",
                        page_number,
                        path,
                        exc,
                    )

        return "\n\n".join(
            text_parts
        )

    # =========================================================================
    # PDF TABLE
    # =========================================================================

    def _format_pdf_table(
            self,
            table: list[list[Any]],
            page_number: int,
            table_number: int,
    ) -> str:
        """
        Mengubah tabel pdfplumber menjadi teks terstruktur.

        Setiap baris tetap berdekatan agar chunker tidak mudah
        memisahkan label dan nilainya.
        """

        cleaned_rows: list[list[str]] = []

        for row in table:

            if not row:
                continue

            cleaned_row: list[str] = []

            for cell in row:

                if cell is None:

                    cleaned_row.append("")
                    continue

                value = str(cell)

                value = " ".join(
                    value.split()
                )

                value = (
                    self._normalize_pdf_number(
                        value
                    )
                )

                value = (
                    self._normalize_month_names(
                        value
                    )
                )

                cleaned_row.append(
                    value
                )

            if any(
                    cell.strip()
                    for cell in cleaned_row
            ):
                cleaned_rows.append(
                    cleaned_row
                )

        if not cleaned_rows:
            return ""

        lines: list[str] = [
            "[TABLE - Halaman "
            f"{page_number} - Tabel "
            f"{table_number}]"
        ]

        for row_number, row in enumerate(
                cleaned_rows,
                start=1,
        ):

            cells: list[str] = []

            for column_number, value in enumerate(
                    row,
                    start=1,
            ):

                if value:

                    cells.append(
                        f"Kolom {column_number}: "
                        f"{value}"
                    )

            if cells:

                lines.append(
                    f"Baris {row_number}: "
                    + " | ".join(cells)
                )

        return "\n".join(
            lines
        )

    # =========================================================================
    # PDF TEXT NORMALIZATION
    # =========================================================================

    def _normalize_pdf_text(
            self,
            text: str,
    ) -> str:
        """
        Normalisasi teks native PDF.
        """

        lines = text.splitlines()

        normalized_lines: list[str] = []

        for line in lines:

            line = (
                self._normalize_pdf_number(
                    line
                )
            )

            line = (
                self._normalize_month_names(
                    line
                )
            )

            normalized_lines.append(
                line
            )

        return "\n".join(
            normalized_lines
        )

    # =========================================================================
    # PDF NUMBER
    # =========================================================================

    def _normalize_pdf_number(
            self,
            value: str,
    ) -> str:
        """
        Memperbaiki artefak angka pada PDF native.

        Contoh:

            Rp 6 7,580,133,333
            ->
            Rp 67,580,133,333

            Rp 1 02,411,649,646
            ->
            Rp 102,411,649,646
        """

        # Normalisasi spasi setelah Rp
        value = re.sub(
            r"\bRp\s+",
            "Rp ",
            value,
        )

        def fix_currency(
                match: re.Match[str],
        ) -> str:

            prefix = match.group(1)

            number = match.group(2)

            number = re.sub(
                r"(?<=\d)\s+(?=\d)",
                "",
                number,
            )

            number = re.sub(
                r"(?<=\d)\s+(?=,)",
                "",
                number,
            )

            number = re.sub(
                r"(?<=,)\s+(?=\d)",
                "",
                number,
            )

            return (
                    prefix
                    + number
            )

        value = re.sub(
            r"(Rp\s*)([\d\s,().-]+)",
            fix_currency,
            value,
        )

        return value

    # =========================================================================
    # DOCX
    # =========================================================================

    def _docx(
            self,
            path: Path,
    ) -> str:

        try:
            import docx  # type: ignore

        except ImportError:

            raise ImportError(
                "python-docx diperlukan untuk "
                "parsing DOCX: pip install python-docx"
            )

        doc = docx.Document(
            str(path)
        )

        parts: list[str] = []

        # Paragraph
        for para in doc.paragraphs:

            text = para.text.strip()

            if text:
                parts.append(
                    text
                )

        # Tables
        for table_index, table in enumerate(
                doc.tables,
                start=1,
        ):

            parts.append(
                f"[TABLE - Tabel {table_index}]"
            )

            for row_number, row in enumerate(
                    table.rows,
                    start=1,
            ):

                values: list[str] = []

                for column_number, cell in enumerate(
                        row.cells,
                        start=1,
                ):

                    value = (
                        cell.text.strip()
                    )

                    if value:

                        values.append(
                            f"Kolom {column_number}: "
                            f"{value}"
                        )

                if values:

                    parts.append(
                        f"Baris {row_number}: "
                        + " | ".join(values)
                    )

        return "\n".join(
            parts
        )

    # =========================================================================
    # MARKDOWN
    # =========================================================================

    def _markdown(
            self,
            path: Path,
    ) -> str:

        try:

            import markdown  # type: ignore
            from html.parser import HTMLParser

            class _Stripper(
                HTMLParser
            ):

                def __init__(self):
                    super().__init__()
                    self._parts: list[str] = []

                def handle_data(
                        self,
                        data: str,
                ) -> None:
                    self._parts.append(
                        data
                    )

                def get_text(
                        self,
                ) -> str:
                    return "".join(
                        self._parts
                    )

            raw = path.read_text(
                encoding="utf-8",
                errors="replace",
            )

            html = markdown.markdown(
                raw
            )

            stripper = _Stripper()

            stripper.feed(
                html
            )

            return stripper.get_text()

        except ImportError:

            logger.warning(
                "Package markdown tidak tersedia. "
                "Menggunakan teks MD asli."
            )

            return path.read_text(
                encoding="utf-8",
                errors="replace",
            )

    # =========================================================================
    # PLAIN
    # =========================================================================

    def _plain(
            self,
            path: Path,
    ) -> str:

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    # =========================================================================
    # HTML
    # =========================================================================

    def _html(
            self,
            path: Path,
    ) -> str:

        from html.parser import HTMLParser

        class _Stripper(
            HTMLParser
        ):

            def __init__(self):
                super().__init__()

                self._parts: list[str] = []

                self._skip_tags = {
                    "script",
                    "style",
                }

                self._current_skip: str | None = None

            def handle_starttag(
                    self,
                    tag: str,
                    attrs,
            ) -> None:

                if tag in self._skip_tags:
                    self._current_skip = tag

            def handle_endtag(
                    self,
                    tag: str,
            ) -> None:

                if tag == self._current_skip:
                    self._current_skip = None

            def handle_data(
                    self,
                    data: str,
            ) -> None:

                if self._current_skip is None:
                    self._parts.append(
                        data
                    )

            def get_text(
                    self,
            ) -> str:

                return " ".join(
                    self._parts
                )

        raw = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        stripper = _Stripper()

        stripper.feed(
            raw
        )

        return stripper.get_text()

    # =========================================================================
    # XLSX
    # =========================================================================

    def _xlsx(
            self,
            path: Path,
    ) -> str:

        try:
            import openpyxl  # type: ignore

        except ImportError:

            raise ImportError(
                "openpyxl diperlukan untuk XLSX: "
                "pip install openpyxl"
            )

        workbook = openpyxl.load_workbook(
            filename=str(path),
            read_only=True,
            data_only=True,
        )

        parts: list[str] = []

        try:

            for worksheet in workbook.worksheets:

                parts.append(
                    f"[SHEET - {worksheet.title}]"
                )

                for row_number, row in enumerate(
                        worksheet.iter_rows(
                            values_only=True
                        ),
                        start=1,
                ):

                    values: list[str] = []

                    for column_number, value in enumerate(
                            row,
                            start=1,
                    ):

                        if value is not None:

                            text = str(
                                value
                            ).strip()

                            if text:

                                values.append(
                                    f"Kolom {column_number}: "
                                    f"{text}"
                                )

                    if values:

                        parts.append(
                            f"Baris {row_number}: "
                            + " | ".join(values)
                        )

        finally:

            workbook.close()

        return "\n".join(
            parts
        )

    # =========================================================================
    # XLS
    # =========================================================================

    def _xls(
            self,
            path: Path,
    ) -> str:

        try:
            import xlrd  # type: ignore

        except ImportError:

            raise ImportError(
                "xlrd diperlukan untuk XLS: "
                "pip install xlrd"
            )

        workbook = xlrd.open_workbook(
            str(path)
        )

        parts: list[str] = []

        for sheet in workbook.sheets():

            parts.append(
                f"[SHEET - {sheet.name}]"
            )

            for row_number in range(
                    sheet.nrows
            ):

                values: list[str] = []

                for column_number in range(
                        sheet.ncols
                ):

                    value = sheet.cell_value(
                        row_number,
                        column_number,
                    )

                    if (
                            value is not None
                            and str(value).strip()
                    ):

                        values.append(
                            f"Kolom {column_number + 1}: "
                            f"{value}"
                        )

                if values:

                    parts.append(
                        f"Baris {row_number + 1}: "
                        + " | ".join(values)
                    )

        return "\n".join(
            parts
        )

    # =========================================================================
    # CODE
    # =========================================================================

    def _code(
            self,
            path: Path,
    ) -> str:

        content = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        return (
            f"Source code: {path.name}\n\n"
            f"{content}"
        )

    # =========================================================================
    # JSON
    # =========================================================================

    def _json(
            self,
            path: Path,
    ) -> str:

        raw = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        try:

            data = _json.loads(
                raw
            )

            return _json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

        except _json.JSONDecodeError:

            return raw

    # =========================================================================
    # XML
    # =========================================================================

    def _xml(
            self,
            path: Path,
    ) -> str:

        from xml.etree import ElementTree

        raw = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        try:

            root = ElementTree.fromstring(
                raw
            )

            parts: list[str] = []

            for elem in root.iter():

                if (
                        elem.text
                        and elem.text.strip()
                ):
                    parts.append(
                        elem.text.strip()
                    )

                if (
                        elem.tail
                        and elem.tail.strip()
                ):
                    parts.append(
                        elem.tail.strip()
                    )

            return "\n".join(
                parts
            )

        except ElementTree.ParseError:

            return raw

    # =========================================================================
    # EML
    # =========================================================================

    def _eml(
            self,
            path: Path,
    ) -> str:

        import email
        from email import policy as _policy

        raw = path.read_bytes()

        msg = email.message_from_bytes(
            raw,
            policy=_policy.default,
        )

        parts: list[str] = [
            f"From: {msg.get('From', '')}",
            f"To: {msg.get('To', '')}",
            f"Subject: {msg.get('Subject', '')}",
            f"Date: {msg.get('Date', '')}",
            "",
        ]

        if msg.is_multipart():

            for part in msg.walk():

                if (
                        part.get_content_type()
                        == "text/plain"
                ):

                    try:

                        payload = (
                            part.get_payload(
                                decode=True
                            )
                        )

                        if payload:

                            parts.append(
                                payload.decode(
                                    "utf-8",
                                    errors="replace",
                                )
                            )

                    except Exception:
                        pass

        else:

            try:

                payload = (
                    msg.get_payload(
                        decode=True
                    )
                )

                if payload:

                    parts.append(
                        payload.decode(
                            "utf-8",
                            errors="replace",
                        )
                    )

            except Exception:

                parts.append(
                    str(
                        msg.get_payload()
                    )
                )

        return "\n".join(
            parts
        )

    # =========================================================================
    # EPUB
    # =========================================================================

    def _epub(
            self,
            path: Path,
    ) -> str:

        try:

            import ebooklib  # type: ignore
            from ebooklib import epub  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore

        except ImportError:

            raise ImportError(
                "ebooklib dan beautifulsoup4 diperlukan "
                "untuk EPUB: "
                "pip install ebooklib beautifulsoup4"
            )

        book = epub.read_epub(
            str(path),
            options={
                "ignore_ncx": True
            },
        )

        text_parts: list[str] = []

        for item in book.get_items_of_type(
                ebooklib.ITEM_DOCUMENT
        ):

            try:

                soup = BeautifulSoup(
                    item.get_content(),
                    "html.parser",
                )

                text = soup.get_text(
                    separator="\n",
                    strip=True,
                )

                if text:

                    text_parts.append(
                        text
                    )

            except Exception as exc:

                logger.warning(
                    "Gagal membaca chapter EPUB %s: %s",
                    path,
                    exc,
                )

        return "\n\n".join(
            text_parts
        )
