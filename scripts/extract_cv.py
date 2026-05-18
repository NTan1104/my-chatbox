"""
extract_cv_docling.py
---------------------
Trích xuất text từ CV PDF bằng Docling.
Hỗ trợ cả CV tiếng Việt và tiếng Anh, fallback sang pdfplumber nếu Docling thất bại.

Cấu trúc thư mục:
    data/raw/          ← để toàn bộ PDF ở đây (bất kể ngôn ngữ)
    data/extracted/
        docling/
            md/        ← file .md per PDF
            docling_cv_texts.jsonl
            docling_extract_report.jsonl
"""

from __future__ import annotations

import json
import re
import html
import unicodedata
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"

OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted" / "docling"
OUTPUT_MD_DIR = OUTPUT_DIR / "md"
OUTPUT_JSONL = OUTPUT_DIR / "docling_cv_texts.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "docling_extract_report.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tech name normalization — mở rộng thêm nhiều pattern
# ---------------------------------------------------------------------------

TECH_NORMALIZATION: dict[str, str] = {
    # JavaScript ecosystem
    r"\bJavascript\b": "JavaScript",
    r"\bJava Script\b": "JavaScript",
    r"\bJS\b": "JavaScript",
    r"\bTypescript\b": "TypeScript",
    r"\bTS\b": "TypeScript",
    r"\bNodejs\b": "Node.js",
    r"\bNode JS\b": "Node.js",
    r"\bNode\.JS\b": "Node.js",
    r"\bReactjs\b": "React.js",
    r"\bReact JS\b": "React.js",
    r"\bReact-native\b": "React Native",
    r"\bReactNative\b": "React Native",
    r"\bVuejs\b": "Vue.js",
    r"\bVue JS\b": "Vue.js",
    r"\bNextjs\b": "Next.js",
    r"\bNext JS\b": "Next.js",
    r"\bNuxtjs\b": "Nuxt.js",
    r"\bNuxt JS\b": "Nuxt.js",
    r"\bNestjs\b": "NestJS",
    r"\bNest\.js\b": "NestJS",
    r"\bExpressjs\b": "Express.js",
    r"\bExpress JS\b": "Express.js",
    r"\bAngularjs\b": "AngularJS",
    r"\bAngular JS\b": "AngularJS",
    r"\bSveltekit\b": "SvelteKit",
    # Backend frameworks
    r"\bSpringboot\b": "Spring Boot",
    r"\bSpring-boot\b": "Spring Boot",
    r"\bDjango REST\b": "Django REST Framework",
    r"\bFastAPI\b": "FastAPI",
    r"\bWordpress\b": "WordPress",
    r"\bLaravel\b": "Laravel",
    # Databases
    r"\bMysql\b": "MySQL",
    r"\bSQL [Ss]erver\b": "SQL Server",
    r"\bPostgre\b": "PostgreSQL",
    r"\bPostgres\b": "PostgreSQL",
    r"\bPostgresql\b": "PostgreSQL",
    r"\bMongodb\b": "MongoDB",
    r"\bRedis\b": "Redis",
    r"\bElasticsearch\b": "Elasticsearch",
    r"\bFirebase\b": "Firebase",
    r"\bSQLite\b": "SQLite",
    r"\bCassandra\b": "Cassandra",
    # DevOps / Cloud
    r"\bGithub\b": "GitHub",
    r"\bGitlab\b": "GitLab",
    r"\bDocker compose\b": "Docker Compose",
    r"\bKubernetes\b": "Kubernetes",
    r"\bK8s\b": "Kubernetes",
    r"\bAWS\b": "AWS",
    r"\bGCP\b": "GCP",
    r"\bAzure\b": "Azure",
    r"\bCI/CD\b": "CI/CD",
    r"\bJenkins\b": "Jenkins",
    r"\bTerraform\b": "Terraform",
    r"\bAnsible\b": "Ansible",
    # Web / Markup
    r"\bHtml\b": "HTML",
    r"\bCss\b": "CSS",
    r"\bSass\b": "Sass",
    r"\bScss\b": "SCSS",
    r"\bBoostrap\b": "Bootstrap",
    r"\bBootstrap\b": "Bootstrap",
    r"\bTailwind\b": "Tailwind CSS",
    # API
    r"\bRESTful API\b": "REST API",
    r"\bRest API\b": "REST API",
    r"\bGraphql\b": "GraphQL",
    r"\bGRPC\b": "gRPC",
    # AI/ML
    r"\bTensorflow\b": "TensorFlow",
    r"\bPytorch\b": "PyTorch",
    r"\bScikit-learn\b": "scikit-learn",
    r"\bOpenCV\b": "OpenCV",
}

# Compile trước để tăng tốc
_TECH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), rep)
    for pat, rep in TECH_NORMALIZATION.items()
]

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Pattern xóa noise — compile một lần
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_PRIVATE_UNICODE = re.compile(r"[\uE000-\uF8FF]")
_CHECKBOX_MD = re.compile(r"-\s*\[\s*[xX ]?\s*\]\s*")
_WHITESPACE_LINE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

# Docling đôi khi tạo ra dòng chỉ gồm ký tự đặc biệt lặp lại (----, ====)
_DECORATIVE_LINE = re.compile(r"^[-=_*#|]{4,}$")

# Số điện thoại VN — chuẩn hóa format
_PHONE_VN = re.compile(r"(\+84|0)[\s\-\.]?(3[2-9]|5[6-9]|7[06-9]|8[1-9]|9[0-9])[\s\-\.]?(\d{3})[\s\-\.]?(\d{4})")


def _normalize_phone(text: str) -> str:
    """Chuẩn hóa SĐT Việt Nam về dạng 0xxxxxxxxx."""
    def replace_phone(m: re.Match) -> str:
        prefix = m.group(1)
        mid = m.group(2)
        p1 = m.group(3)
        p2 = m.group(4)
        if prefix == "+84":
            return f"0{mid}{p1}{p2}"
        return f"0{mid}{p1}{p2}"
    return _PHONE_VN.sub(replace_phone, text)


def normalize_text(text: str) -> str:
    """
    Làm sạch text sau khi Docling extract.
    - Không merge từ tiếng Việt để tránh lỗi dấu.
    - Xử lý noise đặc trưng của PDF CV.
    """
    if not text:
        return ""

    # Unicode normalization (NFC) — quan trọng cho tiếng Việt
    text = unicodedata.normalize("NFC", text)

    # Unescape HTML entities (&amp; &lt; …)
    text = html.unescape(text)

    # Chuẩn hóa line ending
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Xóa null bytes
    text = text.replace("\x00", "")

    # Xóa HTML comment của Docling (<!-- image -->, ...)
    text = _HTML_COMMENT.sub("", text)

    # Xóa private-use unicode (icon font)
    text = _PRIVATE_UNICODE.sub("", text)

    # Xóa checkbox markdown [ ] [x]
    text = _CHECKBOX_MD.sub("- ", text)

    # Chuẩn hóa tên công nghệ
    for pattern, replacement in _TECH_PATTERNS:
        text = pattern.sub(replacement, text)

    # Chuẩn hóa SĐT
    text = _normalize_phone(text)

    # Xử lý từng dòng
    lines: list[str] = []
    for line in text.split("\n"):
        line = _WHITESPACE_LINE.sub(" ", line).strip()

        # Bỏ dòng chỉ toàn ký tự trang trí
        if _DECORATIVE_LINE.match(line):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # Giới hạn dòng trống liên tiếp (tối đa 2)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_filename(filename: str) -> str:
    """Tạo tên file an toàn từ tên PDF gốc."""
    name = Path(filename).stem
    name = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def collect_pdf_files() -> list[dict[str, Any]]:
    """
    Lấy toàn bộ PDF từ data/raw/ (không phân biệt thư mục con).
    """
    if not RAW_DIR.exists():
        log.warning("Không tìm thấy thư mục: %s", RAW_DIR.resolve())
        return []

    pdf_items = []
    for pdf_path in sorted(RAW_DIR.rglob("*.pdf")):
        pdf_items.append({"pdf_path": pdf_path})

    return pdf_items


# ---------------------------------------------------------------------------
# Extraction: Docling (primary) + pdfplumber (fallback)
# ---------------------------------------------------------------------------

def _extract_docling(pdf_path: Path, converter: Any) -> tuple[str, dict]:
    """Trích xuất bằng Docling, export Markdown."""
    try:
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        markdown = normalize_text(markdown)

        if len(markdown.strip()) < 50:
            return "", {
                "method": "docling",
                "success": False,
                "error": "Text quá ngắn sau extract (< 50 ký tự) — nghi ngờ PDF scan",
            }

        return markdown, {"method": "docling", "success": True, "error": None}

    except Exception as e:
        return "", {"method": "docling", "success": False, "error": str(e)}


def _extract_pdfplumber(pdf_path: Path) -> tuple[str, dict]:
    """
    Fallback: dùng pdfplumber để lấy raw text.
    Giữ layout (x_tolerance nhỏ) để tránh merge từ tiếng Việt.
    """
    try:
        import pdfplumber

        pages_text: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                # extract_text với layout=True giữ khoảng cách cột
                page_text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=3,
                    layout=True,
                    x_density=7.25,
                    y_density=13,
                ) or ""
                pages_text.append(page_text)

        text = "\n\n".join(pages_text)
        text = normalize_text(text)

        if not text.strip():
            return "", {
                "method": "pdfplumber",
                "success": False,
                "error": "pdfplumber không extract được text — có thể PDF scan",
            }

        return text, {"method": "pdfplumber", "success": True, "error": None}

    except ImportError:
        return "", {
            "method": "pdfplumber",
            "success": False,
            "error": "pdfplumber chưa được cài (pip install pdfplumber)",
        }
    except Exception as e:
        return "", {"method": "pdfplumber", "success": False, "error": str(e)}


def process_pdf(pdf_path: Path, converter: Any) -> dict[str, Any]:
    """
    Xử lý 1 file PDF:
    1. Thử Docling
    2. Nếu fail → fallback pdfplumber
    3. Ghi log warning nếu cả hai đều fail
    """
    # --- Primary: Docling ---
    text, meta = _extract_docling(pdf_path, converter)

    fallback_used = False
    fallback_error = None

    # --- Fallback: pdfplumber ---
    if not meta["success"]:
        log.warning("[%s] Docling fail (%s) → fallback pdfplumber", pdf_path.name, meta["error"])
        fallback_error = meta["error"]

        text, meta = _extract_pdfplumber(pdf_path)
        fallback_used = True

        if not meta["success"]:
            log.error("[%s] pdfplumber cũng fail: %s", pdf_path.name, meta["error"])

    status = "ok" if meta["success"] else "failed"
    method = meta["method"]
    if fallback_used:
        method = f"pdfplumber_fallback (docling_error: {fallback_error})"

    return {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "status": status,
        "extract_method": method,
        "text": text,
        "text_length": len(text),
        "error": meta.get("error"),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_jsonl(f: Any, record: dict) -> None:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_record(result: dict, md_path: Path) -> dict:
    return {
        "file_name": result["file_name"],
        "file_path": result["file_path"],
        "status": result["status"],
        "extract_method": result["extract_method"],
        "text_length": result["text_length"],
        "text": result["text"],
        "output_md_path": str(md_path),
    }


def _build_report(result: dict, md_path: Path) -> dict:
    return {
        "file_name": result["file_name"],
        "file_path": result["file_path"],
        "status": result["status"],
        "extract_method": result["extract_method"],
        "text_length": result["text_length"],
        "error": result["error"],
        "output_md_path": str(md_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD_DIR.mkdir(parents=True, exist_ok=True)

    pdf_items = collect_pdf_files()

    if not pdf_items:
        log.error("Không tìm thấy PDF nào trong: %s", RAW_DIR.resolve())
        return

    log.info("Tìm thấy %d file PDF trong %s", len(pdf_items), RAW_DIR.resolve())
    log.info("Output dir: %s", OUTPUT_DIR.resolve())

    # Khởi tạo Docling converter một lần duy nhất (tốn RAM nếu tạo nhiều lần)
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        # Bật OCR cho PDF scan — quan trọng với CV chụp ảnh
        pipeline_options.do_ocr = True
        # Bật table detection — CV thường có bảng kỹ năng, timeline
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: {"pipeline_options": pipeline_options}
            }
        )
        log.info("Docling DocumentConverter đã sẵn sàng (OCR + Table detection bật)")

    except ImportError:
        log.error("docling chưa được cài. Chạy: pip install docling")
        return
    except Exception as e:
        log.error("Không thể khởi tạo Docling: %s", e)
        return

    # Stats
    stats = {"ok": 0, "failed": 0, "fallback": 0}

    with (
        open(OUTPUT_JSONL, "w", encoding="utf-8") as data_f,
        open(OUTPUT_REPORT, "w", encoding="utf-8") as report_f,
    ):
        for item in tqdm(pdf_items, desc="Extracting CVs", unit="file"):
            pdf_path: Path = item["pdf_path"]

            result = process_pdf(pdf_path, converter)

            # Đường dẫn output .md — giữ cấu trúc thư mục con nếu có
            rel = pdf_path.relative_to(RAW_DIR)
            md_path = OUTPUT_MD_DIR / rel.parent / f"{safe_filename(pdf_path.name)}.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)

            # Ghi file .md
            if result["text"]:
                md_path.write_text(result["text"], encoding="utf-8")
            else:
                md_path.write_text(
                    f"# {pdf_path.stem}\n\n> ⚠️ Không extract được text.\n",
                    encoding="utf-8",
                )

            _write_jsonl(data_f, _build_record(result, md_path))
            _write_jsonl(report_f, _build_report(result, md_path))

            # Cập nhật stats
            if result["status"] == "ok":
                if "fallback" in result["extract_method"]:
                    stats["fallback"] += 1
                else:
                    stats["ok"] += 1
            else:
                stats["failed"] += 1

    # Summary
    total = len(pdf_items)
    log.info("=" * 50)
    log.info("Hoàn tất xử lý %d file PDF", total)
    log.info("  ✅ Docling OK     : %d", stats["ok"])
    log.info("  ⚠️  Fallback OK   : %d", stats["fallback"])
    log.info("  ❌ Thất bại       : %d", stats["failed"])
    log.info("Markdown : %s", OUTPUT_MD_DIR.resolve())
    log.info("JSONL    : %s", OUTPUT_JSONL.resolve())
    log.info("Report   : %s", OUTPUT_REPORT.resolve())


if __name__ == "__main__":
    main()