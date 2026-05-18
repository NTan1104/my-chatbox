"""
parse_jd_gemini.py — Parse raw JD .txt/.md files into structured JSON using Gemini.

Features:
- Load GEMINI_API_KEY from .env
- Structured output with Gemini
- Fallback JSON mode if schema mode fails
- Exponential backoff retry with jitter
- Rate limiting between API calls
- Resume mode: skip already parsed jd_ids
- Cross-field deduplication
- Parse experience_required, education_requirement, salary_range
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

INPUT_DIR = PROJECT_ROOT / "data" / "jd" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "jd" / "parsed"
OUTPUT_JSONL = OUTPUT_DIR / "parsed_jd.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "parsed_jd_report.jsonl"

DEFAULT_MODEL = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = ("*.txt", "*.md")

MIN_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tech normalization
# ---------------------------------------------------------------------------

_RAW_NORMALIZATION: Dict[str, str] = {
    "javascript": "JavaScript",
    "java script": "JavaScript",
    "typescript": "TypeScript",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node.js": "Node.js",
    "reactjs": "React.js",
    "react js": "React.js",
    "react.js": "React.js",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "vue.js": "Vue.js",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "next.js": "Next.js",
    "nestjs": "NestJS",
    "nest.js": "NestJS",
    "expressjs": "Express.js",
    "express js": "Express.js",
    "express.js": "Express.js",
    "springboot": "Spring Boot",
    "spring boot": "Spring Boot",
    "wordpress": "WordPress",
    "mysql": "MySQL",
    "postgre": "PostgreSQL",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "github": "GitHub",
    "gitlab": "GitLab",
    "rest api": "REST API",
    "restful api": "REST API",
    "restful apis": "REST API",
    "docker compose": "Docker Compose",
    "deepstream": "DeepStream SDK",
    "deepstream sdk": "DeepStream SDK",
    "gstreamer": "GStreamer",
    "opencv": "OpenCV",
    "ci/cd": "CI/CD",
    "wifi": "WiFi",
    "wi-fi": "WiFi",
}

_SORTED_KEYS = sorted(_RAW_NORMALIZATION.keys(), key=len, reverse=True)
_NORM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SORTED_KEYS) + r")\b",
    flags=re.IGNORECASE,
)


def _normalize_token(match: re.Match) -> str:
    return _RAW_NORMALIZATION[match.group(0).lower()]


def normalize_tech_terms(text: str) -> str:
    if not text:
        return ""
    return _NORM_PATTERN.sub(_normalize_token, text)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

class ExperienceRequired(BaseModel):
    min_years: Optional[float] = None
    max_years: Optional[float] = None
    description: Optional[str] = None


class EducationRequirement(BaseModel):
    min_degree: Optional[str] = None
    majors: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class ParsedJD(BaseModel):
    jd_id: str = Field(description="Unique JD id")
    source_file: Optional[str] = None

    job_title: Optional[str] = None
    domain: Optional[str] = "IT"
    sub_domain: Optional[str] = None
    level: Optional[str] = None

    responsibilities: List[str] = Field(default_factory=list)

    must_have_skills: List[str] = Field(default_factory=list)
    strong_preferred_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)

    tools_platforms: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    devops_cloud: List[str] = Field(default_factory=list)

    soft_skills: List[str] = Field(default_factory=list)
    language_requirement: List[str] = Field(default_factory=list)

    experience_required: ExperienceRequired = Field(default_factory=ExperienceRequired)
    education_requirement: EducationRequirement = Field(default_factory=EducationRequirement)

    salary_range: Optional[str] = None
    working_time: Optional[str] = None

    keywords: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_jd_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    text = normalize_tech_terms(text)

    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_jd_id(file_path: Path, index: int) -> str:
    stem = re.sub(r"[^\w\-]+", "_", file_path.stem, flags=re.UNICODE).strip("_")

    if stem.lower().startswith("jd_"):
        return stem

    return f"JD_{stem}" if stem else f"JD_{index:04d}"


def collect_jd_files(input_dir: Path) -> List[Path]:
    files: List[Path] = []

    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(ext))

    return sorted(set(files))


def load_already_parsed_ids(output_jsonl: Path) -> Set[str]:
    seen: Set[str] = set()

    if not output_jsonl.exists():
        return seen

    with output_jsonl.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                jd_id = record.get("jd_id")
                if jd_id:
                    seen.add(jd_id)
            except json.JSONDecodeError:
                continue

    return seen


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_prompt(jd_raw: str, jd_id: str, source_file: str) -> str:
    return f"""
Bạn là hệ thống trích xuất thông tin JD ngành CNTT phục vụ CV-JD Matching.

## Nhiệm vụ
Đọc JD raw và trả về JSON object theo schema bên dưới.

## Quy tắc output
- Chỉ trả về JSON object hợp lệ.
- Không giải thích.
- Không dùng markdown code block.
- Không tự bịa thông tin không có trong JD.
- Nếu không tìm thấy: dùng null cho string/object field hoặc [] cho array field.

## Quy tắc phân loại kỹ năng/công nghệ

### Nguyên tắc chung
- Mỗi kỹ năng/công nghệ chỉ xuất hiện trong MỘT field duy nhất.
- Nếu là database cụ thể, đưa vào databases.
- Nếu là cloud/devops/deployment/container/CI-CD, đưa vào devops_cloud.
- Nếu là phần mềm, SDK, framework, platform, công cụ kỹ thuật cụ thể, đưa vào tools_platforms.
- Nếu JD yêu cầu một nhóm năng lực chung, đưa nhóm năng lực đó vào must_have_skills hoặc strong_preferred_skills.

Ví dụ:
- "Có kiến thức Network LAN, WAN, WiFi" → strong_preferred_skills: ["Network (LAN/WAN/WiFi)"]
- "MySQL, Redis, MongoDB" → databases
- "Docker, Kubernetes, AWS, CI/CD" → devops_cloud
- "Jira, Git, Postman, Figma" → tools_platforms

### must_have_skills
Kỹ năng/yêu cầu bắt buộc, thiếu là rất khó phù hợp.
Trigger: "yêu cầu", "bắt buộc", "thành thạo", "có kinh nghiệm X năm", "kinh nghiệm ít nhất", "kiến thức vững", "nắm rõ".

Không đưa vào must_have_skills:
- Bằng cấp.
- Số năm kinh nghiệm tổng quát.
- Công cụ cụ thể đã thuộc databases/devops_cloud/tools_platforms.
- Kỹ năng mềm.

### strong_preferred_skills
Kỹ năng quan trọng, không có thì yếu thế nhưng vẫn có thể xem xét.
Trigger: "có kiến thức về", "hiểu biết về", "quen thuộc với", "có khả năng".

### nice_to_have_skills
Kỹ năng phụ, có thì là lợi thế.
Trigger: "điểm cộng", "nice to have", "biết thêm", "ưu tiên nếu có", "là một lợi thế".

### tools_platforms
Tên phần mềm, framework, SDK, platform, ứng dụng, hệ thống cụ thể.
Ví dụ: Git, Jira, Postman, Figma, Android Studio, Microsoft Office, Outlook, DeepStream SDK, OpenCV, GStreamer.

Không đưa vào tools_platforms:
- Thiết bị vật lý quá chung như "máy in", "máy tính", "camera" nếu chúng chỉ là đối tượng hỗ trợ.
- Tham số cấu hình như SSID, Channel, Security Policy.
- Skill đã nằm ở must/preferred/nice.

### databases
Tên hệ quản trị cơ sở dữ liệu hoặc cache/search engine.
Ví dụ: MySQL, PostgreSQL, MongoDB, Redis, SQL Server, Oracle, Elasticsearch.

### devops_cloud
Cloud, CI/CD, container, hạ tầng, deployment.
Ví dụ: Docker, Kubernetes, AWS, GCP, Azure, CI/CD, Jenkins, GitHub Actions, Linux, Nginx.

### soft_skills
Kỹ năng mềm phi kỹ thuật.
Ví dụ: giao tiếp, làm việc nhóm, xử lý vấn đề, tư duy logic, chủ động học hỏi.

Không đưa nhiệm vụ công việc vào soft_skills.
Nếu câu có tân ngữ cụ thể như "thiết lập mối quan hệ với khách hàng" thì đưa vào responsibilities, không đưa vào soft_skills.

### language_requirement
Yêu cầu ngoại ngữ, ghi rõ mức độ nếu có.
Ví dụ: "Tiếng Anh đọc hiểu tài liệu kỹ thuật", "Tiếng Anh giao tiếp", "Tiếng Nhật N3".

## Quy tắc các field khác

### responsibilities
- Tối đa 8-12 nhiệm vụ chính.
- Gộp các task cùng nhóm, không copy từng bullet nhỏ.
- Dùng động từ hành động mở đầu.
- Không đưa yêu cầu kỹ năng, bằng cấp, số năm kinh nghiệm vào responsibilities.

Ví dụ:
Sai:
- "Thêm máy in mới"
- "Cập nhật driver"
- "Quản lý quyền in"

Đúng:
- "Quản lý hệ thống in ấn, bao gồm cài đặt máy in, cập nhật driver và phân quyền truy cập"

### experience_required
Trích xuất yêu cầu kinh nghiệm nếu có.

min_years:
- "ít nhất 2 năm" → 2
- "1-3 năm" → 1
- "trên 3 năm" → 3
- "không yêu cầu kinh nghiệm" → 0
- không đề cập → null

max_years:
- "1-3 năm" → 3
- "2-4 năm" → 4
- chỉ ghi tối thiểu → null
- không đề cập → null

description:
- Ghi ngắn gọn yêu cầu kinh nghiệm từ JD.
- Không có → null.

### education_requirement
Trích xuất yêu cầu học vấn nếu có.

min_degree:
Chỉ dùng một trong: "Trung cấp", "Cao đẳng", "Đại học", "Thạc sĩ", "Tiến sĩ", null.

majors:
Danh sách ngành học yêu cầu hoặc ưu tiên.
Ví dụ: ["Công nghệ thông tin", "Khoa học máy tính", "Kỹ thuật phần mềm"].

notes:
Ghi chú thêm nếu có.

### salary_range
- Trích xuất mức lương nếu JD có đề cập.
- Ví dụ: "10.000.000 - 15.000.000 VND".
- Nếu không có → null.

### level
Suy luận theo số năm kinh nghiệm:
- 0 hoặc không yêu cầu kinh nghiệm / sinh viên mới ra trường → "Fresher"
- 1-3 năm → "Junior"
- 3-5 năm → "Middle"
- 5+ năm hoặc có từ Senior/Lead → "Senior"
- Có từ Lead rõ ràng → "Lead"
- Nếu không rõ → null

Giá trị hợp lệ:
"Intern" | "Fresher" | "Junior" | "Middle" | "Senior" | "Lead" | null.

### job_title
Suy luận từ nội dung nếu JD không ghi rõ.
Dùng tên chuẩn ngành tiếng Anh.
Ví dụ: "Backend Engineer", "Frontend Developer", "Data Engineer", "DevOps Engineer", "IT Support Officer".

### domain / sub_domain
domain: lĩnh vực lớn, mặc định "IT".
sub_domain: lĩnh vực con nếu xác định được.
Ví dụ: "Mobile Development", "Data Engineering", "AI/ML", "IT Support", "Cybersecurity", "Frontend", "Backend", "Fullstack", "QA/Testing".

### keywords
- Tối đa 15 từ khóa quan trọng nhất cho CV-JD Matching.
- Ưu tiên tên công nghệ/kỹ năng cụ thể.
- Không đưa động từ hành động quá chung như "cài đặt", "bảo trì", "quản lý", "triển khai".
- Không đưa từ quá chung như "system", "management", "developer", "engineer".

## Thông tin cố định
jd_id = "{jd_id}"
source_file = "{source_file}"

## Schema output
{{
  "jd_id": "{jd_id}",
  "source_file": "{source_file}",
  "job_title": string | null,
  "domain": string | null,
  "sub_domain": string | null,
  "level": string | null,
  "responsibilities": string[],
  "must_have_skills": string[],
  "strong_preferred_skills": string[],
  "nice_to_have_skills": string[],
  "tools_platforms": string[],
  "databases": string[],
  "devops_cloud": string[],
  "soft_skills": string[],
  "language_requirement": string[],
  "experience_required": {{
    "min_years": number | null,
    "max_years": number | null,
    "description": string | null
  }},
  "education_requirement": {{
    "min_degree": "Trung cấp" | "Cao đẳng" | "Đại học" | "Thạc sĩ" | "Tiến sĩ" | null,
    "majors": string[],
    "notes": string | null
  }},
  "salary_range": string | null,
  "working_time": string | null,
  "keywords": string[]
}}

## JD raw
\"\"\"
{jd_raw}
\"\"\"
""".strip()


# ---------------------------------------------------------------------------
# Post-processing & validation
# ---------------------------------------------------------------------------

SKILL_PRIORITY_FIELDS = [
    "databases",
    "devops_cloud",
    "tools_platforms",
    "must_have_skills",
    "strong_preferred_skills",
    "nice_to_have_skills",
]


def deduplicate_list(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []

    for item in items:
        normalized = normalize_tech_terms(str(item).strip())

        if not normalized:
            continue

        key = normalized.lower()

        if key not in seen:
            seen.add(key)
            result.append(normalized)

    return result


def cross_field_deduplicate(data: Dict[str, Any]) -> Dict[str, Any]:
    global_seen: Set[str] = set()

    for field in SKILL_PRIORITY_FIELDS:
        cleaned: List[str] = []

        for item in data.get(field, []):
            key = str(item).lower().strip()

            if key and key not in global_seen:
                global_seen.add(key)
                cleaned.append(item)

        data[field] = cleaned

    return data


def normalize_nested_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    experience = data.get("experience_required") or {}
    education = data.get("education_requirement") or {}

    data["experience_required"] = {
        "min_years": experience.get("min_years"),
        "max_years": experience.get("max_years"),
        "description": experience.get("description"),
    }

    data["education_requirement"] = {
        "min_degree": education.get("min_degree"),
        "majors": deduplicate_list(education.get("majors") or []),
        "notes": education.get("notes"),
    }

    return data


def postprocess_jd(
    jd: ParsedJD,
    jd_id: str,
    source_file: str,
    raw_text: str,
) -> Dict[str, Any]:
    data = jd.model_dump()

    data["jd_id"] = jd_id
    data["source_file"] = source_file
    data["domain"] = data.get("domain") or "IT"

    list_fields = [
        "responsibilities",
        "must_have_skills",
        "strong_preferred_skills",
        "nice_to_have_skills",
        "tools_platforms",
        "databases",
        "devops_cloud",
        "soft_skills",
        "language_requirement",
        "keywords",
    ]

    for field in list_fields:
        data[field] = deduplicate_list(data.get(field) or [])

    data = cross_field_deduplicate(data)
    data = normalize_nested_fields(data)

    for str_field in (
        "job_title",
        "domain",
        "sub_domain",
        "level",
        "salary_range",
        "working_time",
    ):
        if data.get(str_field):
            data[str_field] = normalize_tech_terms(str(data[str_field]).strip())

    data["raw_text"] = raw_text

    return data


def validate_parsed_jd(data: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []

    if not data.get("job_title"):
        issues.append("missing_job_title")

    if not data.get("responsibilities"):
        issues.append("missing_responsibilities")

    if len(data.get("must_have_skills", [])) < 1:
        issues.append("too_few_must_have_skills")

    if not data.get("keywords"):
        issues.append("missing_keywords")

    responsibilities = data.get("responsibilities", [])
    if len(responsibilities) > 12:
        issues.append("too_many_responsibilities")

    total_skills = sum(
        len(data.get(f, []))
        for f in (
            "must_have_skills",
            "strong_preferred_skills",
            "nice_to_have_skills",
            "tools_platforms",
            "databases",
            "devops_cloud",
        )
    )

    if total_skills > 35:
        issues.append("too_many_skills_possible_hallucination")

    experience = data.get("experience_required") or {}
    education = data.get("education_requirement") or {}

    if (
        experience.get("min_years") is None
        and experience.get("max_years") is None
        and not experience.get("description")
    ):
        issues.append("missing_experience_required")

    if not education.get("min_degree") and not education.get("majors"):
        issues.append("missing_education_requirement")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Gemini integration
# ---------------------------------------------------------------------------

def parse_response_to_model(response: Any) -> ParsedJD:
    parsed_obj = getattr(response, "parsed", None)

    if isinstance(parsed_obj, ParsedJD):
        return parsed_obj

    if isinstance(parsed_obj, dict):
        return ParsedJD.model_validate(parsed_obj)

    raw_text = (getattr(response, "text", None) or "").strip()

    if not raw_text:
        raise ValueError("Gemini response is empty.")

    return ParsedJD.model_validate_json(raw_text)


def generate_with_gemini(
    client: genai.Client,
    model: str,
    prompt: str,
) -> ParsedJD:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=ParsedJD,
            ),
        )

        return parse_response_to_model(response)

    except Exception as schema_err:
        logger.warning(
            "Structured-schema mode failed (%s). Falling back to plain JSON mode.",
            schema_err,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
            ),
        )

        return parse_response_to_model(response)


def parse_one_jd_with_gemini(
    client: genai.Client,
    file_path: Path,
    jd_id: str,
    model: str,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_text = clean_jd_text(raw_text)

    prompt = build_prompt(
        jd_raw=cleaned_text,
        jd_id=jd_id,
        source_file=str(file_path),
    )

    last_error: Optional[str] = None

    for attempt in range(max_retries + 1):
        try:
            parsed_jd = generate_with_gemini(
                client=client,
                model=model,
                prompt=prompt,
            )

            data = postprocess_jd(
                jd=parsed_jd,
                jd_id=jd_id,
                source_file=str(file_path),
                raw_text=cleaned_text,
            )

            validation = validate_parsed_jd(data)

            return {
                "status": "ok" if validation["is_valid"] else "warning",
                "jd": data,
                "validation": validation,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)

            if attempt < max_retries:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                    attempt + 1,
                    max_retries + 1,
                    file_path.name,
                    last_error,
                    delay,
                )
                time.sleep(delay)

    logger.error("All attempts failed for %s: %s", file_path.name, last_error)

    return {
        "status": "failed",
        "jd": {
            "jd_id": jd_id,
            "source_file": str(file_path),
            "raw_text": cleaned_text,
        },
        "validation": {
            "is_valid": False,
            "issues": ["gemini_parse_failed"],
        },
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse raw JD files into structured JSON using Gemini."
    )

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jd_ids that already exist in the output file.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=MIN_DELAY_SECONDS,
        help="Minimum seconds to wait between API calls.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_jsonl = output_dir / "parsed_jd.jsonl"
    output_report = output_dir / "parsed_jd_report.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)

    jd_files = collect_jd_files(input_dir)

    if not jd_files:
        logger.error("Không tìm thấy file JD trong: %s", input_dir.resolve())
        return

    already_parsed: Set[str] = set()

    if args.resume:
        already_parsed = load_already_parsed_ids(output_jsonl)
        logger.info("Resume mode: %d JD đã được parse trước đó.", len(already_parsed))

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY chưa được set. Kiểm tra file .env ở project root.")
        return

    client = genai.Client(api_key=api_key)

    logger.info("Tìm thấy %d file JD.", len(jd_files))
    logger.info("Input : %s", input_dir.resolve())
    logger.info("Output: %s", output_dir.resolve())
    logger.info("Model : %s", args.model)

    ok_count = 0
    warning_count = 0
    failed_count = 0
    skipped_count = 0

    open_mode = "a" if args.resume else "w"

    with (
        open(output_jsonl, open_mode, encoding="utf-8") as data_f,
        open(output_report, open_mode, encoding="utf-8") as report_f,
    ):
        for idx, file_path in enumerate(tqdm(jd_files, desc="Parsing JDs"), start=1):
            jd_id = safe_jd_id(file_path, idx)

            if jd_id in already_parsed:
                skipped_count += 1
                continue

            result = parse_one_jd_with_gemini(
                client=client,
                file_path=file_path,
                jd_id=jd_id,
                model=args.model,
            )

            status = result["status"]

            if status == "ok":
                ok_count += 1
            elif status == "warning":
                warning_count += 1
            else:
                failed_count += 1

            jd_record = result["jd"]

            report_record = {
                "jd_id": jd_record.get("jd_id"),
                "source_file": str(file_path),
                "status": status,
                "validation": result["validation"],
                "error": result["error"],
            }

            data_f.write(json.dumps(jd_record, ensure_ascii=False) + "\n")
            report_f.write(json.dumps(report_record, ensure_ascii=False) + "\n")

            time.sleep(args.delay)

    logger.info("Hoàn tất parse JD.")
    logger.info("OK      : %d", ok_count)
    logger.info("Warning : %d", warning_count)
    logger.info("Failed  : %d", failed_count)
    logger.info("Skipped : %d", skipped_count)
    logger.info("Output  : %s", output_jsonl.resolve())
    logger.info("Report  : %s", output_report.resolve())


if __name__ == "__main__":
    main()