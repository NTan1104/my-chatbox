"""
parse_cv.py — Parse cleaned CV .md/.txt files into structured JSON using Gemini.

Improvements over v1:
- raw_text excluded from parsed output (stored separately in report)
- Skills normalization consistent: normalize before dedup key
- Prompt externalized as a structured template constant
- Response parser handles ```json fences robustly
- --dry-run flag for testing without API calls
- Cleaner CV ID generation
- Type hints modernized (X | None, list[X])
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any

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

INPUT_DIR = PROJECT_ROOT / "data" / "cv" / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "cv" / "parsed"
OUTPUT_JSONL = OUTPUT_DIR / "parsed_cv.jsonl"
OUTPUT_REPORT = OUTPUT_DIR / "parsed_cv_report.jsonl"

DEFAULT_MODEL = "gemini-2.5-flash"
SUPPORTED_EXTENSIONS = ("*.md", "*.txt")

MIN_DELAY_SECONDS = 0.5
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

SKILL_SUBFIELDS = [
    "programming_languages",
    "backend",
    "frontend",
    "database",
    "devops_cloud",
    "tools",
    "others",
]

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

_RAW_NORMALIZATION: dict[str, str] = {
    "javascript": "JavaScript",
    "java script": "JavaScript",
    "typescript": "TypeScript",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "node.js": "Node.js",
    "reactjs": "React.js",
    "react js": "React.js",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "nextjs": "Next.js",
    "next js": "Next.js",
    "nestjs": "NestJS",
    "nest.js": "NestJS",
    "expressjs": "Express.js",
    "express js": "Express.js",
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
    "gstreamer": "GStreamer",
    "opencv": "OpenCV",
    "rabbitmq": "RabbitMQ",
    "redis pub/sub": "Redis Pub/Sub",
    "websocket": "WebSocket",
    "webrtc": "WebRTC",
}

_SORTED_KEYS = sorted(_RAW_NORMALIZATION.keys(), key=len, reverse=True)
_NORM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _SORTED_KEYS) + r")\b",
    flags=re.IGNORECASE,
)
_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def normalize_tech(text: str) -> str:
    return _NORM_PATTERN.sub(lambda m: _RAW_NORMALIZATION[m.group(0).lower()], text)


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class Skills(BaseModel):
    programming_languages: list[str] = Field(default_factory=list)
    backend: list[str] = Field(default_factory=list)
    frontend: list[str] = Field(default_factory=list)
    database: list[str] = Field(default_factory=list)
    devops_cloud: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    others: list[str] = Field(default_factory=list)


class WorkExperience(BaseModel):
    company: str | None = None
    position: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    duration_months: int | None = None
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)


class Project(BaseModel):
    name: str | None = None
    role: str | None = None
    description: str | None = None
    tech_stack: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    url: str | None = None


class Education(BaseModel):
    institution: str | None = None
    degree: str | None = None
    major: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    gpa: str | None = None


class Certificate(BaseModel):
    name: str | None = None
    issuer: str | None = None
    date: str | None = None


class ParsedCV(BaseModel):
    cv_id: str
    source_file: str | None = None
    target_role: str | None = None
    level: str | None = None
    experience_years: float | None = None
    skills: Skills = Field(default_factory=Skills)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    projects: list[Project] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    certificates: list[Certificate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Bạn là hệ thống trích xuất thông tin CV ngành CNTT phục vụ CV-JD Matching.

## Nhiệm vụ
Đọc CV raw (định dạng Markdown/text) và trả về JSON object theo schema bên dưới.

## Quy tắc output
- Chỉ trả về JSON object hợp lệ. Không giải thích, không markdown code block.
- Không tự bịa thông tin không có trong CV.
- Nếu không tìm thấy: null (string/number) hoặc [] (array).

## Quy tắc từng field

### target_role
Vị trí ứng tuyển hoặc mục tiêu nghề nghiệp của ứng viên.
Lấy từ: tiêu đề CV, mục "Objective"/"Mục tiêu", hoặc suy luận từ kinh nghiệm gần nhất.
CHỈ lấy tên vị trí ngắn gọn (3–6 từ).
✓ "Backend Engineer", "Middle Web Developer", "Frontend Developer"
✗ Không copy nguyên đoạn mô tả mục tiêu nghề nghiệp.

### level
Suy luận từ tổng số năm kinh nghiệm:
  0 hoặc đang học   → "Fresher"
  < 1 năm           → "Fresher"
  1–3 năm           → "Junior"
  3–5 năm           → "Middle"
  5+ năm            → "Senior"
  CV ghi "intern"   → "Intern"
Giá trị hợp lệ: "Intern" | "Fresher" | "Junior" | "Middle" | "Senior" | null

### experience_years
Tổng số năm kinh nghiệm làm việc thực tế (không tính thời gian học).
Làm tròn đến 0.5 năm.

### skills
Mỗi item CHỈ xuất hiện ở MỘT sub-field (ưu tiên field đứng trước):

**programming_languages**: Ngôn ngữ lập trình thuần túy.
  ✓ Python, Java, C++, JavaScript, TypeScript, Go, Kotlin, Swift, PHP, Rust

**backend**: Framework/library phía server.
  ✓ Spring Boot, Django, FastAPI, Express.js, NestJS, Laravel, .NET, Flask

**frontend**: Framework/library phía client, CSS framework.
  ✓ React.js, Vue.js, Angular, Next.js, Nuxt.js, Tailwind CSS, Bootstrap

**database**: DBMS, cache, search engine.
  ✓ MySQL, PostgreSQL, MongoDB, Redis, Elasticsearch, SQLite, Oracle

**devops_cloud**: Cloud, CI/CD, container, hạ tầng.
  ✓ Docker, Kubernetes, AWS, GCP, Azure, GitHub Actions, Jenkins, Nginx

**tools**: Công cụ dev, quản lý dự án, test, thiết kế.
  ✓ Git, GitHub, GitLab, Jira, Postman, Figma, VS Code, Swagger

**others**: Kỹ thuật không thuộc các nhóm trên, bao gồm cả protocol/messaging.
  ✓ REST API, GraphQL, WebSocket, WebRTC, RabbitMQ, Kafka, Redis Pub/Sub,
    Microservices, Machine Learning, OpenCV, Agile/Scrum
  ✗ Soft skills, ngôn ngữ tự nhiên

### work_experience
Liệt kê từng vị trí, mới nhất trước.
- start_date / end_date: "MM/YYYY" hoặc "YYYY". end_date = "present" nếu đang làm.
- duration_months: số tháng làm việc tại vị trí đó.
  Nếu chỉ biết năm (không có tháng): lấy 12 tháng × số năm (không cộng thêm).
  Ví dụ: 2019–2020 = 12 tháng, 2021–2023 = 24 tháng.
- description: mô tả ngắn công việc chính (1–2 câu). Không để null nếu CV có mô tả.
- achievements: thành tích cụ thể.
  ✓ Câu có động từ hành động rõ ràng và kết quả/số liệu cụ thể.
  ✗ Câu mô tả chung chung không có kết quả ("Làm việc chăm chỉ").

### projects
Dự án cá nhân / side project / dự án nổi bật.
Mỗi dự án CHỈ tạo MỘT record duy nhất, dù CV mô tả ở nhiều dòng.
✗ Không tách một dự án thành nhiều record.

### education
- degree: Cử nhân, Kỹ sư, Thạc sĩ, Cao đẳng...
- gpa: dạng string, ví dụ "3.2/4.0".

### certificates
- name, issuer, date.

## Thông tin cố định
cv_id = "{cv_id}"
source_file = "{source_file}"

## Schema output
{{
  "cv_id": "{cv_id}",
  "source_file": "{source_file}",
  "target_role": string | null,
  "level": "Intern" | "Fresher" | "Junior" | "Middle" | "Senior" | null,
  "experience_years": number | null,
  "skills": {{
    "programming_languages": string[],
    "backend": string[],
    "frontend": string[],
    "database": string[],
    "devops_cloud": string[],
    "tools": string[],
    "others": string[]
  }},
  "work_experience": [
    {{
      "company": string | null,
      "position": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "duration_months": number | null,
      "description": string | null,
      "tech_stack": string[],
      "achievements": string[]
    }}
  ],
  "projects": [
    {{
      "name": string | null,
      "role": string | null,
      "description": string | null,
      "tech_stack": string[],
      "achievements": string[],
      "url": string | null
    }}
  ],
  "education": [
    {{
      "institution": string | null,
      "degree": string | null,
      "major": string | null,
      "start_date": string | null,
      "end_date": string | null,
      "gpa": string | null
    }}
  ],
  "certificates": [
    {{
      "name": string | null,
      "issuer": string | null,
      "date": string | null
    }}
  ]
}}

## CV raw
\"\"\"
{cv_raw}
\"\"\"\
"""


def build_prompt(cv_raw: str, cv_id: str, source_file: str) -> str:
    return _PROMPT_TEMPLATE.format(
        cv_id=cv_id,
        source_file=source_file,
        cv_raw=cv_raw,
    )


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_cv_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_tech(text.strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_cv_id(file_path: Path, index: int) -> str:
    stem = re.sub(r"[^\w]+", "_", file_path.stem, flags=re.UNICODE).strip("_").upper()
    if re.match(r"^CV_\d+$", stem):
        return stem
    return f"CV_{stem}" if stem else f"CV_{index:04d}"


def collect_cv_files(input_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(input_dir.rglob(ext))
    return sorted(set(files))


def load_parsed_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    seen: set[str] = set()
    with output_jsonl.open(encoding="utf-8") as f:
        for line in f:
            try:
                cv_id = json.loads(line).get("cv_id")
                if cv_id:
                    seen.add(cv_id)
            except json.JSONDecodeError:
                pass
    return seen


# ---------------------------------------------------------------------------
# Normalization & dedup
# ---------------------------------------------------------------------------


def dedup_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        item = normalize_tech(str(raw).strip())
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def dedup_skills_cross_field(skills: dict[str, Any]) -> dict[str, Any]:
    """Each skill appears in exactly one sub-field (first wins)."""
    global_seen: set[str] = set()
    for field in SKILL_SUBFIELDS:
        deduped: list[str] = []
        for item in skills.get(field, []):
            key = item.lower()
            if key not in global_seen:
                global_seen.add(key)
                deduped.append(item)
        skills[field] = deduped
    return skills


# ---------------------------------------------------------------------------
# Response parsing — robust against ```json fences
# ---------------------------------------------------------------------------


def extract_json_text(raw: str) -> str:
    """Strip markdown code fences if present, return raw JSON string."""
    raw = raw.strip()
    match = _JSON_FENCE_PATTERN.search(raw)
    return match.group(1).strip() if match else raw


def parse_gemini_response(response: Any) -> ParsedCV:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, ParsedCV):
        return parsed
    if isinstance(parsed, dict):
        return ParsedCV.model_validate(parsed)

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise ValueError("Empty Gemini response.")

    json_text = extract_json_text(text)
    return ParsedCV.model_validate_json(json_text)


# ---------------------------------------------------------------------------
# Post-processing & validation
# ---------------------------------------------------------------------------


def postprocess(cv: ParsedCV, cv_id: str, source_file: str) -> dict[str, Any]:
    data = cv.model_dump()
    data["cv_id"] = cv_id
    data["source_file"] = source_file

    # Normalize & dedup skills
    skills = data.get("skills", {})
    for field in SKILL_SUBFIELDS:
        skills[field] = dedup_list(skills.get(field) or [])
    data["skills"] = dedup_skills_cross_field(skills)

    # Normalize work_experience
    for exp in data.get("work_experience", []):
        exp["tech_stack"] = dedup_list(exp.get("tech_stack") or [])
        exp["achievements"] = [
            a.strip() for a in (exp.get("achievements") or []) if a and a.strip()
        ]

    # Normalize projects
    for proj in data.get("projects", []):
        proj["tech_stack"] = dedup_list(proj.get("tech_stack") or [])
        proj["achievements"] = [
            a.strip() for a in (proj.get("achievements") or []) if a and a.strip()
        ]

    return data


def validate(data: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []

    if not data.get("target_role"):
        issues.append("missing_target_role")

    skills = data.get("skills", {})
    if sum(len(skills.get(f, [])) for f in SKILL_SUBFIELDS) == 0:
        issues.append("missing_skills")

    if not data.get("work_experience") and not data.get("projects"):
        issues.append("missing_experience_and_projects")

    if not data.get("education"):
        issues.append("missing_education")

    yoe = data.get("experience_years")
    if yoe is not None and yoe > 40:
        issues.append("suspicious_experience_years")

    return {"is_valid": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------


def call_gemini(client: genai.Client, model: str, prompt: str) -> ParsedCV:
    shared_cfg = dict(temperature=0, response_mime_type="application/json")
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**shared_cfg, response_schema=ParsedCV),
        )
        return parse_gemini_response(resp)
    except Exception as e:
        logger.warning("Structured schema failed (%s). Falling back to plain JSON.", e)
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**shared_cfg),
        )
        return parse_gemini_response(resp)


def parse_one(
    client: genai.Client,
    file_path: Path,
    cv_id: str,
    model: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_cv_text(raw_text)
    source = str(file_path)

    if dry_run:
        logger.info("[dry-run] Would parse %s", file_path.name)
        return {
            "status": "dry_run",
            "cv": {"cv_id": cv_id, "source_file": source},
            "validation": {"is_valid": True, "issues": []},
            "error": None,
            "raw_text": cleaned,
        }

    prompt = build_prompt(cv_raw=cleaned, cv_id=cv_id, source_file=source)
    last_err: str | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            parsed_cv = call_gemini(client, model, prompt)
            data = postprocess(parsed_cv, cv_id, source)
            result_validation = validate(data)
            status = "ok" if result_validation["is_valid"] else "warning"
            return {
                "status": status,
                "cv": data,
                "validation": result_validation,
                "error": None,
                "raw_text": cleaned,
            }
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retry in %.1fs",
                    attempt + 1, MAX_RETRIES + 1,
                    file_path.name, last_err, delay,
                )
                time.sleep(delay)

    logger.error("All attempts failed for %s: %s", file_path.name, last_err)
    return {
        "status": "failed",
        "cv": {"cv_id": cv_id, "source_file": source},
        "validation": {"is_valid": False, "issues": ["gemini_parse_failed"]},
        "error": last_err,
        "raw_text": cleaned,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Parse cleaned CV files into structured JSON using Gemini."
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--input", default=str(INPUT_DIR))
    ap.add_argument("--output", default=str(OUTPUT_DIR))
    ap.add_argument("--resume", action="store_true", help="Skip already-parsed CV IDs.")
    ap.add_argument("--dry-run", action="store_true", help="Read files but skip API calls.")
    ap.add_argument("--delay", type=float, default=MIN_DELAY_SECONDS,
                    help="Minimum seconds between API calls.")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_jsonl = output_dir / "parsed_cv.jsonl"
    output_report = output_dir / "parsed_cv_report.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_files = collect_cv_files(input_dir)
    if not cv_files:
        logger.error("No CV files found in: %s", input_dir.resolve())
        return

    already_parsed: set[str] = set()
    if args.resume:
        already_parsed = load_parsed_ids(output_jsonl)
        logger.info("Resume mode: %d CVs already parsed.", len(already_parsed))

    client: genai.Client | None = None
    if not args.dry_run:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is not set.")
            return
        client = genai.Client(api_key=api_key)

    logger.info("Found %d CV files | model=%s | dry_run=%s", len(cv_files), args.model, args.dry_run)

    counters = {"ok": 0, "warning": 0, "failed": 0, "skipped": 0}
    open_mode = "a" if args.resume else "w"

    with (
        open(output_jsonl, open_mode, encoding="utf-8") as data_f,
        open(output_report, open_mode, encoding="utf-8") as report_f,
    ):
        for idx, file_path in enumerate(tqdm(cv_files, desc="Parsing CVs"), start=1):
            cv_id = make_cv_id(file_path, idx)

            if cv_id in already_parsed:
                counters["skipped"] += 1
                continue

            result = parse_one(
                client=client,
                file_path=file_path,
                cv_id=cv_id,
                model=args.model,
                dry_run=args.dry_run,
            )

            status = result["status"]
            counters[status] = counters.get(status, 0) + 1

            # Write parsed CV (without raw_text to save storage)
            cv_record = result["cv"]
            data_f.write(json.dumps(cv_record, ensure_ascii=False) + "\n")

            # Write report (include raw_text here for debugging)
            report_f.write(json.dumps({
                "cv_id": cv_record.get("cv_id"),
                "source_file": str(file_path),
                "status": status,
                "validation": result["validation"],
                "error": result["error"],
                "raw_text": result.get("raw_text", ""),
            }, ensure_ascii=False) + "\n")

            if not args.dry_run:
                time.sleep(args.delay)

    logger.info("Done. OK=%d | Warning=%d | Failed=%d | Skipped=%d",
                counters["ok"], counters["warning"], counters["failed"], counters["skipped"])
    logger.info("Output : %s", output_jsonl.resolve())
    logger.info("Report : %s", output_report.resolve())


if __name__ == "__main__":
    main()