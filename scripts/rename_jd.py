from __future__ import annotations

from pathlib import Path
import uuid



PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = [
	PROJECT_ROOT / "data" / "jd" / "raw"
]
PDF_SUFFIX = ".txt"


def collect_pdfs(source_dirs: list[Path]) -> list[Path]:
	pdf_files: list[Path] = []

	for source_dir in source_dirs:
		if not source_dir.exists():
			print(f"[WARN] Không tìm thấy thư mục: {source_dir}")
			continue

		pdf_files.extend(sorted(source_dir.glob(f"*{PDF_SUFFIX}")))

	return sorted(pdf_files, key=lambda path: (str(path.parent).lower(), path.name.lower()))


def rename_txt_to_jd(source_dirs: list[Path], start_index: int = 1) -> None:
	pdf_files = collect_pdfs(source_dirs)

	if not pdf_files:
		print("Không tìm thấy file TXT nào để đổi tên.")
		return

	temp_targets: list[tuple[Path, Path]] = []
	for index, pdf_path in enumerate(pdf_files, start=start_index):
		temp_name = f"__tmp_jd_{uuid.uuid4().hex}_{index}{pdf_path.suffix}"
		temp_targets.append((pdf_path, pdf_path.with_name(temp_name)))

	for original_path, temp_path in temp_targets:
		original_path.rename(temp_path)

	for index, (_, temp_path) in enumerate(temp_targets, start=start_index):
		final_name = f"JD_{index:03d}{temp_path.suffix.lower()}"
		final_path = temp_path.with_name(final_name)

		if final_path.exists():
			raise FileExistsError(f"File đích đã tồn tại: {final_path}")

		temp_path.rename(final_path)
		print(f"{temp_path.name} -> {final_path.name}")


def main() -> None:
	rename_txt_to_jd(SOURCE_DIRS)


if __name__ == "__main__":
	main()
