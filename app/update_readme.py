"""
Update README.md Leaderboard from aggregated metrics in output/<prompt>/<locale>/<model>/<split>/<subject>/result.json.

- Locales come from prompts/ (e.g., prompts/en, prompts/ko)
- Base path: prefer output/mcqa; fallback to output/test
- For each locale, aggregate per model & per subject across all splits (weighted by num_example)
- Render per-locale HTML tables with columns: Rank, Model, and each subject's accuracy
- Rank within each locale by overall weighted accuracy across subjects

Usage:
	uv run python -m app.update_readme
	# or
	uv run python app/update_readme.py
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
README_PATH = os.path.join(REPO_ROOT, "README.md")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", os.path.join(REPO_ROOT, "output"))
PROMPTS_DIR = os.path.join(REPO_ROOT, "prompts")

KNOWN_LOCALES = {"en", "ko", "zh", "ja", "es", "fr", "de", "it", "pt", "ru"}


@dataclass
class SubjectAgg:
	acc_sum: float = 0.0
	n: int = 0

	@property
	def acc(self) -> float:
		return 0.0 if self.n <= 0 else self.acc_sum / self.n


@dataclass
class ModelAgg:
	model: str
	locale: str
	subjects: Dict[str, SubjectAgg] = field(default_factory=dict)

	@property
	def overall_acc(self) -> float:
		total_sum = sum(sa.acc_sum for sa in self.subjects.values())
		total_n = sum(sa.n for sa in self.subjects.values())
		return 0.0 if total_n == 0 else total_sum / total_n


def iter_result_files(base_dir: str):
	for root, _dirs, files in os.walk(base_dir):
		if "result.json" in files:
			yield os.path.join(root, "result.json")


def parse_path_components(base_prompt_dir: str, result_path: str) -> Optional[Tuple[str, str, str, str]]:
	"""Return (locale, model, split, subject) for a result.json path under base_prompt_dir.

	Expected pattern: <base_prompt_dir>/<locale>/<model>/<split>/<subject>/result.json
	If the locale segment is missing, skip (we only aggregate per-locale).
	"""
	rel = os.path.relpath(result_path, base_prompt_dir)
	parts = rel.replace("\\", "/").split("/")
	if len(parts) < 5:
		return None
	# Require the standard with-locale pattern
	locale = parts[0]
	model = parts[1]
	split = parts[2]
	subject = parts[3]
	return locale, model, split, subject


def load_result(path: str) -> Optional[Dict]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception:
		return None


def aggregate_per_locale(base_prompt_dir: str, locales: List[str]) -> Dict[str, Dict[str, ModelAgg]]:
	"""Return mapping locale -> model -> ModelAgg with per-subject weighted accuracies.

	We combine all splits for a given (locale, model, subject) by weighting with num_example.
	"""
	results: Dict[str, Dict[str, ModelAgg]] = {}
	for res_file in iter_result_files(base_prompt_dir):
		parsed = parse_path_components(base_prompt_dir, res_file)
		if not parsed:
			continue
		locale, model, _split, subject = parsed
		if locale not in locales:
			continue
		data = load_result(res_file)
		if not data:
			continue
		acc = float(data.get("acc", 0.0))
		n = int(data.get("num_example", 0))

		loc_bucket = results.setdefault(locale, {})
		mod_agg = loc_bucket.setdefault(model, ModelAgg(model=model, locale=locale))
		subj_agg = mod_agg.subjects.setdefault(subject, SubjectAgg())
		subj_agg.acc_sum += acc * n
		subj_agg.n += n

	return results


def format_leaderboard_per_locale(agg: Dict[str, Dict[str, ModelAgg]]) -> str:
	out: List[str] = []
	out.append("## Leaderboard\n\n")
	out.append("<p>Per-locale rankings aggregated across splits. Columns show subject accuracies. Higher is better.</p>\n\n")

	for locale in sorted(agg.keys()):
		models = agg[locale]
		if not models:
			continue

		# Collect union of subjects for this locale
		subjects = sorted({s for m in models.values() for s in m.subjects.keys()})
		out.append(f"### Locale: {locale}\n\n")
		if not subjects:
			out.append("<p><em>No results found.</em></p>\n\n")
			continue

		# Rank by overall accuracy
		ranked = sorted(models.values(), key=lambda m: m.overall_acc, reverse=True)

		# Build HTML table
		out.append("<table>\n<thead>\n<tr>")
		out.append("<th>Rank</th><th>Model</th>")
		for s in subjects:
			out.append(f"<th>{s}</th>")
		out.append("</tr>\n</thead>\n<tbody>\n")

		for i, m in enumerate(ranked, start=1):
			out.append("<tr>")
			out.append(f"<td>{i}</td><td>{m.model}</td>")
			for subj in subjects:
				sa = m.subjects.get(subj)
				cell = f"{sa.acc*100:.2f}%" if sa and sa.n > 0 else "-"
				out.append(f"<td>{cell}</td>")
			out.append("</tr>\n")

		out.append("</tbody>\n</table>\n\n")

	return "".join(out)


def replace_leaderboard_section(readme_text: str, new_section_md: str) -> str:
	lines = readme_text.splitlines(keepends=True)
	start_idx = None
	end_idx = None

	for idx, line in enumerate(lines):
		if line.strip().lower().startswith("## leaderboard"):
			start_idx = idx
			break

	if start_idx is None:
		# No section yet; insert after title if present, else prepend
		insert_at = 1 if lines and lines[0].startswith("# ") else 0
		return "".join(lines[:insert_at] + ["\n", new_section_md, "\n"] + lines[insert_at:])

	# find next '## ' level heading as end
	for idx in range(start_idx + 1, len(lines)):
		if lines[idx].startswith("## "):
			end_idx = idx
			break

	if end_idx is None:
		# Replace from start to end of file
		return "".join(lines[:start_idx] + [new_section_md, "\n"])

	return "".join(lines[:start_idx] + [new_section_md, "\n"] + lines[end_idx:])


def ensure_project_header(readme_text: str) -> str:
	"""If the header looks like another project's template, replace it with KoCEM header."""
	lines = readme_text.splitlines(keepends=True)
	if lines and lines[0].strip().startswith("# "):
		# Always set to KoCEM title
		lines[0] = "# KoCEM: A Multimodal Knowledge and Reasoning Benchmark for Korean Construction Engineering & Management\n"
		return "".join(lines)
	return "# KoCEM: A Multimodal Knowledge and Reasoning Benchmark for Korean Construction Engineering & Management\n\n" + readme_text


def get_locales_from_prompts(prompts_dir: str) -> List[str]:
	if not os.path.isdir(prompts_dir):
		return []
	return [d for d in os.listdir(prompts_dir) if os.path.isdir(os.path.join(prompts_dir, d))]


def update_readme() -> int:
	if not os.path.isdir(OUTPUT_PATH):
		print(f"No output directory found at {OUTPUT_PATH}")
		return 1

	# Choose base prompt folder: mcqa preferred, else test
	mcqa_dir = os.path.join(OUTPUT_PATH, "mcqa")
	test_dir = os.path.join(OUTPUT_PATH, "test")
	if os.path.isdir(mcqa_dir):
		base_prompt_dir = mcqa_dir
	elif os.path.isdir(test_dir):
		base_prompt_dir = test_dir
	else:
		print("Neither output/mcqa nor output/test found; nothing to update.")
		return 1

	locales = get_locales_from_prompts(PROMPTS_DIR)
	if not locales:
		print("No locales found under prompts/.")
		return 1

	agg = aggregate_per_locale(base_prompt_dir, locales)
	if not any(agg.values()):
		print("No result.json files found for detected locales; nothing to update.")
		return 1

	leaderboard_md = format_leaderboard_per_locale(agg)

	try:
		with open(README_PATH, "r", encoding="utf-8") as f:
			current = f.read()
	except FileNotFoundError:
		current = ""

	current = ensure_project_header(current)
	updated = replace_leaderboard_section(current, leaderboard_md)

	with open(README_PATH, "w", encoding="utf-8") as f:
		f.write(updated)

	print("README.md Leaderboard updated.")
	return 0


if __name__ == "__main__":
	from fire import Fire
	Fire(update_readme)
	
__all__ = [
    "update_readme"
]