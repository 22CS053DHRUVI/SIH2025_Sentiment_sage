import os
import tarfile
from pathlib import Path

SUPPORTED_EXTS = [".parquet", ".json", ".jsonl", ".csv"]


def extract_all(archives_dir: str = ".", out_dir: str = "data/fcc") -> None:
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	for name in ["fcc.tar.gz", "attachments.tar.gz", "search.tar.gz"]:
		arc_path = Path(archives_dir) / name
		if not arc_path.exists():
			print(f"Skip: {name} not found in {archives_dir}")
			continue
		print(f"Extracting {arc_path} -> {out_dir}")
		with tarfile.open(arc_path, "r:gz") as tar:
			tar.extractall(out_dir)
	print("Extraction complete.")


def probe_main_table(root: str = "data/fcc") -> None:
	root_p = Path(root)
	candidates = []
	for p in root_p.rglob("*"):
		if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
			candidates.append(p)
	candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
	if not candidates:
		print("No candidate table files (.parquet/.json/.jsonl/.csv) found.")
		return
	print("Top candidate files (by size):")
	for p in candidates[:10]:
		print(f" - {p}  ({p.stat().st_size/1e6:.1f} MB)")
	print("\nUpdate configs/default.yaml accordingly:")
	print(" - paths.data_dir: directory containing the chosen file")
	print(" - data.source: local_parquet | local_json | csv")
	print(" - data.text_field: actual text column (e.g., 'text', 'body', 'comment')")


def main():
	extract_all()
	probe_main_table()


if __name__ == "__main__":
	main()
