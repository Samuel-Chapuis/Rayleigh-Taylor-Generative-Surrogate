from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Mapping


class Logger:
	def __init__(self, log_path: str | Path, csv_path: str | Path | None = None, name: str = "training_logger"):
		self.log_path = Path(log_path)
		self.csv_path = Path(csv_path) if csv_path is not None else self.log_path.with_suffix(".csv")

		self.log_path.parent.mkdir(parents=True, exist_ok=True)
		self.csv_path.parent.mkdir(parents=True, exist_ok=True)

		self.logger = logging.getLogger(name)
		self.logger.setLevel(logging.INFO)
		self.logger.propagate = False

		for handler in list(self.logger.handlers):
			self.logger.removeHandler(handler)
			handler.close()

		file_handler = logging.FileHandler(self.log_path, encoding="utf-8")
		file_handler.setLevel(logging.INFO)
		file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
		self.logger.addHandler(file_handler)

		self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
		self._csv_fieldnames = self._read_csv_header() if self._csv_header_written else None

	def info(self, message: str) -> None:
		self.logger.info(message)

	def warning(self, message: str) -> None:
		self.logger.warning(message)

	def error(self, message: str) -> None:
		self.logger.error(message)

	def log_experiment_start(self, parameters: Mapping[str, Any]) -> None:
		self.info("Experiment started")
		for key, value in parameters.items():
			self.info(f"{key}: {value}")

	def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
		row = {"epoch": epoch, **metrics}
		self._append_csv_row(row)
		metric_string = ", ".join(f"{key}={value}" for key, value in row.items())
		self.info(f"Epoch summary: {metric_string}")

	def log_experiment_end(self, message: str = "Experiment finished") -> None:
		self.info(message)

	def _append_csv_row(self, row: Mapping[str, Any]) -> None:
		fieldnames = self._csv_fieldnames or list(row.keys())
		missing_fields = [key for key in row.keys() if key not in fieldnames]
		if missing_fields:
			fieldnames = [*fieldnames, *missing_fields]
			self._rewrite_csv_with_header(fieldnames)

		with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			if not self._csv_header_written:
				writer.writeheader()
				self._csv_header_written = True
				self._csv_fieldnames = fieldnames
			writer.writerow(row)

	def _read_csv_header(self) -> list[str] | None:
		with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
			reader = csv.reader(csv_file)
			return next(reader, None)

	def _rewrite_csv_with_header(self, fieldnames: list[str]) -> None:
		existing_rows = []
		if self._csv_header_written:
			with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
				reader = csv.DictReader(csv_file)
				existing_rows = list(reader)

		with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
			writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerows(existing_rows)

		self._csv_header_written = True
		self._csv_fieldnames = fieldnames
