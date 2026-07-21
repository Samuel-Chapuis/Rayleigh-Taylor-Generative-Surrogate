from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Mapping


class Logger:
	"""
	Logger d'experience ecrivant a la fois un fichier texte et un CSV de metriques.

	Le fichier texte conserve les messages chronologiques. Le CSV stocke les
	resumes d'epoque et peut etendre automatiquement son en-tete si de nouvelles
	metriques apparaissent.
	"""

	def __init__(self, log_path: str | Path, csv_path: str | Path | None = None, name: str = "training_logger"):
		"""
		Initialise les fichiers de log et configure le logger Python.

		Args:
			log_path: Chemin du fichier texte.
			csv_path: Chemin du fichier CSV. Si ``None``, remplace l'extension de
				``log_path`` par ``.csv``.
			name: Nom du logger Python sous-jacent.
		"""
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
		"""Ecrit un message de niveau info dans le log texte."""
		self.logger.info(message)

	def warning(self, message: str) -> None:
		"""Ecrit un message de niveau warning dans le log texte."""
		self.logger.warning(message)

	def error(self, message: str) -> None:
		"""Ecrit un message de niveau error dans le log texte."""
		self.logger.error(message)

	def log_experiment_start(self, parameters: Mapping[str, Any]) -> None:
		"""Journalise le debut d'une experience et ses parametres."""
		self.info("Experiment started")
		for key, value in parameters.items():
			self.info(f"{key}: {value}")

	def save_config(self, parameters: Mapping[str, Any], config_path: str | Path) -> None:
		"""Sauvegarde une configuration JSON serialisable."""
		config_path = Path(config_path)
		config_path.parent.mkdir(parents=True, exist_ok=True)
		with config_path.open("w", encoding="utf-8") as config_file:
			json.dump(self._json_ready(parameters), config_file, indent=2)
			config_file.write("\n")
		self.info(f"Config saved to {config_path}")

	def log_epoch(self, epoch: int, metrics: Mapping[str, Any]) -> None:
		"""Ajoute les metriques d'une epoque au CSV et au log texte."""
		row = {"epoch": epoch, **metrics}
		self._append_csv_row(row)
		metric_string = ", ".join(f"{key}={value}" for key, value in row.items())
		self.info(f"Epoch summary: {metric_string}")

	def log_experiment_end(self, message: str = "Experiment finished") -> None:
		"""Journalise la fin d'une experience."""
		self.info(message)

	def _append_csv_row(self, row: Mapping[str, Any]) -> None:
		"""Ajoute une ligne au CSV en etendant l'en-tete si necessaire."""
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
		"""Lit l'en-tete du CSV existant, s'il existe."""
		with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
			reader = csv.reader(csv_file)
			return next(reader, None)

	def _rewrite_csv_with_header(self, fieldnames: list[str]) -> None:
		"""Reecrit le CSV avec un nouvel ordre de colonnes."""
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

	def _json_ready(self, value: Any) -> Any:
		"""Convertit recursivement une valeur en structure compatible JSON."""
		if isinstance(value, Mapping):
			return {str(key): self._json_ready(item) for key, item in value.items()}
		if isinstance(value, (list, tuple)):
			return [self._json_ready(item) for item in value]
		if isinstance(value, Path):
			return str(value)
		if isinstance(value, (str, int, float, bool)) or value is None:
			return value
		return str(value)
