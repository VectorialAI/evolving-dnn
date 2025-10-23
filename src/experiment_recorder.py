import copy
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional


def _utc_now():
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class ExperimentRecorder:
    """Collects structured data for an evolution run and persists it as JSON."""

    def __init__(self, experiment_path: str, run_config: Dict[str, Any], record_filename: str = "experiment_summary.json"):
        self.experiment_path = experiment_path
        self.record_path = os.path.join(experiment_path, record_filename)
        self._start_time = _utc_now()
        self._generation_index: Dict[int, int] = {}

        self.data: Dict[str, Any] = {
            "run": {
                "id": self._start_time.strftime("%Y%m%dT%H%M%S"),
                "status": "running",
                "start_time": _isoformat(self._start_time),
                "paths": {"experiment": os.path.abspath(experiment_path)},
                "config": copy.deepcopy(run_config),
                "metadata": {},
            },
            "system": {},
            "generations": [],
            "individuals": {},
            "events": [],
        }

        os.makedirs(experiment_path, exist_ok=True)
        self._save()

    # ------------------------------------------------------------------ #
    # Run level metadata helpers
    # ------------------------------------------------------------------ #
    def update_run_metadata(self, **metadata: Any) -> None:
        """Attach additional metadata about the run (e.g., tokenizer path, notes)."""
        self.data["run"]["metadata"].update(self._sanitize(metadata))
        self._save()

    def update_system_info(self, info: Dict[str, Any]) -> None:
        """Record information about the host environment (e.g., Python, PyTorch versions)."""
        self.data["system"].update(self._sanitize(info))
        self._save()

    # ------------------------------------------------------------------ #
    # Individual tracking
    # ------------------------------------------------------------------ #
    def record_initial_individual(self, individual_id: int, generation: int) -> None:
        entry = self.data["individuals"].setdefault(str(individual_id), {"id": individual_id})
        if "creation" not in entry:
            entry["creation"] = self._sanitize({
                "origin": "initial_population",
                "generation": generation,
                "timestamp": _isoformat(_utc_now()),
            })
        self._save()

    def record_child_creation(
        self,
        individual_id: int,
        generation: int,
        parents: Optional[Iterable[int]],
        operations: Iterable[Dict[str, Any]],
        strategy: Optional[str] = None,
    ) -> None:
        entry = self.data["individuals"].setdefault(str(individual_id), {"id": individual_id})
        entry["creation"] = self._sanitize({
            "origin": "child",
            "generation": generation,
            "parents": list(parents) if parents is not None else [],
            "operations": list(operations),
            "strategy": strategy,
            "timestamp": _isoformat(_utc_now()),
        })
        self.data["events"].append(
            self._sanitize({
                "type": "creation",
                "individual_id": individual_id,
                "generation": generation,
                "timestamp": _isoformat(_utc_now()),
                "parents": list(parents) if parents is not None else [],
                "operations": list(operations),
                "strategy": strategy,
            })
        )
        self._save()

    def record_individual_evaluation(
        self,
        individual: Any,
        generation: int,
        artifacts: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        entry = self.data["individuals"].setdefault(str(individual.id), {"id": individual.id})

        if hasattr(individual, "train_config") and hasattr(individual.train_config, "to_dict"):
            entry.setdefault("train_config", self._sanitize(individual.train_config.to_dict()))

        if artifacts:
            entry.setdefault("artifacts", {})
            entry["artifacts"].update({k: self._relative_path(v) if v else None for k, v in artifacts.items()})

        fitness_value = None
        if individual.fitness is not None:
            fitness_value = float(individual.fitness)
            if not math.isfinite(fitness_value):
                fitness_value = None

        evaluation_record = {
            "generation": generation,
            "timestamp": _isoformat(_utc_now()),
            "fitness": fitness_value,
            "param_count": getattr(individual, "param_count", None),
        }

        if hasattr(individual, "evaluation_metrics"):
            evaluation_record["metrics"] = self._sanitize(individual.evaluation_metrics)

        if hasattr(individual, "evaluation_error"):
            evaluation_record["error"] = individual.evaluation_error

        evaluation_record = self._sanitize(evaluation_record)
        entry["evaluation"] = evaluation_record

        self.data["events"].append(
            self._sanitize({
                "type": "evaluation",
                "individual_id": individual.id,
                "generation": generation,
                "timestamp": evaluation_record["timestamp"],
                "fitness": evaluation_record["fitness"],
            })
        )
        self._save()

    # ------------------------------------------------------------------ #
    # Generation summaries
    # ------------------------------------------------------------------ #
    def record_generation(self, generation: int, summary: Dict[str, Any]) -> None:
        timestamp = _isoformat(_utc_now())
        summary_with_meta = self._sanitize({"generation": generation, "timestamp": timestamp, **summary})

        if generation in self._generation_index:
            idx = self._generation_index[generation]
            self.data["generations"][idx] = summary_with_meta
        else:
            self._generation_index[generation] = len(self.data["generations"])
            self.data["generations"].append(summary_with_meta)

        self.data["events"].append(
            self._sanitize({
                "type": "generation_summary",
                "generation": generation,
                "timestamp": timestamp,
                "max_fitness": summary.get("max_fitness"),
                "avg_fitness": summary.get("average_fitness"),
            })
        )
        self._save()

    # ------------------------------------------------------------------ #
    # Finalization
    # ------------------------------------------------------------------ #
    def finalize(self, status: str = "completed") -> None:
        end_time = _utc_now()
        self.data["run"]["status"] = status
        self.data["run"]["end_time"] = _isoformat(end_time)

        duration_seconds = (end_time - self._start_time).total_seconds()
        self.data["run"]["duration_seconds"] = duration_seconds
        self._save()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _relative_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        try:
            return os.path.relpath(path, self.experiment_path)
        except ValueError:
            return path

    def _save(self) -> None:
        temp_path = self.record_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=False)
        os.replace(temp_path, self.record_path)

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._sanitize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(v) for v in value]
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return None
        return value
