import copy
import json
import math
import os
import random
import textwrap
from datetime import datetime, timezone
from html import escape
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

    def record_individual_evaluation( # TODO this should not include nn.individual.py stuff in this file, but we should create an inherited class of this in nn package which does
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

        # Extract loss_curve before sanitizing metrics
        loss_curve = None
        if hasattr(individual, "evaluation_metrics"):
            metrics_copy = copy.deepcopy(individual.evaluation_metrics)
            loss_curve = metrics_copy.pop("loss_curve", None)
            evaluation_record["metrics"] = self._sanitize(metrics_copy)
            
            # Save loss curve to separate file if it exists
            if loss_curve:
                saved_path = self._save_loss_curve(individual.id, generation, loss_curve)
                entry.setdefault("artifacts", {})
                entry["artifacts"]["loss_curve"] = self._relative_path(saved_path)

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
        self._render_lineage_svg()
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

    def _save_loss_curve(self, individual_id: int, generation: int, loss_curve: list) -> str:
        """Save loss curve data to a separate JSON file under a subfolder."""
        filename = f"loss_curve_ind{individual_id}_gen{generation}.json"
        curves_dir = os.path.join(self.experiment_path, "loss_curves")
        os.makedirs(curves_dir, exist_ok=True)
        filepath = os.path.join(curves_dir, filename)
        temp_filepath = filepath + ".tmp"
        
        loss_curve_data = {
            "individual_id": individual_id,
            "generation": generation,
            "timestamp": _isoformat(_utc_now()),
            "loss_curve": self._sanitize(loss_curve),
        }
        
        with open(temp_filepath, "w", encoding="utf-8") as f:
            json.dump(loss_curve_data, f, indent=2, sort_keys=False)
        os.replace(temp_filepath, filepath)
        return filepath

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

    def _render_lineage_svg(self) -> None:
        """Render a lightweight lineage SVG for the entire run."""
        individuals = self.data.get("individuals", {})
        if not individuals:
            return

        generation_nodes: Dict[int, list[int]] = {}
        for entry in individuals.values():
            creation = entry.get("creation")
            if not creation:
                continue
            generation = creation.get("generation")
            if generation is None:
                continue
            try:
                individual_id = int(entry["id"])
            except (TypeError, ValueError):
                continue
            generation_nodes.setdefault(int(generation), []).append(individual_id)

        if not generation_nodes:
            return

        generations = sorted(generation_nodes.keys())
        max_nodes = max(len(nodes) for nodes in generation_nodes.values())
        if max_nodes == 0:
            return

        margin_x, margin_y = 60, 60
        x_gap, y_gap, node_radius = 120, 140, 22
        svg_width = max(320, margin_x * 2 + (max_nodes - 1) * x_gap)
        svg_height = margin_y * 2 + (len(generations) - 1) * y_gap

        def _jitter(seed: int) -> float:
            """Deterministic jitter value in [-0.5, 0.5] derived via random.Random."""
            rng = random.Random(seed)
            return rng.uniform(-0.5, 0.5)

        node_positions: Dict[int, tuple[float, float]] = {}
        for row, generation in enumerate(generations):
            nodes = sorted(generation_nodes[generation])
            span = (len(nodes) - 1) * x_gap if len(nodes) > 1 else 0
            start_x = (svg_width - span) / 2
            y_pos = margin_y + row * y_gap
            for idx, node_id in enumerate(nodes):
                x_pos = start_x + idx * x_gap
                jitter = _jitter(node_id) * (x_gap * 0.35)
                x_pos = max(margin_x, min(svg_width - margin_x, x_pos + jitter))
                node_positions[node_id] = (x_pos, y_pos)

        final_generation_ids: set[int] = set()
        if self.data.get("generations"):
            final_snapshot = self.data["generations"][-1].get("population_snapshot", [])
            for snapshot in final_snapshot:
                node_id = snapshot.get("id")
                if node_id is not None:
                    try:
                        final_generation_ids.add(int(node_id))
                    except (TypeError, ValueError):
                        continue

        def build_label(strategy: Optional[str], operations: Iterable[Dict[str, Any]], parent_idx: int, parent_id: int) -> str:
            names: list[str] = []
            for op in operations:
                op_name = op.get("name") or op.get("type")
                if not op_name:
                    continue
                op_parent = op.get("with_parent_id")
                if op_parent is not None:
                    try:
                        if int(op_parent) == parent_id:
                            names.append(str(op_name))
                    except (TypeError, ValueError):
                        continue
                elif parent_idx == 0:
                    names.append(str(op_name))
            if not names and strategy:
                names.append(strategy)
            label = ", ".join(names) if names else "lineage"
            return label if len(label) <= 48 else f"{label[:45]}..."

        def format_node_operation_lines(strategy: Optional[str], operations: Iterable[Dict[str, Any]], max_chars: int = 24, max_lines: int = 2) -> list[str]:
            names: list[str] = []
            for op in operations:
                op_name = op.get("name") or op.get("type")
                if not op_name:
                    continue
                names.append(str(op_name))
            if not names and strategy:
                names = [strategy]
            if not names:
                return []
            combined = ", ".join(names)
            wrapped = textwrap.wrap(combined, width=max_chars)
            if len(wrapped) > max_lines:
                wrapped = wrapped[:max_lines]
                if not wrapped[-1].endswith("..."):
                    trimmed = wrapped[-1][:max(0, max_chars - 3)].rstrip(", ")
                    wrapped[-1] = (trimmed if trimmed else "") + "..."
            return wrapped

        edges: list[tuple[int, int, str, int]] = []  # (parent_id, child_id, label, parent_idx)
        node_operation_lines: Dict[int, list[str]] = {}
        for entry in individuals.values():
            creation = entry.get("creation")
            if not creation:
                continue
            try:
                child_id = int(entry["id"])
            except (TypeError, ValueError):
                continue
            if child_id not in node_positions:
                continue
            parents = creation.get("parents") or []
            strategy = creation.get("strategy")
            operations = creation.get("operations") or []
            # Attach readable labels under the node itself
            node_operation_lines[child_id] = format_node_operation_lines(strategy, operations)
            if strategy == "mutation":  # NOTE: backwards compatibility for experiments 1-4 that had bug
                parents = [parents[0]]
            for idx, parent_id in enumerate(parents):
                if parent_id not in node_positions:
                    continue
                label = build_label(strategy, operations, idx, parent_id)
                edges.append((parent_id, child_id, label or "lineage", idx))

        if not node_positions:
            return

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(svg_width)}" height="{int(svg_height)}" viewBox="0 0 {int(svg_width)} {int(svg_height)}" font-family="Arial, sans-serif">',
            "  <defs>",
            '    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" fill="#666">',
            '      <polygon points="0 0, 10 3.5, 0 7" />',
            "    </marker>",
            '    <marker id="arrowhead-primary" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" fill="#4c8bf5">',
            '      <polygon points="0 0, 10 3.5, 0 7" />',
            "    </marker>",
            "  </defs>",
        ]

        for parent_id, child_id, label, parent_idx in edges:
            x1, y1 = node_positions[parent_id]
            x2, y2 = node_positions[child_id]
            # Shorten line to stop at circle edge so arrowhead is visible
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                x2 -= (dx / dist) * node_radius
                y2 -= (dy / dist) * node_radius
            # First parent (primary) gets blue arrow, others get gray
            if parent_idx == 0:
                stroke_color, marker_id = "#4c8bf5", "arrowhead-primary"
            else:
                stroke_color, marker_id = "#888", "arrowhead"
            svg_lines.append(
                f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke_color}" stroke-width="1" marker-end="url(#{marker_id})">'
            )
            svg_lines.append(f'    <title>{escape(label)}</title>')
            svg_lines.append("  </line>")

        nodes_with_children = {parent_id for parent_id, _, _, _ in edges}
        for node_id, (x_pos, y_pos) in node_positions.items():
            highlight = node_id in final_generation_ids
            has_children = node_id in nodes_with_children
            if highlight:
                fill, stroke, text_color = "#4c8bf5", "#2457d3", "#fff"
            elif not has_children:
                fill, stroke, text_color = "#e0e0e0", "#aaa", "#666"
            else:
                fill, stroke, text_color = "#f6f8fa", "#9aa0a6", "#202124"
            svg_lines.append(
                f'  <circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="{node_radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
            )
            svg_lines.append(
                f'  <text x="{x_pos:.1f}" y="{y_pos:.1f}" font-size="12" text-anchor="middle" dominant-baseline="middle" fill="{text_color}">{escape(str(node_id))}</text>'
            )
            operation_lines = node_operation_lines.get(node_id)
            if operation_lines:
                text_y = y_pos + node_radius + 12
                svg_lines.append(
                    f'  <text x="{x_pos:.1f}" y="{text_y:.1f}" font-size="10" text-anchor="middle" fill="#444">'
                )
                svg_lines.append(f'    <tspan x="{x_pos:.1f}" dy="0">{escape(operation_lines[0])}</tspan>')
                for line in operation_lines[1:]:
                    svg_lines.append(f'    <tspan x="{x_pos:.1f}" dy="12">{escape(line)}</tspan>')
                svg_lines.append("  </text>")

        for row, generation in enumerate(generations):
            y_pos = margin_y + row * y_gap - node_radius - 12
            if y_pos < 0:
                y_pos = 12
            svg_lines.append(
                f'  <text x="{margin_x/2:.1f}" y="{y_pos:.1f}" font-size="11" fill="#555">Gen {generation}</text>'
            )

        svg_lines.append("</svg>")

        svg_path = os.path.join(self.experiment_path, "lineage.svg")
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        self.data["run"]["paths"]["lineage_graph"] = self._relative_path(svg_path)
