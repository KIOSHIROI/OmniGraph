from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def log_alignment_step_metrics(
    module: Any,
    *,
    phase: str,
    loss: Any,
    metrics: Mapping[str, Any],
    batch_size: int,
    include_xgv: bool = False,
) -> None:
    phase_norm = str(phase).strip().lower()
    if phase_norm not in {"train", "val"}:
        raise ValueError(f"Unsupported phase for logging: {phase}")

    on_step = phase_norm == "train"
    prefix = phase_norm
    module.log(
        f"{prefix}_loss",
        loss,
        prog_bar=True,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    module.log(
        f"{prefix}_loss_lm",
        metrics["loss_lm"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    module.log(
        f"{prefix}_loss_aux",
        metrics["loss_aux"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    module.log(
        f"{prefix}_loss_xtc",
        metrics["loss_xtc"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    module.log(
        f"{prefix}_loss_xtm",
        metrics["loss_xtm"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    if include_xgv:
        module.log(
            f"{prefix}_loss_xgv",
            metrics["loss_xgv"],
            prog_bar=False,
            on_step=on_step,
            on_epoch=True,
            batch_size=int(batch_size),
        )
    module.log(
        f"{prefix}_xtm_acc",
        metrics["xtm_acc"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )
    module.log(
        f"{prefix}_xtm_valid_neg_ratio",
        metrics["xtm_valid_neg_ratio"],
        prog_bar=False,
        on_step=on_step,
        on_epoch=True,
        batch_size=int(batch_size),
    )


def build_stage_meta(
    *,
    stage: str,
    num_obj: int,
    num_attr: int,
    base_scene_graphs: str,
    extra_scene_graphs: Sequence[str],
    merge_stats: Mapping[str, Any],
    pseudo_graph_qa_max_per_image: int,
    pseudo_graph_qa_repeat: int,
    provenance_key: str,
    provenance_value: Any,
    graph_tokenizer_config: Mapping[str, Any],
    node_encoder_config: Mapping[str, Any],
    architecture_config: Mapping[str, Any],
    alignment_config: Mapping[str, Any],
    xtm_stats_summary: Mapping[str, Any],
    best_ckpt: str,
    export_path: str,
    extra_fields: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "stage": str(stage),
        "num_obj": int(num_obj),
        "num_attr": int(num_attr),
        "scene_graph_sources": {
            "base_scene_graphs": str(base_scene_graphs),
            "extra_scene_graphs": list(extra_scene_graphs),
            "merge_stats": dict(merge_stats),
        },
        "pseudo_graph_qa_config": {
            "max_per_image": int(pseudo_graph_qa_max_per_image),
            "repeat": int(pseudo_graph_qa_repeat),
        },
        str(provenance_key): provenance_value,
        "graph_tokenizer_config": dict(graph_tokenizer_config),
        "node_encoder_config": dict(node_encoder_config),
        "architecture_config": dict(architecture_config),
        "alignment_config": dict(alignment_config),
        "xtm_stats_summary": dict(xtm_stats_summary),
        "best_ckpt": str(best_ckpt or ""),
        "export_path": str(export_path),
    }
    if extra_fields:
        meta.update(dict(extra_fields))
    return meta
