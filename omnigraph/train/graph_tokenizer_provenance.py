from __future__ import annotations

from typing import Any, Dict, Mapping


def normalize_graph_tokenizer_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    t = str(cfg.get("type", "qformer")).strip().lower()
    if t not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Unsupported graph tokenizer type in provenance: {t}")
    return {
        "type": t,
        "num_latents": int(cfg.get("num_latents", cfg.get("num_query_tokens", 32))),
        "hidden_dim": int(cfg.get("hidden_dim", cfg.get("qformer_hidden_dim", 768))),
        "num_layers": int(cfg.get("num_layers", 3)),
        "num_heads": int(cfg.get("num_heads", 8)),
        "ff_mult": int(cfg.get("ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.0)),
    }


def assert_graph_tokenizer_match(
    *,
    expected_cfg: Mapping[str, Any],
    got_cfg: Mapping[str, Any],
    stage_name: str,
) -> None:
    if str(expected_cfg.get("type")) != str(got_cfg.get("type")):
        raise RuntimeError(
            f"{stage_name} graph tokenizer type mismatch: expected={expected_cfg.get('type')} got={got_cfg.get('type')}"
        )
    if str(got_cfg.get("type")) == "perceiver":
        for k in ("num_latents", "hidden_dim", "num_layers", "num_heads", "ff_mult"):
            if int(expected_cfg.get(k)) != int(got_cfg.get(k)):
                raise RuntimeError(
                    f"{stage_name} perceiver config mismatch on {k}: "
                    f"expected={expected_cfg.get(k)} got={got_cfg.get(k)}"
                )
        if abs(float(expected_cfg.get("dropout")) - float(got_cfg.get("dropout"))) > 1e-6:
            raise RuntimeError(
                f"{stage_name} perceiver config mismatch on dropout: "
                f"expected={expected_cfg.get('dropout')} got={got_cfg.get('dropout')}"
            )


def resolve_graph_tokenizer_from_upstream(
    *,
    args: Any,
    upstream_graph_tokenizer_config: Mapping[str, Any],
    stage_name: str,
    upstream_stage_name: str,
) -> Dict[str, Any]:
    req_type = str(getattr(args, "graph_tokenizer_type", "auto")).strip().lower()
    upstream_type = str(upstream_graph_tokenizer_config.get("type")).strip().lower()
    resolved_type = upstream_type if req_type == "auto" else req_type
    if resolved_type not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Unsupported graph_tokenizer_type for {stage_name}: {resolved_type}")
    if resolved_type != upstream_type:
        raise RuntimeError(
            f"{stage_name} tokenizer type must match {upstream_stage_name} provenance: "
            f"requested={resolved_type} upstream={upstream_type}"
        )

    resolved_cfg = {
        "type": resolved_type,
        "num_latents": int(upstream_graph_tokenizer_config.get("num_latents", 32)),
        "hidden_dim": int(upstream_graph_tokenizer_config.get("hidden_dim", 768)),
        "num_layers": int(upstream_graph_tokenizer_config.get("num_layers", 3)),
        "num_heads": int(upstream_graph_tokenizer_config.get("num_heads", 8)),
        "ff_mult": int(upstream_graph_tokenizer_config.get("ff_mult", 4)),
        "dropout": float(upstream_graph_tokenizer_config.get("dropout", 0.0)),
    }
    if resolved_type == "perceiver":
        if int(getattr(args, "perceiver_num_latents", -1)) > 0:
            resolved_cfg["num_latents"] = int(getattr(args, "perceiver_num_latents"))
        if int(getattr(args, "perceiver_num_layers", -1)) > 0:
            resolved_cfg["num_layers"] = int(getattr(args, "perceiver_num_layers"))
        if int(getattr(args, "perceiver_num_heads", -1)) > 0:
            resolved_cfg["num_heads"] = int(getattr(args, "perceiver_num_heads"))
        if int(getattr(args, "perceiver_ff_mult", -1)) > 0:
            resolved_cfg["ff_mult"] = int(getattr(args, "perceiver_ff_mult"))
        if float(getattr(args, "perceiver_dropout", -1.0)) >= 0:
            resolved_cfg["dropout"] = float(getattr(args, "perceiver_dropout"))
        assert_graph_tokenizer_match(
            expected_cfg=upstream_graph_tokenizer_config,
            got_cfg=resolved_cfg,
            stage_name=stage_name,
        )
    return resolved_cfg


def parse_bootstrap_and_graph_tokenizer_from_hparams(
    *,
    hp: Mapping[str, Any],
    bootstrap_field: str,
    bootstrap_mode_field: str,
    context_label: str,
    require_legacy_stage1_qformer_ckpt: bool = False,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    bootstrap = hp.get(bootstrap_field)
    if not isinstance(bootstrap, dict):
        mode = str(hp.get(bootstrap_mode_field, "")).strip().lower()
        tok = str(hp.get("graph_tokenizer_type", "")).strip().lower()
        stage1_qformer_ckpt = hp.get("stage1_qformer_ckpt", None)
        if mode not in {"legacy_stage1", "no_stage1"}:
            if stage1_qformer_ckpt:
                mode = "legacy_stage1"
            elif tok in {"qformer", "perceiver"}:
                mode = "legacy_stage1" if tok == "qformer" else "no_stage1"
        if mode not in {"legacy_stage1", "no_stage1"}:
            raise RuntimeError(
                f"{context_label} checkpoint metadata missing valid bootstrap info. "
                f"Expected {bootstrap_field} or {bootstrap_mode_field}."
            )
        if tok not in {"qformer", "perceiver"}:
            tok = "qformer" if mode == "legacy_stage1" else "perceiver"
        bootstrap = {
            "mode": mode,
            "graph_tokenizer_type": tok,
            "stage1_qformer_ckpt": stage1_qformer_ckpt,
        }

    mode = str(bootstrap.get("mode", "")).strip().lower()
    tok = str(bootstrap.get("graph_tokenizer_type", "")).strip().lower()
    if mode not in {"legacy_stage1", "no_stage1"}:
        raise RuntimeError(f"Invalid {context_label} bootstrap mode in checkpoint: {mode}")
    if tok not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Invalid {context_label} graph tokenizer type in checkpoint: {tok}")
    if require_legacy_stage1_qformer_ckpt and mode == "legacy_stage1" and not bootstrap.get("stage1_qformer_ckpt"):
        raise RuntimeError(
            f"{context_label} checkpoint has legacy_stage1 bootstrap but missing stage1_qformer_ckpt."
        )

    raw_cfg = hp.get("graph_tokenizer_config")
    if isinstance(raw_cfg, dict):
        tokenizer_cfg = normalize_graph_tokenizer_config(raw_cfg)
    else:
        tokenizer_cfg = normalize_graph_tokenizer_config(
            {
                "type": tok,
                "num_latents": hp.get("perceiver_num_latents", 32),
                "hidden_dim": hp.get("perceiver_hidden_dim", 768),
                "num_layers": hp.get("perceiver_num_layers", 3),
                "num_heads": hp.get("perceiver_num_heads", 8),
                "ff_mult": hp.get("perceiver_ff_mult", 4),
                "dropout": hp.get("perceiver_dropout", 0.0),
            }
        )
    if tokenizer_cfg["type"] != tok:
        raise RuntimeError(
            f"{context_label} provenance mismatch: bootstrap type={tok}, tokenizer_config type={tokenizer_cfg['type']}"
        )
    return dict(bootstrap), tokenizer_cfg
