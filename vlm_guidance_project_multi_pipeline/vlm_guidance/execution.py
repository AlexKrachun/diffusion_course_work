from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig

from vlm_guidance.generation.base import Text2ImageRunner
from vlm_guidance.guidance.vqa_gradient import VQAGradientGuidanceRunner
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)


def execute_selected_pipelines(cfg: DictConfig, run_dir: Path, prompt: str) -> Dict[str, Dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict] = {}

    selected: List[str] = []
    if cfg.run.vqa_score:
        selected.append("vqa_score")
    if cfg.run.vanilla_sd:
        selected.append("vanilla_sd")
    if cfg.run.flux1:
        selected.append("flux1")
    if not selected:
        raise ValueError("At least one pipeline must be enabled: run.vqa_score, run.vanilla_sd or run.flux1.")

    common_kwargs = dict(
        prompt=prompt,
        negative_prompt=cfg.run.negative_prompt,
        height=cfg.run.height,
        width=cfg.run.width,
        num_inference_steps=cfg.run.num_inference_steps,
        guidance_scale=cfg.run.guidance_scale,
        seed=cfg.run.seed,
        batch_size=cfg.run.batch_size,
    )

    if cfg.run.vqa_score:
        log.info("Running VQA guidance pipeline")
        diffusion = instantiate(cfg.diffusion)
        scorer = instantiate(cfg.scorer)
        run_cfg = instantiate(cfg.run)
        run_cfg.prompt = prompt
        guidance_cfg = instantiate(cfg.algorithm)
        runner = VQAGradientGuidanceRunner(
            diffusion=diffusion,
            scorer=scorer,
            run_cfg=run_cfg,
            guidance_cfg=guidance_cfg,
        )
        results["vqa_score"] = runner.run(run_dir / "vqa_score")

    if cfg.run.vanilla_sd:
        log.info("Running vanilla SD1.5 pipeline")
        vanilla_pipe = instantiate(cfg.vanilla_sd)
        runner = Text2ImageRunner(vanilla_pipe)
        results["vanilla_sd"] = runner.run(
            run_dir / "vanilla_sd",
            **common_kwargs,
        )

    if cfg.run.flux1:
        log.info("Running FLUX.1-dev pipeline")
        flux_pipe = instantiate(cfg.flux1)
        runner = Text2ImageRunner(flux_pipe)
        results["flux1"] = runner.run(
            run_dir / "flux1",
            **common_kwargs,
        )

    save_json(results, run_dir / "run_summary.json")
    return results
