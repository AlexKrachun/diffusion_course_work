from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from vlm_guidance.guidance.vqa_gradient import VQAGradientGuidanceRunner
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    diffusion = instantiate(cfg.diffusion)
    scorer = instantiate(cfg.scorer)
    run_cfg = instantiate(cfg.run)
    guidance_cfg = instantiate(cfg.algorithm)

    runner = VQAGradientGuidanceRunner(
        diffusion=diffusion,
        scorer=scorer,
        run_cfg=run_cfg,
        guidance_cfg=guidance_cfg,
    )
    result = runner.run(Path.cwd())
    save_json(result, Path.cwd() / "result_summary.json")
    log.info("Run completed. Final score: %.6f", result["final_score"])
    log.info("Saved outputs to %s", Path.cwd())


if __name__ == "__main__":
    main()
