from __future__ import annotations

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from vlm_guidance.execution import execute_selected_pipelines

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=False))
    if not cfg.run.prompt:
        raise ValueError("run.prompt must be provided for single-run mode.")

    results = execute_selected_pipelines(cfg=cfg, run_dir=Path.cwd(), prompt=cfg.run.prompt)
    log.info("Completed %d pipelines. Outputs saved to %s", len(results), Path.cwd())


if __name__ == "__main__":
    main()
