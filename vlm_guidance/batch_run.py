from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vlm_guidance.guidance.vqa_gradient import VQAGradientGuidanceRunner
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)


def read_prompts(path: str, skip_empty_lines: bool = True) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f]
    if skip_empty_lines:
        return [p.strip() for p in raw if p.strip()]
    return [p.strip() for p in raw]


def safe_prompt_dirname(index: int, prompt: str, max_len: int = 80) -> str:
    import re
    prompt_clean = prompt.strip().lower()
    prompt_clean = re.sub(r"\s+", "_", prompt_clean)
    prompt_clean = re.sub(r"[^a-zA-Z0-9а-яА-Я_=-]+", "", prompt_clean)
    prompt_clean = prompt_clean[:max_len].strip("_") or "prompt"
    return f"{index:04d}_{prompt_clean}"


@hydra.main(version_base=None, config_path="configs", config_name="batch_config")
def main(cfg: DictConfig) -> None:
    log.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    prompts = read_prompts(cfg.batch.prompts_file, skip_empty_lines=cfg.batch.skip_empty_lines)
    if not prompts:
        raise ValueError(f"No prompts found in {cfg.batch.prompts_file}")

    diffusion = instantiate(cfg.diffusion)
    scorer = instantiate(cfg.scorer)
    guidance_cfg = instantiate(cfg.algorithm)

    results = []
    root_dir = Path.cwd() / cfg.batch.output_root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images", total=len(prompts), dynamic_ncols=True)):
        run_cfg = instantiate(cfg.run, prompt=prompt)
        save_dir = root_dir / safe_prompt_dirname(i, prompt)
        runner = VQAGradientGuidanceRunner(diffusion=diffusion, scorer=scorer, run_cfg=run_cfg, guidance_cfg=guidance_cfg)
        result = runner.run(save_dir)
        result["prompt_index"] = i
        results.append(result)

    save_json(results, root_dir / "run_summary.json")
    log.info("Completed %d runs. Summary saved to %s", len(results), root_dir / "run_summary.json")


if __name__ == "__main__":
    main()
