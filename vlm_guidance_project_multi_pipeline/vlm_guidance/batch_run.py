from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from vlm_guidance.execution import execute_selected_pipelines
from vlm_guidance.utils.io import save_json

log = logging.getLogger(__name__)


def read_prompts(path: str, skip_empty_lines: bool = True) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        raw = [line.rstrip("\n") for line in f]
    if skip_empty_lines:
        prompts = [p.strip() for p in raw if p.strip()]
    else:
        prompts = [p.strip() for p in raw]
    if not prompts:
        raise ValueError(f"No prompts found in file: {path}")
    return prompts


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

    root_dir = Path.cwd() / cfg.batch.output_root_dir
    root_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating images", total=len(prompts), dynamic_ncols=True)):
        save_dir = root_dir / safe_prompt_dirname(i, prompt)
        run_results = execute_selected_pipelines(cfg=cfg, run_dir=save_dir, prompt=prompt)
        results.append({
            "prompt_index": i,
            "prompt": prompt,
            "save_dir": str(save_dir),
            "pipelines": run_results,
        })

    save_json(results, root_dir / "run_summary.json")
    log.info("Completed %d prompts. Summary saved to %s", len(results), root_dir / "run_summary.json")


if __name__ == "__main__":
    main()
