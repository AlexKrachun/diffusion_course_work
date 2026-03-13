# VLM guidance refactor

Это рефактор пайплайна для VQA-guided diffusion sampling.

## Что изменено

- Убран глобальный state для SD-компонентов.
- Diffusion backend, scorer и guidance algorithm разделены по интерфейсам.
- Scorer можно заменить новым классом без переписывания sampler-а.
- Mixed precision (`torch.amp.autocast`) изолирован внутри backend/scorer вместо размазанной логики в одном скрипте.
- Добавлены Hydra-конфиги и логирование запусков.

## Структура

- `vlm_guidance/diffusion/` — backend для diffusion-моделей.
- `vlm_guidance/scorers/` — differentiable scorer'ы.
- `vlm_guidance/guidance/` — алгоритмы guidance.
- `vlm_guidance/configs/` — Hydra-конфиги.
- `vlm_guidance/run.py` — одиночный запуск.
- `vlm_guidance/batch_run.py` — запуск по txt-файлу с prompt'ами.

## Запуск

```bash
python -m vlm_guidance.run
```

С override из CLI:

```bash
python -m vlm_guidance.run run.prompt="a red car" algorithm.gd_lr=0.05 algorithm.gd_steps=4
```

Батч:

```bash
python -m vlm_guidance.batch_run batch.prompts_file=datasets/subset.txt batch.output_root_dir=subset
```

Sweep:

```bash
python -m vlm_guidance.run -m algorithm.gd_lr=0.05,0.1,0.2 algorithm.gd_steps=1,2
```

## Как добавить новый VQA scorer

1. Создай новый класс в `vlm_guidance/scorers/`, унаследованный от `BaseDifferentiableScorer`.
2. Реализуй метод `forward(image, prompt, **kwargs) -> ScoreOutput`.
3. Добавь YAML-конфиг в `configs/scorer/`.
4. Подмени scorer через `defaults` или CLI.

## Замечания по mixed precision

Для differentiable scorer безопаснее считать сам loss в `float32`, даже если forward идет под autocast. Поэтому в scorer-е `cross_entropy` считается от `logits.float()`.
