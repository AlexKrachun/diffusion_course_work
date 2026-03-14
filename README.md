
```shell
git clone https://github.com/AlexKrachun/diffusion_course_work
cd diffusion_course_work/
conda env create -f environment.yaml
conda activate t2v
pip install flash-attn --no-build-isolation
pip install hydra-core

```

корректно исполнялось на Python 3.10.20, cuda 12.0, 12.8 на A100 80 gb vram. 

для исполнения flux пайплайна надо авторизоваться в hugging face:

```shell
hf auth login
```



запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на своем промпте
запускать внутри vlm_guidance_project_multi_pipeline_optim
```shell
python -m vlm_guidance.run run.prompt="a cat on a chair" run.vqa_score=True run.vanilla_sd=True run.flux1=True
```


запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на текстовом файле ../datasets/subset.txt - где каждая строка - это один промпт
запускать внутри vlm_guidance_project_multi_pipeline_optim
```shell
python -m vlm_guidance.batch_run batch.prompts_file=../datasets/simple_cases.txt run.vqa_score=True run.vanilla_sd=True run.flux1=True batch.output_root_dir=vlm_guidance_project_multi_pipeline_optim/simple_cases_generations    

python -m vlm_guidance.batch_run batch.prompts_file=../datasets/complex_cases.txt run.vqa_score=True run.vanilla_sd=True run.flux1=True  batch.output_root_dir=vlm_guidance_project_multi_pipeline_optim/complex_cases_generations   
```


посчитат и сохранить в csv файл clip score по папке с изображениями полученными разными пайплайнами (папка получена с помощью прогона `python -m vlm_guidance.batch_run`)
```shell
python3 metrics/clip_score_clalc.py -generations vlm_guidance_project_multi_pipeline_optim/simple_cases_generations --output metrics/clip_score_simple_result.csv
python3 metrics/clip_score_clalc.py -generations vlm_guidance_project_multi_pipeline_optim/complex_cases_generations --output metrics/clip_score_complex_result.csv
```

построить графики clip score по csv со значениями метрики 
```shell
python3 metrics/clip_visualize.py --input metrics/clip_score_simple_result.csv --output-dir metrics/clip_plots_simple_data
python3 metrics/clip_visualize.py --input metrics/clip_score_complex_result.csv --output-dir metrics/clip_score_complex_result

```


Для подсчетам метрики alignment




смотри, мне надо оценить clip score n пайплайнов генерации изображений. В моем случае сейчас n = 3. Посмотри, как устроена папка vlm_guidance_project_multi_pipeline_optim/subset - там на есть множество папок - по одной на промпт. В каждой из этих папок есть по одной подпапке на каждый пайплайн: flux1, vanilla_sd, vqa_score - всего три пайплайна сейчас. Внутри каждой из этих подпапок есть изображение img.png и промпт prompt.txt. Соостветсвенно я хотел бы запустить

python clip_score_clalc.py -generations vlm_guidance_project_multi_pipeline_optim/subset 

и посчитать clip score для каждого изображения каждого пайплайна и записать результаты в metrics/result.csv, где на каждый пайплайн и каждый номер промпта (по порядку) был бы записал clip score соответствующего изображения и промпта. 

напиши, основываясь на clip_score.py новый файл clip_score_clalc.py, который бы выполнил поставленную задачу при запуске
