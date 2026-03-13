
```shell
git clone https://github.com/AlexKrachun/diffusion_course_work
cd diffusion_course_work/
conda env create -f attempt_sgd_done/environment.yml
conda activate t2v
pip install flash-attn --no-build-isolation
```

корректно исполнялось на cuda 12.8 на A100 80 gb. 

для исполнения flux пайплайна надо авторизоваться в hugging face:

```shell
hf auth login
```



запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на своем промпте
```shell
python -m vlm_guidance.run run.prompt="a cat on a chair" run.vqa_score=True run.vanilla_sd=True run.flux1=True
```


запустить vanilla sd1.5, vqa guided sd1.5, flux1-dev на текстовом файле ../datasets/subset.txt - где каждая строка - это один промпт
```shell
python -m vlm_guidance.batch_run batch.prompts_file=../datasets/subset.txt run.vqa_score=True run.vanilla_sd=True run.flux1=True    
```