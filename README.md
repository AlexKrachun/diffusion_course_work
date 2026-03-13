
```shell
git clone https://github.com/AlexKrachun/diffusion_course_work
cd diffusion_course_work/
conda env create -f attempt_sgd_done/environment.yml
conda activate t2v
pip install flash-attn --no-build-isolation
```

корректно исполнялось на cuda 12.8 на A100 80 gb. для пайплайна нужно 