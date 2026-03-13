import os
import json
import contextlib
import types
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor

from t2v_metrics.models.vqascore_models import clip_t5_model

import re


# =========================================================
# utils
# =========================================================
def _to_bchw_float01(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Returns [B,3,H,W] float tensor in [0,1].
    Keeps graph if input is torch.Tensor.
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.ndim != 3:
            raise ValueError("NumPy image must have shape [H,W,C] or [H,W].")
        image = torch.from_numpy(image)

    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be PIL.Image, np.ndarray or torch.Tensor")

    x = image

    if x.ndim == 2:
        x = x.unsqueeze(0).repeat(3, 1, 1)
    elif x.ndim == 3:
        # HWC -> CHW if needed
        if x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(2, 0, 1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported image ndim={x.ndim}")

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    x = x.clamp(0.0, 1.0)
    return x





@dataclass
class VQAEvalResult:
    score: torch.Tensor   # [B]
    ce: torch.Tensor      # [B]


# =========================================================
# Differentiable scorer
# =========================================================
class DifferentiableVQAScorer(torch.nn.Module):
    """
    True differentiable VQA scorer for CLIP-FlanT5.

    Important:
      - does NOT call t2v_metrics.CLIPT5Model.forward()
      - patches vision tower forward to remove internal no_grad
      - keeps model params frozen, but gradients flow to image input
    """
    def __init__(
        self,
        model_name: str = "clip-flant5-xl",
        device: str = "cuda:0",
        use_autocast: bool = True,
    ) -> None:
        super().__init__()

        self.CLIPT5Model = clip_t5_model.CLIPT5Model
        self.format_question = clip_t5_model.format_question
        self.format_answer = clip_t5_model.format_answer
        self.t5_tokenizer_image_token = clip_t5_model.t5_tokenizer_image_token
        self.IGNORE_INDEX = clip_t5_model.IGNORE_INDEX

        self.device = torch.device(device)
        self.use_autocast = use_autocast and self.device.type == "cuda"

        # load official wrapper only as a loader/container
        self.backend = self.CLIPT5Model(model_name=model_name, device=device)

        self.model = self.backend.model
        self.tokenizer = self.backend.tokenizer
        self.image_processor = self.backend.image_processor
        self.conversational_style = self.backend.conversational_style


        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # patch the vision tower so grads can flow through image
        self._patch_vision_tower_forward()

        self.register_buffer(
            "image_mean",
            torch.tensor(self.image_processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(self.image_processor.image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        crop_size = getattr(self.image_processor, "crop_size", None)
        size = getattr(self.image_processor, "size", None)

        if isinstance(crop_size, dict):
            target_h = crop_size.get("height", crop_size.get("shortest_edge", 336))
            target_w = crop_size.get("width", crop_size.get("shortest_edge", 336))
        elif isinstance(size, dict):
            target_h = size.get("height", size.get("shortest_edge", 336))
            target_w = size.get("width", size.get("shortest_edge", 336))
        elif isinstance(size, int):
            target_h = target_w = size
        else:
            target_h = target_w = 336

        self.target_size = (int(target_h), int(target_w))

    def _autocast_context(self):
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _patch_vision_tower_forward(self) -> None:
        """
        Replace CLIPVisionTower.forward (which is wrapped in @torch.no_grad())
        with a differentiable version.
        """
        vision_tower = self.model.get_vision_tower()

        def differentiable_forward(this, images):
            
            image_forward_outs = this.vision_tower(
                images.to(device=this.device, dtype=this.dtype),
                output_hidden_states=True,
            )
            image_features = this.feature_select(image_forward_outs).to(images.dtype)
            return image_features

        vision_tower.forward = types.MethodType(differentiable_forward, vision_tower)

    def preprocess_tensor_image(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Returns normalized BCHW tensor for CLIP image tower.
        Fully differentiable for torch.Tensor input.
        """
        x = _to_bchw_float01(image).to(self.device)


        interpolate_kwargs = dict(
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )

        x = F.interpolate(x, **interpolate_kwargs)
        x = (x - self.image_mean.to(x.device, x.dtype)) / self.image_std.to(x.device, x.dtype)
        return x

    def _build_inputs_and_labels(
        self,
        batch_size: int,
        prompt: Union[str, List[str]],
        question_template: str,
        answer_template: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            texts = [prompt] * batch_size
        else:
            texts = list(prompt)

        if len(texts) != batch_size:
            raise ValueError("Number of prompts must match batch size.")

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        

        questions = [
            self.format_question(q, conversation_style=self.conversational_style)
            for q in questions
        ]
        answers = [
            self.format_answer(a, conversation_style=self.conversational_style)
            for a in answers
        ]


        input_ids = [
            self.t5_tokenizer_image_token(q, self.tokenizer, return_tensors="pt")
            for q in questions
        ]
        labels = [
            self.t5_tokenizer_image_token(a, self.tokenizer, return_tensors="pt")
            for a in answers
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.IGNORE_INDEX,
        )

        input_ids = input_ids[:, : self.tokenizer.model_max_length].to(self.device)
        labels = labels[:, : self.tokenizer.model_max_length].to(self.device)

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(self.device)
        decoder_attention_mask = labels.ne(self.IGNORE_INDEX).to(self.device)

        return input_ids, attention_mask, decoder_attention_mask, labels

    def evaluate(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        prompt: Union[str, List[str]],
        question_template: str = 'Does this figure show "{}"? Please answer yes or no.',
        answer_template: str = "Yes",
    ) -> VQAEvalResult:
        """
        Returns:
            score: [B] = exp(-CE)
            ce:    [B] mean CE over valid answer tokens

        This path is differentiable w.r.t. `image` if `image` is a tensor that
        requires grad.
        """
        images = self.preprocess_tensor_image(image)
        input_ids, attention_mask, decoder_attention_mask, labels = self._build_inputs_and_labels(
            batch_size=images.shape[0],
            prompt=prompt,
            question_template=question_template,
            answer_template=answer_template,
        )

        with self._autocast_context():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                images=images,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
            )

            logits = outputs.logits  # [B, T, V]
            vocab = logits.shape[-1]

            per_token_ce = F.cross_entropy(
                logits.reshape(-1, vocab),
                labels.reshape(-1),
                ignore_index=self.IGNORE_INDEX,
                reduction="none",
            ).view(labels.shape[0], labels.shape[1])

            valid = labels.ne(self.IGNORE_INDEX)
            ce = (per_token_ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            score = torch.exp(-ce)

        return VQAEvalResult(score=score, ce=ce)





# =========================================================
# helpers
# =========================================================
@torch.no_grad()
def encode_prompt(prompt: str, negative_prompt: str = "", max_length: int = 77):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_inputs = tokenizer(
        negative_prompt,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    cond_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
    uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
    return cond_embeds, uncond_embeds


def decode_latents(latents: torch.Tensor) -> torch.Tensor:
    vae_dtype = next(vae.parameters()).dtype
    latents = (latents / 0.18215).to(device=vae.device, dtype=vae_dtype)
    images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images


def save_image_tensor(images_bchw: torch.Tensor, path: str):
    image = images_bchw[0].detach().cpu().clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)



def save_json(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_diff_image(img_a: torch.Tensor, img_b: torch.Tensor, path: str, amplify: float = 4.0):
    diff = (img_b - img_a).abs()
    diff = (diff / (diff.max().clamp(min=1e-8))) * amplify
    diff = diff.clamp(0, 1)
    save_image_tensor(diff, path)



def predict_x0_from_eps(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t,
    scheduler,
) -> torch.Tensor:
    t_idx = int(t.item()) if hasattr(t, "item") else int(t)

    alpha_bar_t = scheduler.alphas_cumprod[t_idx].to(device=x_t.device, dtype=x_t.dtype)
    while alpha_bar_t.ndim < x_t.ndim:
        alpha_bar_t = alpha_bar_t.view(*alpha_bar_t.shape, 1)

    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
    return x0_pred


def predict_eps_with_cfg(
    x_t: torch.Tensor,
    t,
    text_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    latent_model_input = torch.cat([x_t] * 2, dim=0)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    noise_pred = unet(
        latent_model_input,
        t,
        encoder_hidden_states=text_embeds,
    ).sample

    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    eps_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return eps_pred




# =========================================================
# single place for VQA call
# =========================================================
def evaluate_vqa_on_image(
    vqa_scorer: DifferentiableVQAScorer,
    image: torch.Tensor,
    prompt: str,
    question_template: str,
    answer_template: str,
) -> VQAEvalResult:
    return vqa_scorer.evaluate(
        image=image,
        prompt=prompt,
        question_template=question_template,
        answer_template=answer_template,
    )


# =========================================================
# main pipeline
# =========================================================

def _safe_prompt_dirname(index: int, prompt: str, max_len: int = 80) -> str:
    """
    Красивое и безопасное имя папки вида:
    0001_a_cat_on_the_table
    """
    prompt_clean = prompt.strip().lower()
    prompt_clean = re.sub(r"\s+", "_", prompt_clean)
    prompt_clean = re.sub(r"[^a-zA-Z0-9а-яА-Я_=-]+", "", prompt_clean)
    prompt_clean = prompt_clean[:max_len].strip("_")
    if not prompt_clean:
        prompt_clean = "prompt"
    return f"{index:04d}_{prompt_clean}"


def _init_sd15_components(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """
    Инициализирует глобальные компоненты, которые используются helper-функциями:
    encode_prompt, decode_latents, predict_eps_with_cfg и т.д.
    """
    global tokenizer, text_encoder, vae, unet, scheduler, device, dtype

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=dtype,
    ).to(device)

    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)

    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        torch_dtype=dtype,
    ).to(device)

    scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.eval()
    vae.eval()
    unet.eval()

    for m in [text_encoder, vae, unet]:
        for p in m.parameters():
            p.requires_grad_(False)


def generate_with_step_vqa_gd(
    prompt: str,
    negative_prompt: str = "",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: Optional[int] = 0,
    save_dir: str = "sd15_vqa_gd_run",
    vqa_scorer: Optional[DifferentiableVQAScorer] = None,
    question_template: str = 'Does this figure show "{}"? Please answer yes or no.',
    answer_template: str = "Yes",
    gd_steps: int = 3,
    gd_lr: float = 0.05,
    gd_only_first_k_steps: Optional[int] = None,
    save_only_final_img: bool = False,
    final_img_filename: str = "result.png",
):
    """
    Генерация с VQA-guidance.

    Если save_only_final_img=True:
      - сохраняется только финальная картинка + scores.json
      - промежуточные xt/x0/xdiff картинки НЕ сохраняются
      - gd_records / step_records всё равно собираются полностью
    """
    if vqa_scorer is None:
        raise ValueError("vqa_scorer must be provided")

    assert height % 8 == 0 and width % 8 == 0, "height/width must be divisible by 8"

    os.makedirs(save_dir, exist_ok=True)

    save_intermediate_images = not save_only_final_img

    if save_intermediate_images:
        xt_dir = os.path.join(save_dir, "xt_gd")
        x0_dir = os.path.join(save_dir, "x0_gd")
        xdiff_dir = os.path.join(save_dir, "xdiff_gd")
        for d in [xt_dir, x0_dir, xdiff_dir]:
            os.makedirs(d, exist_ok=True)
    else:
        xt_dir = None
        x0_dir = None
        xdiff_dir = None

    cond_embeds, uncond_embeds = encode_prompt(prompt, negative_prompt)
    text_embeds = torch.cat([uncond_embeds, cond_embeds], dim=0)

    scheduler.set_timesteps(num_inference_steps, device=device)

    batch_size = 1
    latent_shape = (batch_size, unet.config.in_channels, height // 8, width // 8)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    n_digits = max(4, len(str(len(scheduler.timesteps))))
    step_records = []

    if gd_only_first_k_steps is None:
        gd_only_first_k_steps = len(scheduler.timesteps)

    gd_only_first_k_steps = max(0, min(int(gd_only_first_k_steps), len(scheduler.timesteps)))

    for i, t in enumerate(scheduler.timesteps):
        filename_base = f"t_{i:0{n_digits}d}"

        x_t_orig = latents.detach().clone()
        x_t_gd = x_t_orig.detach().clone().requires_grad_(True)

        do_gd_this_step = (i < gd_only_first_k_steps and gd_steps > 0)
        gd_records = []
        first_vqa = None

        if do_gd_this_step:
            for gd_iter in range(gd_steps):
                x_t_gd = x_t_gd.detach().clone().requires_grad_(True)

                if save_intermediate_images:
                    with torch.no_grad():
                        xt_img_before = decode_latents(x_t_gd)
                        xt_path = os.path.join(
                            xt_dir,
                            f"{filename_base}_gd_{gd_iter+1:02d}_before_gd_step.png"
                        )
                        save_image_tensor(xt_img_before, xt_path)

                eps_pred = predict_eps_with_cfg(
                    x_t=x_t_gd,
                    t=t,
                    text_embeds=text_embeds,
                    guidance_scale=guidance_scale,
                )

                x0_pred = predict_x0_from_eps(
                    x_t=x_t_gd,
                    eps_pred=eps_pred,
                    t=t,
                    scheduler=scheduler,
                )

                x0_img = decode_latents(x0_pred)

                if save_intermediate_images:
                    with torch.no_grad():
                        x0_img_path = os.path.join(
                            x0_dir,
                            f"{filename_base}_gd_{gd_iter + 1:02d}_before_gd_step.png"
                        )
                        save_image_tensor(x0_img, x0_img_path)

                vqa_result = evaluate_vqa_on_image(
                    vqa_scorer=vqa_scorer,
                    image=x0_img,
                    prompt=prompt,
                    question_template=question_template,
                    answer_template=answer_template,
                )

                vqa_score = vqa_result.score.reshape(-1)[0]
                vqa_ce = vqa_result.ce.reshape(-1)[0]

                if not vqa_ce.requires_grad:
                    raise RuntimeError(
                        "VQA CE is detached from the graph. "
                        "Current CLIP-FlanT5 backend is not differentiable w.r.t. image."
                    )

                grad = torch.autograd.grad(
                    vqa_ce,
                    x_t_gd,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                with torch.no_grad():
                    xt_before = x_t_gd.detach().clone()

                    grad_norm_raw = grad.norm().detach().item()
                    grad = grad / (grad.norm() + 1e-8)
                    x_t_gd -= gd_lr * grad

                    xt_after = x_t_gd.detach().clone()

                    if save_intermediate_images:
                        xt_img_before = decode_latents(xt_before)
                        xt_img_after = decode_latents(xt_after)

                        xdiff_path = os.path.join(
                            xdiff_dir,
                            f"{filename_base}_gd_{gd_iter+1:02d}.png",
                        )
                        save_diff_image(xt_img_before, xt_img_after, xdiff_path)

                if gd_iter == 0:
                    first_vqa = float(vqa_score.detach().item())

                gd_record = {
                    "timestep": int(t.item()),
                    "denoise_step": i,
                    "gd_iter": gd_iter + 1,
                    "grad_norm_raw": float(grad_norm_raw),
                    "vqa_before": float(vqa_score.detach().item()),
                }
                gd_records.append(gd_record)
        else:
            x_t_gd = x_t_orig.detach().clone()

        if save_intermediate_images:
            with torch.no_grad():
                xt_img_after_gd = decode_latents(x_t_gd)
                xt_path = os.path.join(xt_dir, f"{filename_base}_done_gd_step.png")
                save_image_tensor(xt_img_after_gd, xt_path)

        with torch.no_grad():
            eps_pred = predict_eps_with_cfg(
                x_t=x_t_gd,
                t=t,
                text_embeds=text_embeds,
                guidance_scale=guidance_scale,
            )

            x0 = predict_x0_from_eps(x_t_gd, eps_pred, t, scheduler)
            x0_img = decode_latents(x0)

            if save_intermediate_images:
                x0_img_path = os.path.join(x0_dir, f"{filename_base}_before_denoise.png")
                save_image_tensor(x0_img, x0_img_path)

            vqa_after_gd_result = evaluate_vqa_on_image(
                vqa_scorer=vqa_scorer,
                image=x0_img,
                prompt=prompt,
                question_template=question_template,
                answer_template=answer_template,
            )

            vqa_after_gd = float(vqa_after_gd_result.score.reshape(-1)[0].item())
            latents = scheduler.step(eps_pred, t, x_t_gd).prev_sample

        record = {
            "timestep": int(t.item()),
            "denoise_step": i,
            "gd_applied": bool(do_gd_this_step),
            "vqa_score_after_gd": float(vqa_after_gd),
            "gd_stats": gd_records if do_gd_this_step else [],
        }
        step_records.append(record)

        if do_gd_this_step:
            print(
                f"step={i:03d} | t={int(t.item()):04d} | gd_applied=True | "
                f"VQA(x0 before GD)={first_vqa:.4f} | "
                f"VQA(x0 after GD)={vqa_after_gd:.4f}"
            )
        else:
            print(
                f"step={i:03d} | t={int(t.item()):04d} | gd_applied=False | "
                f"VQA(x0 after GD)={vqa_after_gd:.4f}"
            )

    with torch.no_grad():
        final_images = decode_latents(latents)

        final_vqa_result = evaluate_vqa_on_image(
            vqa_scorer=vqa_scorer,
            image=final_images,
            prompt=prompt,
            question_template=question_template,
            answer_template=answer_template,
        )
        final_score = float(final_vqa_result.score.reshape(-1)[0].item())

    step_records.append({
        "timestep": -1,
        "denoise_step": -1,
        "gd_applied": None,
        "vqa_score_after_gd": final_score,
        "gd_stats": None,
    })

    final_image_path = os.path.join(save_dir, final_img_filename)
    save_image_tensor(final_images, final_image_path)

    meta = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "gd_steps": gd_steps,
        "gd_lr": gd_lr,
        "gd_only_first_k_steps": gd_only_first_k_steps,
        "save_only_final_img": save_only_final_img,
        "final_image_path": final_image_path,
        "final_vqa_score": final_score,
        "steps": step_records,
    }

    save_json(meta, os.path.join(save_dir, "scores.json"))
    return final_images, step_records


def run_vlm_guidance_pipeline(
    vqa_scorer: DifferentiableVQAScorer,
    prompt: str,
    negative_prompt: str = "",
    save_only_final_img: bool = True,
    save_dir: str = "vqa",
    final_img_filename: str = "ind.png",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    gd_steps: int = 2,
    gd_lr: float = 1.0,
    gd_only_first_k_steps: int = 5,
) -> Dict[str, Any]:
    """
    Один запуск пайплайна для одного prompt.

    По умолчанию сохраняет только финальную картинку в:
      save_dir / ind.png

    Возвращает словарь с краткой информацией о прогоне.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)

    final_images, step_scores = generate_with_step_vqa_gd(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        save_dir=save_dir,
        vqa_scorer=vqa_scorer,
        question_template='Does this figure show "{}"? Please answer yes or no.',
        answer_template="Yes",
        gd_steps=gd_steps,
        gd_lr=gd_lr,
        gd_only_first_k_steps=gd_only_first_k_steps,
        save_only_final_img=save_only_final_img,
        final_img_filename=final_img_filename,
    )

    result = {
        "prompt": prompt,
        "save_dir": save_dir,
        "final_image_path": os.path.join(save_dir, final_img_filename),
        "scores_json_path": os.path.join(save_dir, "scores.json"),
        "final_vqa_score": step_scores[-1]["vqa_score_after_gd"] if len(step_scores) > 0 else None,
    }
    return result


def run_vlm_guidance_pipeline_multiple_prompts(
    prompts_file: str,
    output_root_dir: str = "vqa_runs",
    negative_prompt: str = "",
    save_only_final_img: bool = True,
    final_img_filename: str = "ind.png",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    vqa_model_name: str = "clip-flant5-xxl",
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    gd_steps: int = 2,
    gd_lr: float = 1.0,
    gd_only_first_k_steps: int = 5,
    skip_empty_lines: bool = True,
) -> List[Dict[str, Any]]:
    """
    Читает txt-файл, где каждая строка = отдельный prompt.
    Для каждого prompt создаёт отдельную папку:
        output_root_dir / 0000_<prompt_name> / ind.png
    и сохраняет туда результат.

    Возвращает список словарей с информацией по всем прогонам.
    """
    os.makedirs(output_root_dir, exist_ok=True)

    _init_sd15_components(model_id=model_id)

    vqa_scorer = DifferentiableVQAScorer(
        model_name=vqa_model_name,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        use_autocast=True,
    )

    with open(prompts_file, "r", encoding="utf-8") as f:
        raw_prompts = [line.rstrip("\n") for line in f]

    if skip_empty_lines:
        prompts = [p.strip() for p in raw_prompts if p.strip()]
    else:
        prompts = [p.strip() for p in raw_prompts]

    all_results = []

    for i, prompt in enumerate(prompts):
        prompt_save_dir = os.path.join(output_root_dir, _safe_prompt_dirname(i, prompt))
        os.makedirs(prompt_save_dir, exist_ok=True)

        print("=" * 100)
        print(f"[{i+1}/{len(prompts)}] prompt: {prompt}")
        print(f"save_dir: {prompt_save_dir}")

        result = run_vlm_guidance_pipeline(
            vqa_scorer=vqa_scorer,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_only_final_img=save_only_final_img,
            save_dir=prompt_save_dir,
            final_img_filename=final_img_filename,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            gd_steps=gd_steps,
            gd_lr=gd_lr,
            gd_only_first_k_steps=gd_only_first_k_steps,
        )
        result["prompt_index"] = i
        all_results.append(result)

    summary_path = os.path.join(output_root_dir, "run_summary.json")
    save_json(all_results, summary_path)

    print("=" * 100)
    print(f"Done. Summary saved to: {summary_path}")

    return all_results


if __name__ == "__main__":
    
    results = run_vlm_guidance_pipeline_multiple_prompts(
        prompts_file="datasets/simple_cases.txt",
        output_root_dir="simple_cases",
        negative_prompt="blurry, low quality",
        save_only_final_img=True,
        final_img_filename="img.png",
        gd_steps=2,
        gd_lr=1.0,
        gd_only_first_k_steps=5,
    )
    
    # results = run_vlm_guidance_pipeline_multiple_prompts(
    #     prompts_file="prompts.txt",
    #     output_root_dir="my_generations_full_logs",
    #     save_only_final_img=False,
    #     final_img_filename="ind.png",
    # )