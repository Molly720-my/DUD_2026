import argparse
import os
from pathlib import Path
from datetime import timedelta

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.utils import set_seed, InitProcessGroupKwargs
from torchvision.transforms.functional import to_pil_image, resize
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    EulerAncestralDiscreteScheduler,
)
from src.pipelines.pipeline_SGH_lora import (
    StableDiffusionHarmonyPipeline,
)
from src.dataset.ihd_dataset import IhdDatasetMultiRes as Dataset
from src.models.condition_vae import ConditionVAE
from src.models.unet_2d import UNet2DCustom
from src.utils import make_stage2_input


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a inference script."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--pretrained_unet_model_name_or_path",
        type=str,
        default=None,
    )
    # NOTE Inpaiting Stage1:
    parser.add_argument(
        "--pretrained_unet_Inpainting_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--stage2_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--random_crop",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--mask_dilate",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--strict_mode",
        default=False,
        action="store_true",
    )
    # NOTE Inpainting Stage1: LoRA weight path
    parser.add_argument(
        "--pretrained_lora_model_name_or_path",
        type=str,
        default=None,
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args

def list_in_str(input_list, target_str):
    for item in input_list:
        if item in target_str:
            return True
    return False

def main(args):
    # kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=120))
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        # kwargs_handlers=[kwargs]
    )
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if list_in_str(["condition_vae", "cvae"], args.pretrained_vae_model_name_or_path):
        vae_cls = ConditionVAE
    else:
        vae_cls = AutoencoderKL

    vae = vae_cls.from_pretrained(
        args.pretrained_vae_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_model_name_or_path,
        torch_dtype=weight_dtype,
    )
    # NOTE Inpainting Stage1:
    unet_Inpainting = UNet2DConditionModel.from_pretrained(
        args.pretrained_unet_Inpainting_model_name_or_path,
        torch_dtype=weight_dtype,
    )

    # NOTE Inpainting Stage1: 
    # now we will add new LoRA weights to the attention layers
    # from peft import LoraConfig
    # unet_lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=16,
    #     init_lora_weights="gaussian",
    #     target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    # )
    # unet_Inpainting.add_adapter(unet_lora_config)

    pipeline = StableDiffusionHarmonyPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        # NOTE Harmony Stage2: 
        unet_Harmony=unet,
        # NOTE Inpainting Stage1:
        unet=unet_Inpainting,
        torch_dtype=weight_dtype,
        # NOTE Inpainting Stage1:
        scheduler_Inpainting=EulerAncestralDiscreteScheduler.from_config(
            args.pretrained_model_name_or_path,
            subfolder="scheduler_Inpainting"
        )
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config
    )
    
    # NOTE Inpainting Stage1:
    # pipeline.scheduler_Inpainting = EulerAncestralDiscreteScheduler.from_config(
    #     pipeline.scheduler.config
    # )
    
    # NOTE Inpainting Stage1: add lora weight
    pipeline.load_lora_weights(args.pretrained_lora_model_name_or_path, unet=pipeline.unet)

    pipeline.to(accelerator.device)
    # pipeline.enable_model_cpu_offload(device=accelerator.device)
    pipeline.enable_xformers_memory_efficient_attention()
    pipeline.set_progress_bar_config(disable=True)

    use_stage2 = args.stage2_model_name_or_path is not None
    if use_stage2:
        stage2_model = UNet2DCustom.from_pretrained(
            args.stage2_model_name_or_path,
            torch_dtype=weight_dtype,
        )
        stage2_model.to(accelerator.device)
        stage2_model.eval()
        stage2_model.requires_grad_(False)
        in_channels = stage2_model.config.in_channels
        stage2_model.enable_xformers_memory_efficient_attention()

    dataset = Dataset(
        split="test",
        tokenizer=None,
        resolutions=[args.resolution, args.output_resolution],
        opt=args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,                              # chanege 'True' -> 'False'
        num_workers=args.dataloader_num_workers,
    )

    dataloader = accelerator.prepare(dataloader)
    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Batches",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    for step, batch in enumerate(dataloader):
        if args.strict_mode:
            eval_mask_images = batch[args.output_resolution]["mask"]
            # NOTE 测试编码：'comp'->'real'
            eval_composite_images = batch[args.output_resolution]["comp"]
            # NOTE Inpainting Stage1:
            eval_shadow_mask_images = batch[args.output_resolution]["shadow_mask"]

            if args.output_resolution != args.resolution:
                tgt_size = [args.resolution, args.resolution]
                eval_mask_images = resize(
                    eval_mask_images,
                    size=tgt_size,
                    antialias=True,
                ).clamp(0, 1)
                eval_composite_images = resize(
                    eval_composite_images,
                    size=tgt_size,
                    antialias=True,
                ).clamp(-1, 1)
                # NOTE Inpainting Stage1:
                eval_shadow_mask_images = resize(
                    eval_shadow_mask_images,
                    size=tgt_size,
                    antialias=True,
                ).clamp(0, 1)
        else:
            eval_mask_images = batch[args.resolution]["mask"]
            # NOTE 测试编码：'comp'->'real'
            eval_composite_images = batch[args.resolution]["comp"]
            # NOTE Inpainting Stage1:
            eval_shadow_mask_images = batch[args.resolution]["shadow_mask"]
            # NOTE debug XXX:
            eval_real_images = batch[args.resolution]["real"]
        

        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
        with torch.inference_mode():
            samples = pipeline(
                prompt=batch[args.resolution]["caption"],
                image=eval_composite_images,
                mask_image=eval_mask_images,
                # NOTE Inpainting Stage1: add shadow_mask
                shadow_mask_image=eval_shadow_mask_images,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=10,
                guidance_scale=1.0,
                generator=generator,
                output_type="pt",
            ).images  # [0,1] torch tensor
        
        # print("第%d步的SD-H pipeline完成，准备进行refinement阶段" % step)
        
        if use_stage2 and accelerator.is_main_process:
            samples = 2 * samples - 1  # [-1,1] torch tensor
            if tuple(samples.shape[-2:]) != (
                args.output_resolution,
                args.output_resolution,
            ):
                samples = resize(
                    samples,
                    size=[args.output_resolution, args.output_resolution],
                    antialias=True,
                ).clamp(-1, 1)
            stage2_input = make_stage2_input(
                samples,
                # NOTE 测试编码：'comp'->'real'
                batch[args.output_resolution]["comp"],
                batch[args.output_resolution]["mask"],
                in_channels,
            )
            samples = (
                stage2_model(
                    stage2_input.to(device=accelerator.device, dtype=stage2_model.dtype)
                )
                .sample.cpu()
                .clamp(-1, 1)
            )
            samples = (samples + 1) / 2

            # print("第%d步的refinement完成，准备进行保存结果" % step)
        
        # NOTE debug XXX
        samples = samples.clamp(0, 1)
        
        for i, sample in enumerate(samples):
            sample = to_pil_image(sample)
            output_shape = (args.output_resolution, args.output_resolution)
            if sample.size != output_shape:
                sample = resize(sample, output_shape, antialias=True)
            save_name = (
                batch[args.output_resolution]["comp_path"][i]
                .split("/")[-1]
                .split(".")[0]
                + ".png"
            )
            sample.save(
                os.path.join(args.output_dir, save_name), compression=None, quality=100
            )

        # 保存对应的前景对象mask
        # for i, eval_mask_image in enumerate(eval_mask_images):
        #     eval_mask_image = to_pil_image(eval_mask_image)
        #     # output_shape = (args.output_resolution, args.output_resolution)
        #     # if eval_mask_image.size != output_shape:
        #     #     eval_mask_image = resize(eval_mask_image, output_shape, antialias=True)
        #     save_name = (
        #         batch[args.output_resolution]["mask_path"][i]
        #         .split("/")[-1]
        #         .split(".")[0]
        #         + "_mask" + ".png"
        #     )
        #     eval_mask_image.save(
        #         os.path.join(args.output_dir, save_name), compression=None, quality=100
        #     )

        # 保存对应的前景对象阴影shadow_mask
        # for i, eval_shadow_mask_image in enumerate(eval_shadow_mask_images):
        #     eval_shadow_mask_image = to_pil_image(eval_shadow_mask_image)
        #     # output_shape = (args.output_resolution, args.output_resolution)
        #     # if eval_mask_image.size != output_shape:
        #     #     eval_mask_image = resize(eval_mask_image, output_shape, antialias=True)
        #     save_name = (
        #         batch[args.output_resolution]["mask_path"][i]
        #         .split("/")[-1]
        #         .split(".")[0]
        #         + "_shadow_mask" + ".png"
        #     )
        #     eval_shadow_mask_image.save(
        #         os.path.join(args.output_dir, save_name), compression=None, quality=100
        #     )

        # 保存对应的前景对象composite
        # for i, eval_composite_image in enumerate(eval_composite_images):
        #     eval_composite_image = to_pil_image(eval_composite_image)
        #     # output_shape = (args.output_resolution, args.output_resolution)
        #     # if eval_mask_image.size != output_shape:
        #     #     eval_mask_image = resize(eval_mask_image, output_shape, antialias=True)
        #     save_name = (
        #         batch[args.output_resolution]["comp_path"][i]
        #         .split("/")[-1]
        #         .split(".")[0]
        #         + "_comp" + ".png"
        #     )
        #     eval_composite_image.save(
        #         os.path.join(args.output_dir, save_name), compression=None, quality=100
        #     )
        
        progress_bar.update(1)
    
        # print("第%d步的更新进度条完成" % step)
    
    # print("所有任务完成，准备关闭进度条")
    
    progress_bar.close()
    
    # print("关闭进度条完成，开始同步所有进程")

    # 确保所有参与的进程或设备都在同一阶段同步
    accelerator.wait_for_everyone()

    # print("同步所有进程完成")



if __name__ == "__main__":
    args = parse_args()
    main(args)
