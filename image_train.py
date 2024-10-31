"""
Train a diffusion model on images.
"""

import torch as th
import argparse
from PIL import Image
import numpy as np
import datetime

from guided_diffusion import logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    args = create_argparser().parse_args()

    # logger.configure('./' + datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(args.tmp_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.model_path:
        model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    model.to(device)

    logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=args.class_cond,
    # )
    # print(next(data)[0].shape)
    # print(next(data)[1])
    # print(next(data)[1].items())
    img = Image.open(args.image_path)
    # resize img to 256
    img = img.resize((256, 256))
    img = img.convert('RGB')
    # transfor img to tensor
    img = th.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    # normalize img
    img = img / 127.5 - 1.0
    img = img.to(device)

    label = th.tensor([args.image_label]).to(device)
    data = (img, {'y': label})

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_path=None,
        image_label=0,
        image_path="",
        tmp_path="./tmp",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
