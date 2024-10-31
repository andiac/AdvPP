"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision

from PIL import Image


from guided_diffusion import logger
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    
    args = create_argparser().parse_args()
    # print(args)
    # print(args.timestep_respacing)
    # print(type(args.timestep_respacing))
    cur_save_path = os.path.join(args.save_path, str(args.image_label))
    os.makedirs(cur_save_path, exist_ok=True)
    
    # load ./corgi.jpg use PIL
    img = Image.open(args.image_path)
    # resize img to 256
    img = img.resize((256, 256))
    img = img.convert('RGB')
    # save img as png
    img.save(os.path.join(cur_save_path, "./origin.png"))
    img.save(os.path.join(cur_save_path, "./origin.eps"))
    # transfor img to tensor
    img = th.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    # normalize img
    img = img / 127.5 - 1.0
    img = img.to(device)
    # add noise to img
    # img = img * 0.0 + th.randn_like(img, device=device) * 1.0
    # img = img * 0.5 + th.randn(*img.shape, device=device) * 0.5 
    # img = th.randn_like(img, device=device,dtype=th.float32)
    # save img as png
    # torchvision.utils.save_image((img + 1.0) / 2.0, "./corgi.png")

    # dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # noised_img = diffusion.q_sample(img, th.tensor([250 - 100], device=device))
    noised_img = diffusion.q_sample(img, th.tensor([int(args.timestep_respacing) - args.start_step], device=device))
    # save noised_img as png
    torchvision.utils.save_image((noised_img + 1.0) / 2.0, os.path.join(cur_save_path, "noised.png"))
    torchvision.utils.save_image((noised_img + 1.0) / 2.0, os.path.join(cur_save_path, "noised.eps"))

    logger.log("loading classifier...")
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     th.load(args.classifier_path, map_location="cpu")
    # )
    # classifier.to(device)
    # if args.classifier_use_fp16:
        # classifier.convert_to_fp16()
    # classifier.eval()

    # surrogate classifier
    classifier = torchvision.models.resnet50(pretrained=True)
    classifier.to(device)
    classifier.eval()

    betas = get_named_beta_schedule("linear", 1000)
    alphas = 1.0 - betas
    alpha_bars = th.tensor(np.cumprod(alphas, axis=0), dtype=th.float32).to(device)

    logits = classifier(img)
    # get max value index and the second value index
    idx = th.argmax(logits, dim=1)
    second_idx = th.argsort(logits, dim=1)[:, -2]

    if args.image_label == idx:
        target_label = second_idx.item()
    else:
        target_label = idx.item()

    print("origin image label: ", idx)
    print("origin image second label: ", second_idx)

    def cond_fn(x, t, y=None, oriy=None):
        assert y is not None
        # y = th.randint(
        #     low=0, high=1, size=(args.batch_size,), device=device
        # )
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # tweedie's formula
            x_0 = (1.0 / th.sqrt(alpha_bars[t])) * (x_in - th.sqrt(1-alpha_bars[t]) * model(x_in, t, oriy)[:, :3, :, :])
            # logits = classifier(x_0, t*0)
            # logits = classifier(x_in, t*0)
            # logits = classifier(x_in)
            logits = classifier(x_0)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None, oriy=None):
        assert oriy is not None
        return model(x, t, oriy if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        # classes = th.randint(
        #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device
        # )
        #classes = th.randint(
        #    low=254, high=255, size=(args.batch_size,), device=device
        #)
        origin_classes = th.randint(
            low=args.image_label, high=args.image_label+1, size=(args.batch_size,), device=device
        )
        target_classes = th.randint(
            low=target_label, high=target_label+1, size=(args.batch_size,), device=device
        )
        model_kwargs["y"] = target_classes
        model_kwargs["oriy"] = origin_classes

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=True,
            # noise=th.randn(*noised_img.shape, device=device),
            # noise=th.randn_like(noised_img, device=device),
            noise=noised_img,
            # start_img=noised_img,
            start_step=args.start_step,
        )

        # classify the samples
        logits = classifier(sample)
        log_probs = F.log_softmax(logits, dim=-1)
        classes_pred = log_probs.argmax(dim=-1)
        print(classes_pred)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        # all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        # gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        # dist.all_gather(gathered_labels, classes)
        # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        all_images.append(sample.cpu().numpy())
        all_labels.append(classes_pred.cpu().numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(cur_save_path, f"sample_{args.sample_id}_{all_labels[-1][0]}.npz")
    # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    Image.fromarray(arr[0]).save(os.path.join(cur_save_path, f"./sample_{args.sample_id}_{all_labels[-1][0]}.png"))
    Image.fromarray(arr[0]).save(os.path.join(cur_save_path, f"./sample_{args.sample_id}_{all_labels[-1][0]}.eps"))

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        start_step=0,
        image_path="",
        image_label=0,
        save_path="",
        sample_id=0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()





