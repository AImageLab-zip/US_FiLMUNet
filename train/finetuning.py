from argparse import Namespace
from collections import defaultdict
from safetensors.torch import load_file
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import InterpolationMode, v2
import torch, wandb, random
from sklearn.metrics import accuracy_score
from nets.cls_net import OmniClsCBAM
from nets.segm_net import UNet2DFiLM, MedSAM, MedSAMPrompt
from utils.utils import (
    organ_to_class_dict,
    class_to_organ_dict,
    multi_cls_labels_dict,
    get_sft_transforms,
    compute_dsc,
    compute_nsd,
    generate_run_hash,
)
import numpy as np
from utils.paths import *
from copy import deepcopy
from pathlib import Path


def load_film_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    """
    Load a pretrained FiLM UNet checkpoint while allowing the FiLM embeddings
    to grow when the current model declares more organs.

    This copies weights that match exactly and, for FiLM embeddings whose first
    dimension encodes organs, it copies the original rows and keeps the newly
    initialised rows untouched so they can learn the new organs.
    """
    pretrained_state = load_file(checkpoint_path)
    current_state = model.state_dict()
    expanded = []
    copied = []
    skipped = []

    for key, current_tensor in current_state.items():
        if key not in pretrained_state:
            continue

        pretrained_tensor = pretrained_state[key]
        if pretrained_tensor.shape == current_tensor.shape:
            current_state[key] = pretrained_tensor
            copied.append(key)
            continue

        same_rank = pretrained_tensor.ndim == current_tensor.ndim
        same_trailing_shape = pretrained_tensor.shape[1:] == current_tensor.shape[1:]
        grows_first_dim = pretrained_tensor.shape[0] <= current_tensor.shape[0]

        if same_rank and same_trailing_shape and grows_first_dim:
            # Keep randomly initialised rows for the new organs, copy the old ones
            updated_tensor = current_tensor.clone()
            updated_tensor[: pretrained_tensor.shape[0]] = pretrained_tensor
            current_state[key] = updated_tensor
            expanded.append(key)
        else:
            skipped.append(key)

    load_result = model.load_state_dict(current_state, strict=False)
    print(
        f"Loaded {checkpoint_path} → copied={len(copied)} expanded={len(expanded)} skipped={len(skipped)}"
    )
    if expanded:
        print("Expanded FiLM embeddings for:", expanded)
    if skipped:
        print("Skipped keys due to incompatible shape:", skipped)
    return load_result


def compute_metrics(eval_pred):
    logits, _ = eval_pred
    logits, masks, organ_ids = logits  # logits/masks shape B, 512, 512
    logits = v2.functional.resize(
        torch.from_numpy(logits),
        (masks.shape[-1], masks.shape[-1]),
        interpolation=InterpolationMode.NEAREST,
    ).numpy()
    pred_th = (torch.sigmoid(torch.from_numpy(logits)) > 0.7).float()
    pred_np = pred_th.cpu().numpy()
    masks = (
        (
            v2.functional.resize(
                torch.from_numpy(masks),
                (masks.shape[-1], masks.shape[-1]),
                interpolation=InterpolationMode.NEAREST,
            )
            > 0.5
        )
        .float()
        .numpy()
    )
    random.seed(42)  # set seed for reproducibility
    numbers = list(range(logits.shape[0]))
    sampled = random.sample(numbers, 30)
    overlays = []
    for s in sampled:
        overlays.append(
            mask_overlap_visualization(pred_th[s], torch.from_numpy(masks[s]))
        )

    organ_stats = defaultdict(lambda: {"dsc": [], "nsd": []})
    gt_np = masks
    for i, organ in enumerate(organ_ids):
        dsc = compute_dsc(gt_np[i], pred_np[i])
        nsd = compute_nsd(gt_np[i], pred_np[i], tolerance=1)
        organ = class_to_organ_dict[organ]
        organ_stats[organ]["dsc"].append(dsc)
        organ_stats[organ]["nsd"].append(nsd)

    wandb_metrics = {}
    for organ, lst in organ_stats.items():
        dsc_m = float(np.mean(lst["dsc"]))
        nsd_m = float(np.mean(lst["nsd"]))

        wandb_metrics[f"dsc_{organ}"] = dsc_m
        wandb_metrics[f"nsd_{organ}"] = nsd_m

    wandb_images = []
    for i, s in enumerate(sampled):
        wandb_images.append(wandb.Image(overlays[i], caption=f"overlap_{s}"))

    wandb.log({"overlays_eval": wandb_images}, commit=False)

    return wandb_metrics


def mask_overlap_visualization(pred: torch.Tensor, mask: torch.Tensor):
    """
    Create a color visualization tensor showing overlap between predicted mask and ground truth.
      - Green: correct prediction (TP)
      - Yellow: missed region (FN)
      - Red: false positive (FP)

    Args:
        mask (Tensor): shape (1, H, W) or (H, W), binary ground truth mask
        threshold (float): threshold for logits -> binary mask

    Returns:
        Tensor: RGB visualization tensor of shape (3, H, W), values in [0,1]
    """
    # Ensure proper shape - squeeze all dimensions of size 1
    pred = pred.squeeze()
    mask = mask.squeeze()

    # Ensure they are 2D

    # Logical masks
    tp = (pred == 1) & (mask == 1)  # True positives
    fn = (pred == 0) & (mask == 1)  # False negatives
    fp = (pred == 1) & (mask == 0)  # False positives

    # Create RGB channels
    h, w = pred.shape
    vis = torch.zeros(3, h, w)

    # Assign colors
    vis[0][tp] = 0  # R
    vis[1][tp] = 1  # G
    vis[2][tp] = 0  # B

    vis[0][fn] = 1  # R
    vis[1][fn] = 1  # G
    vis[2][fn] = 0  # B

    vis[0][fp] = 1  # R
    vis[1][fp] = 0  # G
    vis[2][fp] = 0  # B

    return vis


def train(args: Namespace):
    train_txt_paths = list(Path(args.dataset_path).rglob("train.txt"))
    val_txt_paths = list(Path(args.dataset_path).rglob("val.txt"))
    val_cls_txt_paths = list(Path(args.dataset_path).rglob("val_cls.txt"))  
    if len(train_txt_paths) == 0 or len(val_txt_paths) == 0 or len(val_cls_txt_paths) == 0: 
        print(f"No train/val/val_cls.txt files found in {args.dataset_path}. Exiting...")
        return

    train_dataset = USdatasetOmni(
        args.dataset_path,
        "train",
        transforms=get_sft_transforms(train=True),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
        testicle_split=args.testicle_split,
    )
    val_dataset = USdatasetOmni(
        args.dataset_path,
        "val",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
        testicle_split=args.testicle_split,
    )
    test_dataset = USdatasetOmni(
        args.dataset_path,
        "val_cls",
        transforms=get_sft_transforms(train=False),
        data_type=args.dataset_type,
        out_size=args.dataset_size,
        ccl_crop=args.use_ccl_crop,
        keep_aspect_ratio=args.keep_aspect_ratio,
        self_norm=args.self_norm,
        include_testicles=True,
        testicle_split=args.testicle_split,
    )
    print(
        f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    wandb.login()
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=args,
    )
    if args.use_medsam:
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)

        model = MedSAM(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=args.freeze_image_encoder,
        )
        state_dict = load_file(MEDSAM_UNFREEZED_CHECKPOINT)
        model.load_state_dict(state_dict)
        load_result = model.load_state_dict(state_dict)
        print(load_result)

    elif args.use_medsam_prompt:
        from segment_anything import sam_model_registry

        sam_model = sam_model_registry["vit_b"](checkpoint=MEDSAM_BASE_WEIGHTS)

        model = MedSAMPrompt(
            image_encoder=deepcopy(sam_model.image_encoder),
            mask_decoder=deepcopy(sam_model.mask_decoder),
            prompt_encoder=deepcopy(sam_model.prompt_encoder),
            predict_bboxes=True,
            freeze_image_encoder=args.freeze_image_encoder,
        )
        state_dict = load_file(MEDSAM_PROMPT_CHECKPOINT)
        model.load_state_dict(state_dict)
        load_result = model.load_state_dict(state_dict)
        print(load_result)
    else:
        model = UNet2DFiLM(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict),
            size=32,
            depth=args.unet_depth,
            film_start=args.film_start,
        )
        if args.unet_depth == 5:
            load_result = load_film_checkpoint(model, FILMUNET5_CHECKPOINT)
            print(load_result)
        elif args.unet_depth == 4:
            load_result = load_film_checkpoint(model, FILMUNET4_CHECKPOINT)
            print(load_result)
        else:
            print("No checkpoint loaded!!!!!!!!!")

    # Generate custom hashed directory name
    run_hash = generate_run_hash(args)
    output_dir = f"{run_hash}"
    print(f"Saving results to: {output_dir}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir="./logs",
        seed=args.seed,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to=["wandb"] if args.wandb_project else None,
        run_name=args.wandb_run_name,
        dataloader_num_workers=args.num_workers,
        logging_steps=10,
        log_level="info",
        eval_accumulation_steps=int(args.epochs),
        optim=args.optim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.acc_grad,
        # fp16=True,
        # push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Test dataset 0
    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)
