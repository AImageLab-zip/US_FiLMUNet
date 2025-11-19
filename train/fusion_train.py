from argparse import Namespace
from collections import defaultdict

from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from data_classes.datasets import USdatasetOmni
from torchvision.transforms import v2
import torch, wandb, random
from sklearn.metrics import accuracy_score
from nets.fused_net import FusionWrapper
from utils.stratified_splits import build_train_val_datasets
from utils.paths import DATA_DIR
from utils.utils import (
    organ_to_class_dict,
    multi_cls_labels_dict,
    class_to_organ_dict,
    get_sft_transforms,
    compute_dsc,
    compute_nsd,
    mask_overlap_visualization,
    generate_run_hash
)
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
import hashlib, time


def compute_metrics(eval_pred):
    logits, _ = eval_pred
    logits, masks, organ_ids = logits  # logits/masks shape B, 512, 512
    pred_th = (torch.sigmoid(torch.from_numpy(logits)) > 0.7).float()
    pred_np = pred_th.cpu().numpy()

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


def train(args: Namespace):

    if args.onpublic:
        train_dataset, val_dataset = build_train_val_datasets(
            DATA_DIR, args, seed=args.seed
        )

        test_dataset = USdatasetOmni(
            DATA_DIR,
            "val_cls",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
        )
    else:
        train_dataset = USdatasetOmni(
            DATA_DIR,
            "train",
            transforms=get_sft_transforms(train=True),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
        )
        val_dataset = USdatasetOmni(
            DATA_DIR,
            "val",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
        )
        test_dataset = USdatasetOmni(
            DATA_DIR,
            "test",
            transforms=get_sft_transforms(train=False),
            data_type=args.dataset_type,
            out_size=args.dataset_size,
            ccl_crop=args.use_ccl_crop,
            keep_aspect_ratio=args.keep_aspect_ratio,
            self_norm=args.self_norm,
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
    model = FusionWrapper(args)
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
    trainer.evaluate()

    predictions = trainer.predict(test_dataset=test_dataset)
    print("Test results:", predictions.metrics)
