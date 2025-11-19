from argparse import Namespace
import torch, torchvision, wandb
from torch import nn
from nets.cls_net import OmniClsCBAM
from nets.segm_net import UNet2DFiLM
from utils.utils import organ_to_class_dict
from safetensors.torch import load_file


class FusionWrapper(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        
        self.cls_net = OmniClsCBAM(
            resnet_size=args.rn_size,
            use_cbam=args.use_cbam,
            use_film=args.use_film,
            predict_bboxes=args.regr_bbox,
            mlp_organ=args.cls_organ,
        )
        if args.cls_checkpoint is not None:
            state_dict = load_file(args.cls_checkpoint)
            self.cls_net.load_state_dict(state_dict)
        # Freeze all parameters of cls_net
        for param in self.cls_net.parameters():
            param.requires_grad = False

        self.segm_net = UNet2DFiLM(
            in_channels=3,
            num_classes=1,
            n_organs=len(organ_to_class_dict),
            size=32,
            depth=args.unet_depth,
            film_start=args.film_start,
        )
        if args.segm_checkpoint is not None:
            state_dict_segm = load_file(args.segm_checkpoint)
            self.segm_net.load_state_dict(state_dict_segm)

        if args.freeze_enc:
            self.freeze_encoder_segm(freeze_bottleneck=True)
        self.transformer_fusion = bool(args.transformer_fusion)
        if args.transformer_fusion:
            self.ll1 = nn.Linear(512, 4096)
            self.ll2 = nn.Linear(128, 2048)

            # Stack multiple layers to create TransformerEncoder
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.4,
                    batch_first=True,  # Input shape: (batch, seq, feature)
                ),
                num_layers=6,
            )
            self.pos_embedding = nn.Parameter(
                torch.randn(1, 32 + 1, 128)  # +1 for CLS token
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, 128))
        else:
            self.ll1 = nn.Linear(512, 2048)
        self.alpha = nn.Parameter(torch.zeros(args.num_alphas))

    def freeze_encoder_segm(self, freeze_bottleneck: bool = False):
        """
        Freezes the encoder part (and optionally the bottleneck) of the network.
        This prevents their parameters from being updated during training.

        Args:
            freeze_bottleneck (bool): If True, also freezes the bottleneck layer.
        """
        # Freeze all encoder layers
        for param in self.segm_net.encoder.parameters():
            param.requires_grad = False

        # Optionally freeze the bottleneck too
        if freeze_bottleneck:
            for param in self.segm_net.bottleneck.parameters():
                param.requires_grad = False

        print(
            f"Encoder {'and bottleneck ' if freeze_bottleneck else ''}frozen. "
            f"Decoder remains trainable."
        )

    def forward(
        self, pixel_values, organ_id=None, labels=None, masks=None, bbox_coords=None
    ):
        # def forward(self, x_cls, x_segm, organ_id):
        B = pixel_values.shape[0]
        x_cls = torchvision.transforms.v2.functional.resize(pixel_values, (256, 256))
        x_segm = pixel_values
        with torch.no_grad():
            cls_feat = self.cls_net.get_feat(x_cls, organ_id) # B, 512

        if self.transformer_fusion: 
            cls_feat_ll1 = self.ll1(cls_feat) # B, 2048
            tokenized = cls_feat_ll1.reshape((B, 32, 128))
            cls_tokens = self.cls_token.expand(B, -1, -1)  
            x = torch.cat([cls_tokens, tokenized], dim=1)
            x = x + self.pos_embedding
            sequence_output = self.transformer_encoder(x)  # (B, num_tokens+1, d_model)
            cls_output = sequence_output[:, 0]
            cls_feat = self.ll2(cls_output)
        else:  
            cls_feat = self.ll1(cls_feat)

        gated_cls_feat = torch.tanh(self.alpha) * cls_feat
        gated_cls_feat = gated_cls_feat.unsqueeze(-1).unsqueeze(-1)

        bottleneck_feat, feat_list, pads = self.segm_net.encode(x_segm, organ_id)
        fused_feat = bottleneck_feat + gated_cls_feat

        logits = self.segm_net.decode(
            x_segm, organ_id, fused_feat, feat_list, pads
        ).squeeze()

        if self.alpha.numel() == 1:
            wandb.log({"alpha": self.alpha.item()}, commit=False)
        else:
            wandb.log(
                {"alpha": wandb.Histogram(self.alpha.detach().cpu().numpy())},
                commit=False,
            )

        loss = self.segm_net.criterion(logits, masks)
        return {"loss": loss, "logits": logits, "labels": masks, "organ_id": organ_id}
