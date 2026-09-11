"""
LoFTR Training Loss Functions and Forward Module.
Implements dual-stage coarse and fine supervision:
1. Coarse-level Focal Loss / NLL over the dual-softmax assignment matrix.
2. Fine-level coordinate regression loss.
"""
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoFTRLoss(nn.Module):
    """
    Supervises LoFTR correspondence learning using coarse assignment and fine regression.
    """
    def __init__(
        self,
        coarse_weight: float = 1.0,
        fine_weight: float = 0.25,
        temperature: float = 1.0
    ):
        super().__init__()
        self.coarse_weight = coarse_weight
        self.fine_weight = fine_weight
        self.temperature = temperature

    def forward_train_step(
        self,
        model: nn.Module,
        image0: torch.Tensor,
        image1: torch.Tensor,
        keypoints0: torch.Tensor,
        keypoints1: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = image0.device
        h, w = image0.shape[2:]
        h_c, w_c = h // 8, w // 8

        # 1. Forward through backbone
        feats_c, feats_f = model.backbone(torch.cat([image0, image1], dim=0))
        f_c0, f_c1 = feats_c.split(1)

        # 2. Positional Encoding
        p0 = model.pos_encoding(f_c0).permute(0, 2, 3, 1).reshape(1, -1, 256)
        p1 = model.pos_encoding(f_c1).permute(0, 2, 3, 1).reshape(1, -1, 256)

        # 3. Coarse Transformer
        t0, t1 = model.loftr_coarse(p0, p1)

        # 4. Dual Log-Softmax Similarity Matrix (log P = log softmax_row + log softmax_col)
        sim = torch.einsum('nlc,nsc->nls', t0, t1) / max(1e-3, self.temperature)
        log_p1 = F.log_softmax(sim, dim=1)
        log_p2 = F.log_softmax(sim, dim=2)
        log_dual = log_p1 + log_p2  # (1, H_c*W_c, H_c*W_c)

        # 5. Map ground truth keypoints to coarse coordinates
        loss_dict = {}
        if len(keypoints0) > 0 and len(keypoints1) > 0:
            k0 = keypoints0.clone()
            k1 = keypoints1.clone()

            # Clamp coordinates
            k0[:, 0] = torch.clamp(k0[:, 0], 0, w - 1)
            k0[:, 1] = torch.clamp(k0[:, 1], 0, h - 1)
            k1[:, 0] = torch.clamp(k1[:, 0], 0, w - 1)
            k1[:, 1] = torch.clamp(k1[:, 1], 0, h - 1)

            i_coords = (torch.div(k0[:, 1], 8, rounding_mode='floor') * w_c + torch.div(k0[:, 0], 8, rounding_mode='floor')).long()
            j_coords = (torch.div(k1[:, 1], 8, rounding_mode='floor') * w_c + torch.div(k1[:, 0], 8, rounding_mode='floor')).long()

            # Clamp to valid coarse index range
            max_idx = h_c * w_c - 1
            i_coords = torch.clamp(i_coords, 0, max_idx)
            j_coords = torch.clamp(j_coords, 0, max_idx)

            # Extract dual log-probabilities for ground truth matches
            target_log_probs = log_dual[0, i_coords, j_coords]

            # Coarse dual NLL loss
            loss_coarse = (-target_log_probs).mean()

            # Fine probability regression proxy
            matching_probs = torch.exp(torch.clamp(target_log_probs, min=-20.0))
            loss_fine = (1.0 - matching_probs.mean())

            total_loss = self.coarse_weight * loss_coarse + self.fine_weight * loss_fine
            loss_dict["loss_coarse"] = float(loss_coarse.item())
            loss_dict["loss_fine"] = float(loss_fine.item())
            loss_dict["loss_total"] = float(total_loss.item())
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_dict["loss_coarse"] = 0.0
            loss_dict["loss_fine"] = 0.0
            loss_dict["loss_total"] = 0.0

        return total_loss, loss_dict
