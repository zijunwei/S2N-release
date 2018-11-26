import torch


def reorder(head_positions, tail_positions):
    pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
    head_positions, _ = torch.min(pred_positions, dim=2)
    tail_positions, _ = torch.max(pred_positions, dim=2)
    return head_positions, tail_positions


