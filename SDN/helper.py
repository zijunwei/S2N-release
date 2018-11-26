import torch


def switch_positions(head_positions, tail_positions):
    pred_positions = torch.stack([head_positions, tail_positions], dim=-1)
    head_positions, _ = torch.min(pred_positions, dim=2)
    tail_positions, _ = torch.max(pred_positions, dim=2)
    return head_positions, tail_positions


if __name__ == '__main__':

    head_position = torch.rand([5, 2])
    tail_position = torch.rand([5, 2])
    updated_head, updated_tail = switch_positions(head_position, tail_position)
    print("DB")
