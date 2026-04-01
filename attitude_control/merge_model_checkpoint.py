import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'stage1_checkpoint',
        type=str,
    )
    parser.add_argument(
        'stage2_checkpoint',
        type=str,
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='merged_model_checkpoint.pth',
        help='Path to save the merged checkpoint.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    stage1_ckpt = torch.load(args.stage1_checkpoint)
    stage2_ckpt = torch.load(args.stage2_checkpoint)

    merged_ckpt = {f'_stage1_model.{k}': v for k, v in stage1_ckpt.items()}
    merged_ckpt.update({
        f'_stage2_model.{k}': v
        for k, v in stage2_ckpt.items()
    })

    os.makedirs(
        os.path.dirname(args.output_file),
        exist_ok=True,
    )
    torch.save(merged_ckpt, args.output_file)
