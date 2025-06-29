import argparse
import os


def get_hparams():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--num_lang", type=int, default=2)
    parser.add_argument("--float32_matmul_precision", type=str, default="high")
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--clip_duration", type=int, default=10)
    parser.add_argument("--min_clip_duration", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=0.97)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=35)
    parser.add_argument("--accumulate_grad_batches", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="./results")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--test_size", type=float, default=0.1)

    args = parser.parse_args()
    args.save_path = os.path.expanduser(args.save_path)
    args.data_path = os.path.expanduser(args.data_path)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args
