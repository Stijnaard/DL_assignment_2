import argparse
from src.config.config import *

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog = """
            Examples:
            python main.py                          # use config.py defaults
            python main.py --model lstm              # train LSTM only
            python main.py --model all              # train all models
            python main.py --model cnn_transformer --experiment intra
            python main.py --eval-only              # skip training, only plot
            """)

    parser.add_argument(
        "--model", type = str, default = MODEL,
        choices = ALL_MODELS + ["all"]
    )
    parser.add_argument(
        "--experiment", type = str, default = "both",
        choices = ["intra", "cross", "both"]
    )
    parser.add_argument(
        "--eval-only", action = "store_true"
    )
    return parser.parse_args()
