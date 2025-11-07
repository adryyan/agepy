import argparse


def main():
    args = parser().parse_args()


def parser():
    parser = argparse.ArgumentParser(prog="phexphem_eval")

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information.",
    )

    return parser
