import argparse


def parse_args() -> argparse.Namespace:
    r"""Parses the command line arguments."""
    parser = argparse.ArgumentParser(description='GraphGym')

    parser.add_argument('--cfg',
                        dest='cfg_file',
                        type=str,
                        required=True,
                        help='The configuration file path.')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help='The number of repeated jobs.')
    parser.add_argument('--mark_done',
                        action='store_true',
                        help='Mark yaml as done after a job has finished.')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='The index of gpu to run the experiment (0-N), default=-1, randomly select one.')
    parser.add_argument('opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')

    return parser.parse_args()
