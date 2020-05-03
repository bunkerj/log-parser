import os
import sys


def get_results_dir_from_args():
    return os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
