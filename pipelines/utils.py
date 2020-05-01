import os
import sys


def query_results_dir():
    return os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else None
