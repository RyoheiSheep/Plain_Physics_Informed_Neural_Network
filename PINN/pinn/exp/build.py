import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))

        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp()

    except Exception:
        raise ImportError(f"{exp_file} does not contains class named 'Exp'.")
    return exp
