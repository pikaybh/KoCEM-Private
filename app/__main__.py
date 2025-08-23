from fire import Fire

from run_each import run_each
from eval_difficulties import eval_difficulties
from update_readme import update_readme


def run_sequentially(**kwargs):
    """
    Run both the run_each and eval_difficulties functions sequentially.
    """
    run_each(**kwargs)
    eval_difficulties(**kwargs)
    update_readme()


if __name__ == '__main__':
    Fire({
        "eval_difficulties": eval_difficulties,
        "run_auto": run_sequentially,
        "run_each": run_each,
        "update": update_readme
    })