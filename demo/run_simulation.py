import argparse
import logging
import os
import pathlib

import todd
import torch

from constellation import Controller, TaskManager
from constellation.algorithms import ModelAlgorithm
from constellation.callbacks import ComposedCallback
from constellation.data import Constellation, TaskSet
from constellation.environments import SatsimEnvironment

from .backend import CACHE_ROOT
from .callbacks import RecordCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_id', type=str)
    parser.add_argument('--greedy', action='store_true')
    args = parser.parse_args()
    return args


def run_demo(args: argparse.Namespace) -> None:
    thread_id: str = args.thread_id
    work_dir = CACHE_ROOT / thread_id

    constellation_path = work_dir / 'constellation.json'
    task_path = work_dir / 'taskset.json'
    constellation = Constellation.load(str(constellation_path))
    taskset = TaskSet.load(str(task_path))

    environment = SatsimEnvironment(
        constellation=constellation,
        all_tasks=taskset,
    )
    task_manager = TaskManager(timer=environment.timer, taskset=taskset)
    callbacks = ComposedCallback(
        callbacks=[
            RecordCallback(log_dir=work_dir / 'result'),
        ],
    )
    controller = Controller(
        pathlib.Path(__file__).stem,
        environment=environment,
        task_manager=task_manager,
        callbacks=callbacks,
    )

    algorithm = ModelAlgorithm(timer=environment.timer, greedy=args.greedy)
    algorithm.prepare(environment, task_manager)

    try:
        controller.run(algorithm, progress_bar=False, max_time_step=7200)
    except Exception as e:
        todd.logger.error("thread_id %s failed: %s", thread_id, e)
        with open(work_dir / 'failed.txt', 'w') as f:
            f.write(str(e))
        raise e


def main() -> None:
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
