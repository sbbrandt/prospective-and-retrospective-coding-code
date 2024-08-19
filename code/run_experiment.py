import sys, os
import config
from experiment import (
    run_augmented_parameters,
    run_frequency_response,
    run_sbi,
    run_posterior_check,
    run_sinusoidals,
)

import uuid
import argparse
import logging
import json
from copy import deepcopy

experiments = {
    "augmented_parameters": run_augmented_parameters,
    "sbi": run_sbi,
    "posterior_check": run_posterior_check,
    "frequency_response": run_frequency_response,
    "sinusoidals": run_sinusoidals,
}

standalone_ready_experiments = ["sbi"]


def run_experiment(
    experiment_name,
    experiment_id,
    hashed_save_dir=False,
    on_cluster=False,
    standalone=False,
):
    # experiment_name = "sbi"
    # experiment_id = "001"
    assert experiment_name in config.experiment.keys()
    assert experiment_id in config.experiment[experiment_name].keys()
    print("run {} {}".format(experiment_name, experiment_id))

    # get copy of experiment arguments
    experiment_kw = deepcopy(config.experiment[experiment_name][experiment_id])

    print("experiment parameters:")
    for k, v in experiment_kw.items():
        print("\t{}: {}".format(k, v))

    save_dir = experiment_kw["save_dir"]
    # replace ~ with 'HOME'
    if save_dir[0] == "~":
        assert save_dir[1] == "/"
        save_dir = os.path.join(os.environ.get("HOME"), save_dir[2:])
    # add directory for experiment id
    save_dir = os.path.join(save_dir, experiment_name, experiment_id)

    # add custom directory for single run
    if hashed_save_dir:
        run_id = str(uuid.uuid4())
        save_dir = os.path.join(save_dir, run_id)
    else:
        if os.path.exists(save_dir):
            # update save directory if already exists
            save_dir += "_1"
            while os.path.exists(save_dir):
                save_dir_idx = int(save_dir.split("_")[-1])
                save_dir = "_".join(save_dir.split("_")[:-1] + [str(save_dir_idx + 1)])

    print("save to: {}".format(save_dir))

    # make directories
    os.makedirs(save_dir)

    log_level = "DEBUG"
    logging.basicConfig(
        filename=os.path.join(save_dir, f"{log_level}.log"),
        level=log_level,
        format="%(asctime)s - %(message)s",
    )

    # update copied experiment arguments
    experiment_kw["save_dir"] = save_dir

    # to run sbi in standalone mode
    if standalone:
        assert experiment_name in standalone_ready_experiments
        experiment_kw["standalone"] = standalone

    with open(os.path.join(save_dir, "config.json"), "w") as tmp:
        json.dump(experiment_kw, tmp, indent=4)
    # run experiment
    experiments[experiment_name](**experiment_kw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name")
    parser.add_argument("--experiment_id")
    parser.add_argument("--hashed_save_dir", action="store_true")
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--standalone", action="store_true")

    args = parser.parse_args()

    experiment_name = args.experiment_name
    experiment_id = args.experiment_id
    hashed_save_dir = args.hashed_save_dir
    on_cluster = args.on_cluster
    standalone = args.standalone

    if experiment_name in standalone_ready_experiments:
        run_experiment(
            experiment_name, experiment_id, hashed_save_dir, on_cluster, standalone
        )
    else:
        run_experiment(experiment_name, experiment_id, hashed_save_dir, on_cluster)
