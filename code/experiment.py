import logging
import sys, os
import pickle
import json
import numpy as np
import uuid
from copy import deepcopy

import config
from model import HH_model

# from viz import phase_shift, psth
from analysis import collect_rounds, get_phase_from_save_dir, collect_save_dirs, get_variables, get_augmented_parameters, collect_run_ids
from utils import (
    eval_parameter,
    resolve_parameter_dependencies,
    augment_parameters,
    get_prior,
)
from viz import psth, phase_shift
import joblib
import time


def scale_variable(
    variable, factor, model_type, model_name, parameter_type="parameters"
):
    """Scales a parameter or variable from config.model_dynamics by a factor.

    Parameters
    ----------
    variable : str
        Name of the variable to scale. Can be either a parameter (e.g. 'g_na')
        or a variable (e.g. 'dh/dt', 'alpha_m', ...).
    factor : float
        The factor by which the parameter or the right hand side of the variable
        equation should be multiplicated by.
    model_name : str
        The name of the model, which parameters should be used as default
        parameters. See 'config.model_dynamics' for more info.
    parameter_type : str, optional
        The type of the parameter. Can be either 'parameters' (for parameters)
        or 'eqs' (for equations/variables). See
        'config.model_dynamics.<model_name>' for more info.
        By default 'parameters'.

    Returns
    -------
    str
        The scaled parameter value or right hand side of the variable equation.
    """
    assert parameter_type in ["parameters", "eqs"]
    assert model_type in config.model_dynamics.keys()
    assert model_name in config.model_dynamics[model_type].keys()

    if parameter_type == "eqs":
        p = config.model_dynamics[model_type][model_name]["eqs"]
        eqs = p.replace(" ", "")
        eqs = eqs.split("\n")
        vs = [eq.split("=")[0] for eq in eqs]
        # vs = [v[0] for eq in eqs for v in eq.split('=')]
        i = vs.index(variable)
        eq = eqs[i]
        parameter = eq.split("=")[1]
    if parameter_type == "parameters":
        p = deepcopy(config.model_dynamics[model_type][model_name]["parameters"])
        parameter = p[variable]
        if type(parameter) is not str:
            parameter = "{}".format(parameter)
    # if parameter_type == 'simulation_parameters':
    #     parameter = variable

    parameter = "{}*{}".format(factor, parameter)

    return parameter


def shift_variable(
    variable, shift, model_type, model_name, parameter_type="parameters"
):
    assert parameter_type in ["parameters"]
    assert model_type in config.model_dynamics.keys()
    assert model_name in config.model_dynamics[model_type].keys()

    if parameter_type == "parameters":
        p = deepcopy(config.model_dynamics[model_type][model_name]["parameters"])
        parameter = p[variable]
        if type(parameter) is not str:
            parameter = "{}".format(parameter)
    # if parameter_type == 'simulation_parameters':
    #     parameter = variable

    assert len(parameter.split("*")) == 2

    value, unit = parameter.split("*")

    parameter = "{}*{}".format(float(value) + float(shift), unit)

    return parameter


def save_augmented_params(
    save_dir,
    augmented_model_params,
    augmented_eqs,
    augmented_simulation_params,
    file_pre_str="",
):
    """Utility function used by 'run_augmented_parameters' to store the parameters.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    augmented_model_params : dict
        Contains parameter value pairs of the changed model parameters.
    augmented_eqs : dict
        Contains variable equation pairs of the changed value equations.
    fixed_simulation_params : dict
        Contains parameter value pairs of the simulation parameters to change.
    file_pre_str : str, optional
        String that is joined with 'variables.p' by '_' to form
        the name of the file. Usually the 'run_id'. By default ''.
    """
    save_dict = {
        "model_parameters": augmented_model_params,
        "eqs": augmented_eqs,
        "simulation_parameters": augmented_simulation_params,
    }
    save_str = "_".join([file_pre_str, "augmented_parameters.json"])
    with open(os.path.join(save_dir, save_str), "w") as tmp:
        json.dump(save_dict, tmp, indent=4)


def store_variable(save_dict, m, variable, i):
    """Utility function to store a variable in a provided dictionary.

    Parameters
    ----------
    save_dict : dict
        The dictionary to store the variable in.
    m : HH_model
        The model from which the variable should be obtained.
    variable : str
        The name of the variable to store. Must be implemented in the model.
    i : str, int, None
        Can be either 'mean', 'all', the index of the neuron or 'None', if the
        variable should not be stored but is assessed within the experiment.
    """

    if i == "mean":
        i = {"mean": True, "i": None}
    elif type(i) is dict:
        i = deepcopy(i)
        i.setdefault("mean", False)
    elif type(i) in [int, list, np.array]:
        i = {"mean": False, "i": i}

    if i is not None:
        # mean = True if i['i'] is None else False
        t, y = m.get_variable(variable, mean=False)

        if variable not in save_dict.keys():
            save_dict[variable] = {}
        # save_dict[variable]['t'] = t

        if variable in ["psth", "spikes"]:
            i == "all"
            save_dict[variable]["t"] = t
            save_dict[variable][i] = y
        else:
            if i["mean"]:
                save_dict[variable]["mean"] = y.mean(0)
            if "i" in i.keys():
                idx = i["i"]
                if type(idx) == int:
                    # store by index
                    save_dict[variable][idx] = y[idx]
                elif type(idx) in [list, np.array]:
                    # store each index separately
                    for i_ in idx:
                        save_dict[variable][i_] = y[i_]
                else:
                    raise ValueError("'i={}' not supported.".format(i))


def run_augmented_parameters(
    model_type,
    simulation_name,
    model_name,
    augmented_model_params={},
    augmented_eqs={},
    augmented_simulation_params={},
    fixed_simulation_params={},
    save_variables={},
    save_spikes=False,
    save_dependent_variables={},
    save_dir=".",
    file_pre_str="",
    logger=logging.getLogger(__name__),
    fixed_model_params={},
    min_spikes=0,
    seed=None,
    fixed_eqs={},
):

    for param in fixed_model_params.keys():
        assert param not in fixed_simulation_params.keys()
    save_augmented_params(
        save_dir=save_dir,
        augmented_model_params=augmented_model_params,
        augmented_eqs=augmented_eqs,
        augmented_simulation_params=augmented_simulation_params,
        file_pre_str=file_pre_str,
    )

    model_params, eqs, simulation_params = augment_parameters(
        model_type=model_type,
        model_name=model_name,
        simulation_name=simulation_name,
        augmented_model_params=augmented_model_params,
        augmented_eqs=augmented_eqs,
        augmented_simulation_params=augmented_simulation_params,
        fixed_model_params=fixed_model_params,
        fixed_eqs=fixed_eqs,
        fixed_simulation_params=fixed_simulation_params,
    )

    for key, value in fixed_model_params.items():
        model_params[key] = value

    variables = list(save_variables.keys())
    save_dict = {}

    logger.debug("run_augmented_parameters - run get_model...")
    m = get_model(
        model_type,
        simulation_parameters=deepcopy(simulation_params),
        model_parameters=deepcopy(model_params),
        eqs=eqs,
        variables=variables,
        save_spikes=save_spikes,
        seed=seed,
        logger=logger,
    )
    logger.debug("run_augmented_parameters - run get_model... done")

    if type(m) is np.float64:
        assert np.isnan(m)
        valid_run = False
    elif len(m.get_variable("spikes")[0]) < min_spikes:
        n_spikes = len(m.get_variable("spikes")[0])
        logger.info(
            "n_spikes = {} < min_spikes = {}. Model is invalid.".format(
                n_spikes, min_spikes
            )
        )
        valid_run = False
    else:

        logger.debug("Save variables...")
        t = time.time()

        if save_spikes:
            store_variable(save_dict, m, "spikes", "all")
        for v, i in save_variables.items():
            store_variable(save_dict, m, v, i)
        for v, i in save_dependent_variables.items():
            store_variable(save_dict, m, v, i)

        save_dict["model_parameters"] = model_params
        save_dict["simulation_parameters"] = simulation_params
        save_dict["eqs"] = eqs

        # if standalone:
        #     m.reinit()

        save_str = "_".join([file_pre_str, "variables.p"])
        with open(os.path.join(save_dir, save_str), "wb") as tmp:
            pickle.dump(save_dict, tmp)
        logger.debug(
            "Save variables... done. Took {} seconds.".format(
                np.round(time.time() - t, 1)
            )
        )
        # if early_stop_negative_rheobase:
        valid_run = True  # valid_run
        del m.M
    del save_dict
    return valid_run


class sbi_simulator:
    """Simulator for the sbi experiment.

    In the sbi experiment, the simulator is called to run a simulation. Thus,
    each call initialises a new model with scaling factors that are provided by
    the call. The model is then trained and specified variables are observed and
    saved. The initialisation of the simulator only stores experiment related
    parameters that should be available during the call of the simulator.

    Parameters
    ----------
    simulation_name : str
        The name of the simulation parameters. See 'config.simulation' for more
        info.
    model_name : str
        The name of the model, which parameters should be used. See
        'config.model_dynamics' for more info.
    augmented_model_params : list of str
        The model parameters that should be scaled by factor.
    augmented_eqs : list of str
        The variables defined by eqs that should be scaled by factor.
    fixed_simulation_params : dict
        Simulation parameters that should be different from the parameters in
        'config.<simulation_name>'.
    I0 : str, int, float
        In units of 'I_unit'. Sets the constant term of the input current to the
        system. If set to 'rheobase', the rheobase current will be computed and
        used.
    I1 : int, float
        In units of 'I_unit'. Sets the amplitude of sinusoidal term of the input
        current.
    s : int, float
        In units of 'I_unit'. Sets the variance of the noise term of the input
        current
    I_unit : str
        Sets the unit of the input current. Usually 'nA'.
    adapt_threshold : bool
        If set to 'True', the threshold in 'simulation_params' will be set to
        the voltage at the rheobase current plus 'thr_offset'
        (v_thr = v(I0=I_rh, I1=0, s=0) + thr_offset).
    thr_offset : int, float
        Only used if 'adapt_threshold = True'. Defined how much the threshold
        should be above (or below) the rheobase resting voltage. If to low, the
        simulation will record spikes that are caused by voltage fluctuations
        and not only spikes. For 'I1=0.01' and 's=0.02', an offset of 10 is a
        good choice. By default 10.
    model : HH_model
        Currently only supports 'model.HH_model'.
    rheobase_kw : dict, optional
        Parameters used to compute the rheobase. Passed to 'find_rheobase'.
        By default {}.
    save_variables : dict
        The variables that should be saved each run. Many runs can lead to very
        high storage demand. Use with care! Expected format:
        '{<variable_name>: i}' where 'i' can be the index of a neuron, 'all' or
        'mean' (recommended).
    save_spikes : bool
        Whether to record spikes.
    save_dir : str, optional
        The path to the save directory. By default '.'.
    logger : logging.Logger, optional
        Logger instance to be called for logging.
        By default logging.getLogger(__name__).
    """

    def __init__(
        self,
        model_type,
        simulation_name,
        model_name,
        augmented_model_params=[],
        augmented_eqs=[],
        augmented_simulation_params={},
        fixed_simulation_params={},
        save_variables={},
        save_spikes=False,
        save_dependent_variables={},
        save_dir=".",
        logger=logging.getLogger(__name__),
        log_to_file_pre_str=True,
        min_spikes=0,
        sbi_observation="phase",
        fixed_model_parameters={},
        augment_by="factor",
        log10=False,
        seed=None,
    ):
        import torch

        self.model_type = model_type
        self.simulation_name = simulation_name
        self.model_name = model_name
        self.augmented_model_params = augmented_model_params
        self.augmented_eqs = augmented_eqs
        self.augmented_simulation_params = augmented_simulation_params
        self.fixed_simulation_params = fixed_simulation_params
        self.save_variables = save_variables
        self.save_spikes = save_spikes
        self.save_dependent_variables = save_dependent_variables
        self.save_dir = save_dir
        self.logger = logger
        self.n_theta = len(augmented_model_params + augmented_eqs) + len(
            augmented_simulation_params
        )
        self.log_to_file_pre_str = log_to_file_pre_str
        self.sbi_observation = sbi_observation
        self.fixed_model_parameters = fixed_model_parameters
        self.augment_by = augment_by
        self.log10 = log10
        self.min_spikes = min_spikes
        self.seed = seed

    def __call__(self, theta):
        """Runs one simulation in the sbi context

        Parameters
        ----------
        theta : torch.tensor
            the parameters drawn by sbi ordered by category, first model
            parameters, then equation variables. In the scaled context, theta
            contains the scaling factors.

        Returns
        -------
        list
            List of phases (observations) that match the scaling factors theta.
        """
        import torch

        if (len(theta.size()) > 1) and (theta.size()[0] > 1):
            # sbi tests whether batch processing is supported
            # raise error at the beginning as it is not supported
            raise ValueError

        simulation_id = str(uuid.uuid4())
        file_pre_str = simulation_id  # run_id put in front of every saved file
        if self.log10:
            theta = 10**theta
        # if theta.shape[0] == 2:
        #     raise ValueError

        def get_file_logger(save_dir, file_pre_str, log_level="DEBUG"):
            logger = logging.getLogger(file_pre_str)
            log_handler = logging.handlers.TimedRotatingFileHandler(
                os.path.join(save_dir, file_pre_str + ".log")
            )
            logger.addHandler(log_handler)
            logger.setLevel(log_level)
            return logger

        if self.log_to_file_pre_str:
            logger = get_file_logger(self.save_dir, file_pre_str)
            # logger = logging.getLogger(file_pre_str)
            # log_handler = logging.handlers.TimedRotatingFileHandler(
            #     os.path.join(self.save_dir, file_pre_str + '.log'))
            # logger.addHandler(log_handler)
            # logger.setLevel('DEBUG')
        else:
            logger = self.logger

        logger.debug("sbi_simulator.__call__ input theta: {}".format(theta))

        augmented_model_params = {}
        augmented_eqs = {}
        augmented_simulation_params = {}
        fixed_simulation_params = deepcopy(self.fixed_simulation_params)

        # for model_param, value in self.fixed_model_parameters.items():
        #     augmented_model_params[model_param] = value

        theta = theta.reshape(self.n_theta)

        i = 0
        for k in self.augmented_model_params:
            if self.augment_by == "factor":
                p = scale_variable(
                    k,
                    factor=theta[i],
                    model_type=self.model_type,
                    model_name=self.model_name,
                    parameter_type="parameters",
                )
            elif self.augment_by == "shift":
                p = shift_variable(
                    k,
                    shift=theta[i],
                    model_type=self.model_type,
                    model_name=self.model_name,
                    parameter_type="parameters",
                )
            else:
                raise NotImplementedError(
                    "'sbi_simulator.augment_by' {}".format(self.augment_by)
                )
            augmented_model_params[k] = p
            i += 1

        # This will not add thetas if eqs is an empty list
        for k in self.augmented_eqs:
            if self.augment_by == "factor":
                p = scale_variable(
                    k,
                    factor=theta[i],
                    model_type=self.model_type,
                    model_name=self.model_name,
                    parameter_type="eqs",
                )
            else:
                raise NotImplementedError
            augmented_eqs[k] = p
            i += 1
        for k, v in self.augmented_simulation_params.items():
            if self.augment_by == "factor":
                p = "{}*{}".format(theta[i], v)
                # p = scale_variable(v, factor=theta[i], model_type=self.model_type,
                #                 model_name=self.model_name, parameter_type='simulation_parameters')
            else:
                raise NotImplementedError
            augmented_simulation_params[k] = p
            i += 1

        logger.debug("sbi_simulator.__call__ - run_augmented_parameters...")
        logger.debug("sbi_simulator.__call__ - seed is {}".format(self.seed))
        valid_run = run_augmented_parameters(
            model_type=self.model_type,
            simulation_name=self.simulation_name,
            model_name=self.model_name,
            augmented_model_params=augmented_model_params,
            augmented_eqs=augmented_eqs,
            augmented_simulation_params=augmented_simulation_params,
            fixed_simulation_params=fixed_simulation_params,
            save_variables=self.save_variables,
            save_spikes=self.save_spikes,
            save_dependent_variables=self.save_dependent_variables,
            save_dir=self.save_dir,
            file_pre_str=file_pre_str,
            logger=logger,
            min_spikes=self.min_spikes,
            fixed_model_params=self.fixed_model_parameters,
            seed=self.seed,
        )
        logger.debug("sbi_simulator.__call__ - run_augmented_parameters... done.")

        if valid_run:
            logger.debug("sbi_simulator.__call__ - valid model.")
            logger.debug("sbi_simulator.__call__ - run get_observation...")
            try:
                observation = self.get_observation(
                    file_pre_str=file_pre_str,
                    sbi_observation=self.sbi_observation,
                    model_type=self.model_type,
                    model_name=self.model_name,
                    simulation_name=self.simulation_name,
                    augmented_model_params=augmented_model_params,
                    fixed_simulation_params=self.fixed_simulation_params,
                    logger=logger,
                )
            except RuntimeError:
                observation = np.float64("nan")

            logger.debug("sbi_simulator.__call__ - run get_observation... done.")
        else:
            observation = np.float64("nan")

        logger.debug("sbi_simulator.__call__ - return {}".format(observation))

        return torch.tensor([observation])

    def get_observation(
        self,
        file_pre_str,
        sbi_observation,
        model_type,
        model_name,
        simulation_name,
        augmented_model_params,
        fixed_simulation_params,
        logger=logging.getLogger(__name__),
    ):
        observations = {
            "phase": self._get_phase_from_HH_model,
        }

        observation = observations[sbi_observation](
            file_pre_str,
            model_type,
            model_name,
            simulation_name,
            augmented_model_params,
            fixed_simulation_params,
            logger=logger,
        )
        return observation

    def _get_phase_from_HH_model(
        self,
        file_pre_str,
        model_type,
        model_name,
        simulation_name,
        augmented_model_params,
        fixed_simulation_params,
        initialize=True,
        logger=logging.getLogger(__name__),
    ):
        from brian2 import Hz

        if "f" in augmented_model_params:
            f = eval_parameter(augmented_model_params["f"]) / Hz
        else:
            model_params = config.model_dynamics[model_type][model_name]["parameters"]
            if "f_exp" in model_params:
                if "f_exp" in augmented_model_params:
                    model_params["f_exp"] = augmented_model_params["f_exp"]
                resolve_parameter_dependencies(model_params)
            f = eval_parameter(model_params["f"]) / Hz
        if "N" in fixed_simulation_params:
            N = fixed_simulation_params["N"]
        else:
            N = config.simulation[simulation_name]["N"]

        phase = get_phase_from_save_dir(
            self.save_dir,
            file_pre_str,
            v1={"psth": "all"},
            v2={"I": "mean"},
            hist_kw={"bins": 100},
            N=N,
            phase_shift_kw={"f_0": f},
            initialize=initialize,
            logger=logger,
        )
        return phase


def save_sbi_simulation(theta, x, save_str="", save_dir="."):
    """Saves simulation results from sbi simulation.

    Parameters
    ----------
    theta : any
        The parameters that were drawn from the prior.
    x : any
        The simulation results (observations).
    save_str : str, optional
        The name of the file that should be saved in 'save_dir'. By default ''.
    save_dir : str, optional
        The path to the save directory. By default '.'.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_str)
    save_dict = {"theta": theta, "x": x}
    with open(save_path, "wb") as tmp:
        pickle.dump(save_dict, tmp)


def run_sbi(
    model_type,
    simulation_name,
    model_name,
    augmented_model_params,
    augmented_eqs,
    fixed_simulation_params,
    save_variables,
    save_spikes,
    num_rounds,
    simulations_per_round,
    num_workers,
    save_dir,
    augmented_simulation_params={},
    save_dependent_variables={},
    fixed_model_params={},
    seed=None,
    prior_distribution="uniform",
    low=None,
    high=None,
    prior_distribution_kw={},
    logger=logging.getLogger(__name__),
    min_spikes=0,
    rounds_from_id=None,
    augment_by="factor",
    log10=False,
):
    """Wrapper to run the sbi experiment.

    In multiple rounds, multiple models are run, where for each model, scaling
    factors for parameters and variables are drawn from a prior distribution.
    At the end of each round, the prior is restricted and updated, s.t. areas
    in the parameter space are avoided that lead to invalid simulations. Each
    run and at the end of each round the summary of scaling factors
    and phase shifts (observations) is saved. All runs are used to compute the
    posterior distribution of the parameters (scaling factors) given the
    observations. The posterior can also afterwards be computed from the saved
    summary or run results.

    Parameters
    ----------
    simulation_name : str
        The name of the simulation parameters. See 'config.simulation' for more
        info.
    model_name : str
        The name of the model, which parameters should be used. See
        'config.model_dynamics' for more info.
    augmented_model_params : list of str
        The model parameters that should be scaled by factor.
    augmented_eqs : list of str
        The variables defined by eqs that should be scaled by factor.
    fixed_simulation_params : dict
        Simulation parameters that should be different from the parameters in
        'config.<simulation_name>'.
    I0 : str, int, float
        In units of 'I_unit'. Sets the constant term of the input current to the
        system. If set to 'rheobase', the rheobase current will be computed and
        used.
    I1 : int, float
        In units of 'I_unit'. Sets the amplitude of sinusoidal term of the input
        current.
    s : int, float
        In units of 'I_unit'. Sets the variance of the noise term of the input
        current
    I_unit : str
        Sets the unit of the input current. Usually 'nA'.
    adapt_threshold : bool
        If set to 'True', the threshold in 'simulation_params' will be set to
        the voltage at the rheobase current plus 'thr_offset'
        (v_thr = v(I0=I_rh, I1=0, s=0) + thr_offset).
    thr_offset : int, float
        Only used if 'adapt_threshold = True'. Defined how much the threshold
        should be above (or below) the rheobase resting voltage. If to low, the
        simulation will record spikes that are caused by voltage fluctuations
        and not only spikes. For 'I1=0.01' and 's=0.02', an offset of 10 is a
        good choice.
    save_variables : dict
        The variables that should be saved each run. Many runs can lead to very
        high storage demand. Use with care! Expected format:
        '{<variable_name>: i}' where 'i' can be the index of a neuron, 'all' or
        'mean' (recommended).
    save_spikes : bool
        Whether to record spikes.
    low : float
        Parameter for uniform distribution used as prior.
    high : float
        Parameter for uniform distribution used as prior.
    num_rounds : int
        Number of rounds to simulate. Each round, 'simulations_per_round' are
        simulated and after each round the prior is restricted and updated
        (leading to more efficient parameter space exploration).
    simulations_per_round : int
        Number of simulations to run each round.
    num_workers : int
        Passed to 'joblib.Parallel'. '-1' for all workers.
    save_dir : str
        Path to directory where run results and summary should be saved.
    rheobase_kw : dict, optional
        Parameters used to compute the rheobase. Passed to 'find_rheobase'.
        By default {}.
    logger : logging.Logger, optional
        Logger instance to be called for logging.
        By default logging.getLogger(__name__).
    early_stop_negative_rheobase : bool, optional
        Only works with "I0='rheobase'". Treats parameters for which the
        rheobase current is negative as invalid simulations. The restricted
        prior is trained to avoid parameter regimes that lead to invalid
        simulations and thus avoids the regime of negative rheobase currents.
    rounds_from_id : None or str, optional
        Experiment id to collect simulation rounds from to train the restricted
        prior. The experiment id should have the same sbi parameters.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    import torch
    import sbi
    from sbi import utils
    from sbi import inference

    # sbi parameter dimension
    num_dim = len(augmented_model_params + augmented_eqs) + len(
        augmented_simulation_params
    )

    prior = get_prior(
        prior_distribution=prior_distribution,
        prior_distribution_kw=prior_distribution_kw,
        num_dim=num_dim,
        low=low,
        high=high,
    )

    # get sbi simulator and prior
    simulator = sbi_simulator(
        model_type=model_type,
        simulation_name=simulation_name,
        model_name=model_name,
        augmented_model_params=augmented_model_params,
        augmented_eqs=augmented_eqs,
        augmented_simulation_params=augmented_simulation_params,
        fixed_simulation_params=fixed_simulation_params,
        save_variables=save_variables,
        save_spikes=save_spikes,
        save_dependent_variables=save_dependent_variables,
        fixed_model_parameters=fixed_model_params,
        save_dir=save_dir,
        logger=logger,
        min_spikes=min_spikes,
        augment_by=augment_by,
        log10=log10,
        seed=seed,
    )

    logger.debug("prepare for sbi ...\n")
    t = time.time()
    simulator, prior = sbi.inference.prepare_for_sbi(simulator, prior)
    logger.debug(
        "prepare for sbi...done. Took {} seconds.\n".format(
            np.round(time.time() - t, 2)
        )
    )

    # the restricted estimator will avoid parameters that lead to invalid simulations

    ###
    inference = sbi.inference.SNPE(prior=prior)
    ###

    restriction_estimator = sbi.utils.RestrictionEstimator(prior=prior)
    proposals = [prior]
    if rounds_from_id is not None:
        # proposals = []
        if type(rounds_from_id) is str:
            rounds_from_id = [rounds_from_id]
        else:
            assert type(rounds_from_id) is list
        # experiment_dir = '/'.join(save_dir.split('/')[:-1])
        save_dirs = []
        for exp_id in rounds_from_id:
            save_dirs += collect_save_dirs(exp_id)
        thetas, phases = collect_rounds(save_dirs)
        for theta, x in zip(thetas, phases):
            ###
            inference.append_simulations(theta, x, proposal=proposals[-1])
            ###
            restriction_estimator.append_simulations(theta, x)
            restriction_estimator.train()
            proposals.append(restriction_estimator.restrict_prior())

    if seed is not None:
        logger.info("torch seed set to {}".format(seed))
        torch.manual_seed(seed)
    # update restricted prior each round
    for r in range(num_rounds):
        logger.info("round {}".format(r))
        theta, x = sbi.inference.simulate_for_sbi(
            simulator,
            proposals[-1],
            num_simulations=simulations_per_round,
            num_workers=num_workers,
        )

        logger.debug("theta:\n{}\nx: {}".format(theta, x))
        logger.debug("save simulation...\n")
        t = time.time()
        save_sbi_simulation(
            theta,
            x,
            save_str="simulation_results_round_{}.p".format(r),
            save_dir=save_dir,
        )
        logger.debug(
            "save simulation...done. Took {} seconds.\n".format(
                np.round(time.time() - t, 2)
            )
        )

        logger.debug("append simulations...\n")
        t = time.time()
        restriction_estimator.append_simulations(theta, x)
        logger.debug(
            "append simulations...done. Took {} seconds.\n".format(
                np.round(time.time() - t, 2)
            )
        )

        ###
        inference.append_simulations(theta, x, proposal=proposals[-1])
        ###

        # if (r == 0) or (r < num_rounds - 1): # training not needed in last round because classifier will not be used anymore.
        if (
            r < num_rounds - 1
        ):  # training not needed in last round because classifier will not be used anymore.
            logger.debug("train restricted estimator ...\n")
            t = time.time()
            classifier = restriction_estimator.train()
            logger.debug(
                "train restricted estimator...done. Took {} seconds.\n".format(
                    np.round(time.time() - t, 2)
                )
            )
            logger.debug("restrict prior ...\n")
            t = time.time()
            proposals.append(restriction_estimator.restrict_prior())
            logger.debug("restrict prior...done. Took {} seconds.\n")

    # collect all simulations
    all_theta, all_x, _ = restriction_estimator.get_simulations()

    # train inference
    density_estimator = inference.train()
    
    # build posterior
    posterior = inference.build_posterior()

    # save posterioer
    logger.debug("save posterior ...\n")
    t = time.time()
    with open(os.path.join(save_dir, "posterior.p"), "wb") as tmp:
        pickle.dump(posterior, tmp)
    logger.debug("save posterior...done. Took {} seconds.\n")


def _get_model_from_sbi_experiment(
    model_type,
    experiment_id,
    theta,
    I0=None,
    variables=["I", "v"],
    fixed_model_parameters={},
    fixed_simulation_parameters={},
    seed=None,
    logger=logging.getLogger(__name__),
):
    """Utility function to quickly get a trained model for the sbi experiment.

    Parameters
    ----------
    experiment_id : str
        For each change in parameters, a unique parameter id is (and should be)
        defined in 'config.experiment.<experiment_name>'.
    theta : torch.Tensor, np.Array
        Tensor or Array containing the scaling factors of the parameters and
        variables that have been changed according to
        'experiment.sbi.<experiment_id>'.
    variables : list of str, optional
        Defines which variables to record. By default ['I', 'v'].
    logger : logging.Logger, optional
        Logger instance to be called for logging.
        By default logging.getLogger(__name__).

    Returns
    -------
    HH_model
        Trained HH_model.
    """
    import torch

    experiment_params = config.experiment["sbi"][experiment_id]

    model_name = experiment_params["model_name"]
    simulation_name = experiment_params["simulation_name"]
    fixed_simulation_params = deepcopy(experiment_params["fixed_simulation_params"])
    fixed_model_params = deepcopy(experiment_params["fixed_model_params"])

    for k, v in fixed_model_parameters.items():
        fixed_model_params[k] = v
    for k, v in fixed_simulation_parameters.items():
        fixed_simulation_params[k] = v

    if "log10" in experiment_params.keys():
        if experiment_params["log10"] & (theta is not None):
            theta = 10**theta

    return _get_model_from_augmented_parameters(
        model_type=model_type,
        model_name=model_name,
        simulation_name=simulation_name,
        theta=theta,
        variables=variables,
        augmented_model_params=experiment_params["augmented_model_params"],
        augmented_eqs=experiment_params["augmented_eqs"],
        augmented_simulation_params=experiment_params.get(
            "augmented_simulation_params", {}
        ),
        fixed_simulation_params=fixed_simulation_params,
        fixed_model_params=fixed_model_params,
        I0=I0,
        seed=seed,
        logger=logger,
    )


def _get_model_from_single_variable_experiment(
    experiment_id,
    theta,
    variables=["I", "v"],
    seed=None,
    logger=logging.getLogger(__name__),
):
    exp_params = config.experiment["single_variable"][experiment_id]
    model_type = exp_params["model_type"]
    model_name = exp_params["model_name"]
    simulation_name = exp_params["simulation_name"]
    fixed_simulation_params = deepcopy(exp_params.get("fixed_simulation_params", {}))
    fixed_model_params = deepcopy(exp_params.get("fixed_model_params", {}))

    if exp_params["parameter_type"] == "model_parameter":
        augmented_model_params = [exp_params["parameter_name"]]
        augmented_eqs = []
    elif exp_params["parameter_type"] == "eqs":
        augmented_model_params = []
        augmented_eqs = [exp_params["parameter_name"]]
    else:
        raise NotImplementedError

    if "log10" in exp_params.keys():
        if exp_params["log10"] & (theta is not None):
            logger.info("apply log10: {} -> {}".format(theta, 10**theta))
            theta = 10**theta
    logger.info("Augment {} by factor {}".format(exp_params["parameter_name"], theta))

    return _get_model_from_augmented_parameters(
        model_type=model_type,
        model_name=model_name,
        simulation_name=simulation_name,
        theta=theta,
        variables=variables,
        augmented_model_params=augmented_model_params,
        augmented_eqs=augmented_eqs,
        fixed_simulation_params=fixed_simulation_params,
        fixed_model_params=fixed_model_params,
        seed=seed,
        logger=logger,
    )


def _scale_parameters(
    model_name,
    model_type,
    theta=None,
    augmented_model_params=[],
    augmented_eqs=[],
    augmented_simulation_params={},
):
    augmented_model_params_dict = {}
    augmented_eqs_dict = {}
    augmented_simulation_params_dict = {}

    i = 0
    for v in augmented_model_params:
        p = scale_variable(
            v,
            factor=torch.Tensor([1.0]) if theta is None else theta[i],
            model_type=model_type,
            model_name=model_name,
            parameter_type="parameters",
        )
        augmented_model_params_dict[v] = p
        i += 1
    for v in augmented_eqs:
        p = scale_variable(
            v,
            factor=torch.Tensor([1.0]) if theta is None else theta[i],
            model_type=model_type,
            model_name=model_name,
            parameter_type="eqs",
        )
        augmented_eqs_dict[v] = p
        i += 1

    for k, v in augmented_simulation_params.items():
        p = "{}*{}".format(theta[i], v)
        augmented_simulation_params_dict[k] = p
        i += 1

    return (
        augmented_model_params_dict,
        augmented_eqs_dict,
        augmented_simulation_params_dict,
    )


def _get_model_from_augmented_parameters(
    model_type,
    model_name,
    simulation_name,
    theta,
    variables=["I", "v"],
    augmented_model_params=[],
    augmented_eqs=[],
    augmented_simulation_params={},
    fixed_simulation_params={},
    fixed_model_params={},
    I0=None,
    seed=None,
    logger=logging.getLogger(__name__),
):

    (
        augmented_model_params_dict,
        augmented_eqs_dict,
        augmented_simulation_params_dict,
    ) = _scale_parameters(
        model_name=model_name,
        model_type=model_type,
        theta=theta,
        augmented_model_params=augmented_model_params,
        augmented_eqs=augmented_eqs,
        augmented_simulation_params=augmented_simulation_params,
    )

    for param, value in fixed_model_params.items():
        augmented_model_params_dict[param] = value

    if I0 is not None:
        augmented_model_params_dict["I0"] = I0

    model_params, eqs, simulation_params = augment_parameters(
        model_type=model_type,
        model_name=model_name,
        simulation_name=simulation_name,
        augmented_model_params=augmented_model_params_dict,
        augmented_eqs=augmented_eqs_dict,
        augmented_simulation_params=augmented_simulation_params_dict,
        fixed_simulation_params=fixed_simulation_params,
    )


    save_spikes = True
    m = get_model(
        model_type=model_type,
        model_parameters=model_params,
        eqs=eqs,
        simulation_parameters=simulation_params,
        variables=variables,
        save_spikes=save_spikes,
        seed=seed,
        logger=logger,
    )
    return m


def get_model_from_experiment(
    experiment_name,
    experiment_id,
    results=None,
    variables=["I", "v"],
    fixed_model_parameters={},
    fixed_simulation_parameters={},
    seed=None,
    logger=logging.getLogger(__name__),
):
    """Utility function to quickly get a trained model based on an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment. Must be specified in 'config.experiment' and
        must be implemented in this function.
    experiment_id : str
        For each change in parameters, a unique parameter id is (and should be)
        defined in 'config.experiment.<experiment_name>'.
    results : str, int, None, optional
        A model can be trained according to specified results of an experiment.
        E.g. in 'config.results.<experiment_name>.<experiment_id>, for sbi,
        there are thetas defined for specified observations (phases). In the
        case of 'sbi', the theta defines the model parameters and can be passed
        to the model through the results parameter. By default None.
    variables : list of str, optional
        Defines which variables to record. By default ['I', 'v'].
    logger : logging.Logger, optional
        Logger instance to be called for logging.
        By default logging.getLogger(__name__).

    Returns
    -------
    HH_model
        Trained HH_model.

    Raises
    ------
    NotImplementedError
        If experiment is not implemented.
    """
    if experiment_name == "sbi":
        model_type = config.experiment["sbi"][experiment_id]["model_type"]
        return _get_model_from_sbi_experiment(
            model_type=model_type,
            experiment_id=experiment_id,
            theta=results,
            variables=variables,
            fixed_model_parameters=fixed_model_parameters,
            fixed_simulation_parameters=fixed_simulation_parameters,
            seed=seed,
            logger=logger,
        )
    elif experiment_name == "single_variable":
        return _get_model_from_single_variable_experiment(
            experiment_id=experiment_id,
            theta=results,
            variables=variables,
            seed=seed,
            logger=logger,
        )
    else:
        raise NotImplementedError


def get_model(
    model_type,
    simulation_parameters,
    model_parameters,
    eqs=None,
    init_v=False,
    variables=[],
    save_spikes=True,
    seed=None,
    logger=logging.getLogger(__name__),
):
    import brian2 as b2

    model_parameters = deepcopy(model_parameters)
    models = {
        "HH_model": HH_model,
    }

    logger.debug("model_parameters:\n\t{}".format(model_parameters))
    logger.debug("simulation_parameters:\n\t{}".format(simulation_parameters))
    logger.debug("eqs:\n\t{}".format(eqs))

    logger.debug("get_model - initialise model...")
    m = models[model_type](
        model_parameters=model_parameters,
        simulation_parameters=simulation_parameters,
        eqs=eqs,
        init_v=init_v,
        variables=variables,
        save_spikes=save_spikes,
        seed=seed,
        logger=logger,
    )
    logger.debug("get_model - initialise model... done.")
    if m.valid:
        logger.debug("get_model - model is valid.")
        logger.debug("get_model - run model...")
        m.run()
        logger.debug("get_model - run model... done")
        return m
    else:
        logger.debug("get_model - model is invalid.")
        return np.float64("nan")


def run_posterior_check(
    sbi_experiment_id,
    x_o,
    n_simulations,
    num_workers,
    save_dir,
    sbi_save_dirs=None,
    posterior_save_path=None,
    seed=None,
    logger=logging.getLogger(__name__),
):
    # sbi should not be mandatory for all functions
    import torch
    import sbi
    from sbi import utils
    from sbi import inference

    experiment_params = deepcopy(config.experiment["sbi"][sbi_experiment_id])
    log10 = experiment_params.pop("log10", False)

    if sbi_save_dirs is not None:
        if sbi_save_dirs == "all":
            sbi_save_dirs = collect_save_dirs(sbi_experiment_id, include_config=True)
        else:
            assert type(sbi_save_dirs) == list
        assert type(sbi_save_dirs) == list

        n_theta = len(
            experiment_params["augmented_model_params"]
            + experiment_params["augmented_eqs"]
        )

        low = experiment_params.pop("low", None)
        high = experiment_params.pop("high", None)

        prior_distribution = experiment_params.pop("prior_distribution", "uniform")
        prior_distribution_kw = experiment_params.pop("prior_distribution_kw", {})

        prior = get_prior(
            prior_distribution=prior_distribution,
            prior_distribution_kw=prior_distribution_kw,
            num_dim=n_theta,
            low=low,
            high=high,
        )

        inference = sbi.inference.SNPE(prior=prior)
        restriction_estimator = sbi.utils.RestrictionEstimator(prior=prior)
        
        thetas, phases = collect_rounds(sbi_save_dirs)
        logger.debug(sbi_save_dirs)
        logger.debug("len(thetas): {}".format(len(thetas)))
        logger.debug("thetas[0].shape: {}".format(thetas[0].shape))

        proposal = prior
        for r in range(len(thetas)):
            estimator = inference.append_simulations(
                thetas[r], phases[r], proposal=proposal
            )
            restriction_estimator.append_simulations(thetas[r], phases[r])
            restriction_estimator.train()
            proposal = restriction_estimator.restrict_prior()

        # all_theta, all_x, _ = restriction_estimator.get_simulations()
        # inference.append_simulations(all_theta, all_x)

        inference.train()

        posterior = inference.build_posterior()

    else:
        assert posterior_save_path is not None
        assert os.path.isfile(posterior_save_path)
        with open(posterior_save_path, "rb") as tmp:
            posterior = pickle.load(tmp)

    x_o = x_o * torch.ones(1)

    logger.debug("posterior.sample((1,)) = {}".format(posterior.sample((1,), x=x_o)))

    posterior.set_default_x(x_o)

    logger.debug("posterior(x_o).sample((1,)) = {}".format(posterior.sample((1,))))
    # ToDo
    simulator = sbi_simulator(
        model_type=experiment_params["model_type"],
        simulation_name=experiment_params["simulation_name"],
        model_name=experiment_params["model_name"],
        augmented_model_params=experiment_params["augmented_model_params"],
        augmented_eqs=experiment_params["augmented_eqs"],
        fixed_simulation_params=experiment_params["fixed_simulation_params"],
        save_variables=experiment_params["save_variables"],
        save_spikes=experiment_params["save_spikes"],
        fixed_model_parameters=experiment_params["fixed_model_params"],
        save_dir=save_dir,
        log10=log10,
        logger=logger,
        seed=seed,
    )

    if seed is not None:
        torch.manual_seed(seed)

    theta, x = sbi.inference.simulate_for_sbi(
        simulator, posterior, num_simulations=n_simulations, num_workers=num_workers
    )

    save_sbi_simulation(theta, x, save_str="simulation_results.p", save_dir=save_dir)



def run_frequency_response(
    model_type,
    simulation_name,
    model_name,
    f_0,
    f_1,
    f_N,
    f_dist,
    n_right=5,
    n_init=10,
    adapt_init=False,
    adapt_N=False,
    n_sim=5,
    n_jobs=-1,
    fixed_model_params={},
    fixed_eqs={},
    fixed_simulation_params={},
    save_variables={},
    sbi_id=None,
    theta=None,
    save_spikes=False,
    save_dependent_variables={},
    save_dir=".",
    seed=None,
    logger=logging.getLogger(__name__),
):

    if f_dist == "linear":
        freqs = np.linspace(f_0, f_1, f_N)
    elif f_dist == "log10":
        freqs = 10 ** np.linspace(np.log10(f_0), np.log10(f_1), f_N)
    else:
        raise NotImplementedError

    # sort indexes so not all long runs start at once (memory overload)
    idxs = np.zeros(f_N, dtype=int)
    for i in range(f_N // (n_right + 1)):
        idxs[(n_right + 1) * i] = i
        for ir in range(n_right):
            idxs[(n_right + 1) * i + ir + 1] = f_N - i * n_right - ir - 1

    i_ = -1
    for i in range(f_N):
        if i not in idxs:
            idxs[i_] = i
            i_ -= 1

    if sbi_id is not None:
        import torch

        assert sbi_id in config.experiment["sbi"].keys()

        experiment_params = config.experiment["sbi"][sbi_id]

        augmented_model_params, augmented_eqs, augmented_simulation_params = (
            _scale_parameters(
                model_name=model_name,
                model_type=model_type,
                theta=theta,
                augmented_model_params=experiment_params.get(
                    "augmented_model_params", []
                ),
                augmented_eqs=experiment_params.get("augmented_eqs", []),
                augmented_simulation_params=experiment_params.get(
                    "augmented_simulation_params", {}
                ),
            )
        )
    else:
        augmented_model_params, augmented_eqs, augmented_simulation_params = {}, {}, {}

    if adapt_N and ("N" in fixed_simulation_params.keys()):
        augmented_simulation_params["N"] = fixed_simulation_params.pop("N")

    def eval_freq(freq):
        tmp_simulation_params = deepcopy(augmented_simulation_params)
        tmp_model_params = deepcopy(augmented_model_params)

        tmp_model_params["f"] = "{}*Hz".format(freq)
        t = n_sim / freq
        tmp_simulation_params["t"] = "{}*second".format(t)
        if adapt_init and (freq > 30):
            t_init = 10 * n_init / freq
        if adapt_init and (freq > 20):
            t_init = 6 * n_init / freq
        elif adapt_init and (freq > 10):
            t_init = 4 * n_init / freq
        elif adapt_init and (freq > 5):
            t_init = 2 * n_init / freq
        else:
            t_init = n_init / freq
        if adapt_N:
            tmp_simulation_params["N"] = int(
                tmp_simulation_params["N"] * n_sim / 10 / t
            )

        tmp_simulation_params["t_init"] = "{}*second".format(t_init)

        simulation_id = str(uuid.uuid4())
        # save_pre_str = simulation_id  # directory in $SCRATCH to store temporary files
        file_pre_str = simulation_id  # run_id put in front of every saved file

        run_augmented_parameters(
            model_type=model_type,
            simulation_name=simulation_name,
            model_name=model_name,
            augmented_model_params=tmp_model_params,
            augmented_eqs=augmented_eqs,
            augmented_simulation_params=tmp_simulation_params,
            fixed_model_params=fixed_model_params,
            fixed_eqs=fixed_eqs,
            fixed_simulation_params=fixed_simulation_params,
            save_variables=save_variables,
            save_spikes=save_spikes,
            save_dependent_variables=save_dependent_variables,
            save_dir=save_dir,
            file_pre_str=file_pre_str,
            seed=seed,
            logger=logger,
        )

    joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(eval_freq)(f) for f in freqs[idxs])
    
    run_ids = collect_run_ids(save_dir)
    N = config.simulation[simulation_name].get('N')
    N = fixed_simulation_params.get('N', N)

    fs = []
    data = {}
    for run_id in run_ids:
        variables = get_variables(save_dir, run_id)
        parameters = get_augmented_parameters(save_dir, run_id)
        f = float(parameters['model_parameters']['f'].split('*')[0])
        data[f] = {}
        
        s1 = psth(variables['spikes']['t'], N)
        s2 =  np.arange(len(variables['I']['mean'])) * 1e-4, variables['I']['mean']
        
        amp_1 = np.ptp(s2[1]) / 2

        phase_shift_kw = {'f_0': f, 'amp_1': amp_1}
        shift = phase_shift(s1, s2, f_0=f, fixed_f=True, amp_1=amp_1)
        p1, _ = phase_shift(s1, s2, f_0=f, fixed_f=True, amp_1=amp_1, return_params=True)

        # negative for advanced firing
        data[f]['shift'] = shift
        data[f]['amp'] = np.ptp(s1[1]) / 2
        data[f]['amp_sin'] = p1[1]
        
    fs = list(data.keys())
    fs.sort()

    shifts = np.zeros_like(fs)
    amps = np.zeros_like(fs)

    for i_f, f in enumerate(fs):
        shifts[i_f] = data[f]['shift']
        amps[i_f] = data[f]['amps']

    results = {
        'frequencies': fs,
        'shifts': shifts,
        'amplitudes': amps,
    }
    with open(os.path.join(save_dir, 'results.p'), 'wb') as tmp:
        pickle.dump(results, tmp)


def get_sinusoidal_signal(
    frequencies, amplitudes, phases=None, t=None, T=2, dt=1e-4, zscore=True
):
    if t is None:
        t = np.arange(0, T, dt)

    if phases is None:
        phases = [0] * len(frequencies)

    signal = np.zeros_like(t)
    for f, a, p in zip(frequencies, amplitudes, phases):
        signal += a * np.sin(2 * np.pi * f * t + p)

    if zscore:
        signal -= signal.mean()
        signal /= signal.std()

    return signal


def run_sinusoidals(
    model_type,
    model_name,
    simulation_name,
    frequencies,
    amplitudes,
    num_simulations,
    save_variables,
    phases=None,
    fixed_model_params={},
    fixed_eqs={},
    fixed_simulation_params={},
    save_dependent_variables={},
    save_spikes=True,
    seed=None,
    save_dir=".",
    zscore=True,
    num_workers=32,
):
    from brian2 import ms, second, TimedArray

    assert len(frequencies) == len(amplitudes)
    fixed_eqs["I"] = "I0 + I1*signal(t) + I_noise : ampere"

    t_init = fixed_simulation_params.get(
        "t_init", config.simulation[simulation_name]["t_init"]
    )
    t = fixed_simulation_params.get("t", config.simulation[simulation_name]["t"])

    T = (eval(t_init) + eval(t)) / second
    dt = 1e-4  # seconds

    signal = get_sinusoidal_signal(
        frequencies=frequencies,
        amplitudes=amplitudes,
        phases=phases,
        T=T,
        dt=dt,
        zscore=zscore,
    )

    fixed_model_params["signal"] = TimedArray(signal, dt=dt * second)

    def single_run_():
        simulation_id = str(uuid.uuid4())
        run_augmented_parameters(
            model_type,
            simulation_name,
            model_name,
            fixed_model_params=fixed_model_params,
            fixed_eqs=fixed_eqs,
            fixed_simulation_params=fixed_simulation_params,
            save_variables=save_variables,
            save_spikes=True,
            save_dependent_variables=save_dependent_variables,
            save_dir=save_dir,
            file_pre_str=simulation_id,
        )

    joblib.Parallel(n_jobs=num_workers)(
        joblib.delayed(single_run_)() for _ in range(num_simulations)
    )
    
    
    # get summary
    N_run = config.simulation[simulation_name].get('N')
    N_run = fixed_simulation_params.get('N', N_run)
    N = num_simulations * N_run
    
    run_ids = collect_run_ids(save_dir)
    
    I = []
    t_s = []
    for run_id in run_ids:
        file_path = os.path.join(save_dir, '_'.join([run_id, 'variables.p']))
        with open(file_path, 'rb') as tmp:
            v = pickle.load(tmp)
        I.append(v['I']['mean'])
        t_s.append(v['spikes']['t'])
    
    
    bins = 10000
    t_r, r = psth(np.concatenate(t_s), N=N, hist_kw={'bins': bins})
    
    results = {
        'I': np.array(I).mean(0),
        't_r': t_r,
        'r': r,
    }
    save_path = os.path.join(save_dir, 'results.p')
    with open(save_path, 'wb') as tmp:
        pickle.dump(results, tmp)



def get_sinusoidal_parameters(sin_id, experiment_name="sinusoidals"):
    experiment_params = config.experiment[experiment_name][sin_id]
    if experiment_name == "sinusoidals":
        frequencies = experiment_params["frequencies"]
        amplitudes = experiment_params["amplitudes"]
        phases = experiment_params.get("phases", np.zeros_like(frequencies))
    else:
        raise NotImplementedError
    return frequencies, amplitudes, phases
