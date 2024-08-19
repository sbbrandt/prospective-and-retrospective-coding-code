from brian2 import *
import logging
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import scipy

import re
import uuid

import config
from viz import (
    plot_phase_shift,
    plot_variable,
    plot_spikes,
    psth,
    plot_gating_variables,
)
from utils import zscore, resolve_parameter_dependencies, eval_parameter, get_label

import time


class BaseModel:
    def __init__(
        self,
        model_parameters,
        simulation_parameters,
        eqs=None,
        variables=[],
        save_spikes=False,
        seed=None,
        logger=logging.getLogger(__name__),
        *args,
        **kwargs
    ):
        self.valid = True
        # check for required parameters
        self.check_parameters(model_parameters, simulation_parameters)
        # initialize model
        self.initialise_model(
            model_parameters,
            simulation_parameters,
            eqs=eqs,
            variables=variables,
            save_spikes=save_spikes,
            seed=seed,
            logger=logger,
            *args,
            **kwargs
        )

    def initialise_model(self):
        self.valid = True  # or False
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    @staticmethod
    def get_rheobase():
        raise NotImplementedError

    def check_parameters(self, model_parameters, simulation_parameters):
        required_simulation_parameters = [
            "t",  # simulation_time
            "N",  # number of neurons
            "t_init",  # initialization time
        ]
        for param in required_simulation_parameters:
            assert param in simulation_parameters.keys()

        required_model_parameters = [
            # 'I0',  # constant term of input current; required for rheobase
        ]
        for param in required_model_parameters:
            assert param in model_parameters.keys()


    def get_variable(self, variable, mean=False, diff_method="central", **kwargs):
        """Get a variable from the model

        The variable of a model are expected to be either event based
        (consisting of event time and event neuron id) or continuous (consisting
        of parsed time and value). While the model is expected to have the spike
        variable implemented in 'get_spikes', each continuous variable must be
        implemented in 'timed_variabe'. For continuous variables, the operations
        sign (i.e. 'variable="-v"') or time derivatives (i.e.
        'variable="v_dot"') are implemented. Operations can be recursive (i.e.
        'variable="-v_dot_dot"').

        Parameters
        ----------
        variable : str
            Name of the variable
        mean : bool, optional
            Whether to compute the mean over neurons, by default False
        diff_method : str, optional
            Method to compute time derivatives. Either 'forward' or 'central'.
            'forward' computes (f(t+dt)-f(t))/2 and adjusts the time, which
            leads to derivatives not being aligned to other variables. 'central'
            corrects for that by taking the mean of two consecutive derivatives
            leading to a correctly aligned time raster, compared to other
            variables, but is missing the first and last time point.
            By default 'central'

        Returns
        -------
        tuple of arrays
            The time and value of the variable.
        """
        assert diff_method in ["forward", "central"]
        # non continuous variables
        if variable == "spikes":
            t, i = self.get_spikes()
            return t, i
        # continuous v
        else:
            dot = False  # whether to compute time derivative
            sign = +1  # sign
            if variable.split("_")[-1] == "dot":
                variable = "_".join(variable.split("_")[:-1])
                dot = True
                t_val, val = self.get_variable(
                    variable, mean=False, diff_method=diff_method
                )
            elif variable[0] == "-":
                sign = -1
                variable = variable[1:]
                t_val, val = self.get_variable(variable, mean=False)
            if dot:
                if diff_method == "forward":
                    val = np.diff(val) / np.diff(t_val)
                    t_val = t_val[:-1] + np.diff(t_val)[0] / 2
                elif diff_method == "central":
                    diff = np.diff(val) / np.diff(t_val)
                    val = (diff[:, 1:] + diff[:, :-1]) / 2
                    t_val = t_val[1:-1]
            if mean:
                val = val.mean(0)
            val *= sign
            return (t_val, val.copy())

    def get_spikes(self):
        raise NotImplementedError

    def plot_variable(self, variable, get_variable_kw={}, plot_variable_kw={}):
        t, y = self.get_variable(variable, **get_variable_kw)
        plot_variable(t, y, **plot_variable_kw)


class HH_model(BaseModel):
    """Simulator for Hodgkin-Huxley model

    Parameters
    ----------
    eqs : str
        Equations that describe the dynamics of the model. See 'brian2' doc
        for more info.
    model_parameters : dict
        Parameters for the variables that are described by 'eqs'. The value
        of a parameter can be either a 'str' (including brian2 units) or the
        value.
    simulation_parameters : dict
        Parameters that define how the simulation is run (e.g. running time
        't'). Values can be either a 'str' (including brian2 units) or the
        value.
    variables : list, optional
        List of variables that should be recorded during the simulation,
        by default []
    spikes : bool, optional
        Whether to record spikes, by default False
    device_dir : str, optional
        CAREFUL: MIGHT NOT WORK! Used for standalone modus of brian2. Path
        to the device dir, by default None

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """

    def initialise_model(
        self,
        model_parameters,
        simulation_parameters,
        eqs,
        variables=[],
        continuous_timed_array=True,
        save_spikes=False,
        init_v=False,
        seed=None,
        logger=None,
    ):
        """Initialises the model

        Runs the model for time 't_init' without recording variables.

        Parameters
        ----------
        t_init : any, optional
            Either str (including brian2 unit), value (brian2 unit), False
            (no initialisation time). Defaults to
            t_init=1/eval(self.parameters['f']) if None, by default None
        """
        import brian2 as b2

        b2.prefs.codegen.target = "numpy"

        if logger is None:
            logger = logging.getLogger(__name__)

        # resolve dependencies within simulation parameters
        self.simulation_parameters = deepcopy(simulation_parameters)
        resolve_parameter_dependencies(self.simulation_parameters)

        model_parameters = deepcopy(model_parameters)
        self.eqs = eqs
        self.seed = seed

        self.rheobase_kw = model_parameters.pop("rheobase_kw", {})
        self.rate_based_kw = model_parameters.pop("rate_based_kw", {})

        if model_parameters["I0"] == "rheobase":
            # compute rheobase
            I_rh, V_rest = self.get_rheobase(
                model_parameters=model_parameters,
                simulation_parameters=self.simulation_parameters,
                eqs=eqs,
                logger=logger,
                **self.rheobase_kw
            )
            if np.isnan(I_rh):
                logger.info("Invalid rheobase.")
                self.valid = False
            else:
                I_rh_offset = model_parameters.pop("I_rh_offset", None)
                if I_rh_offset is not None:
                    I_rh += I_rh_offset

                logger.info(
                    "Initialize I0 = {} {}".format(I_rh, model_parameters["I_unit"])
                )
                model_parameters["I0"] = "{}*{}".format(
                    I_rh, model_parameters["I_unit"]
                )
        elif model_parameters["I0"] == "rate_based":
            # estimate average output rate by inter-spike interval for constant input
            I_rb = self.get_I0_rate_based(
                model_parameters=model_parameters,
                simulation_parameters=self.simulation_parameters,
                eqs=eqs,
                logger=logger,
                **self.rate_based_kw
            )
            if np.isnan(I_rb):
                logger.info("Invalid rate based I0.")
                self.valid = False
            else:
                logger.info(
                    "Initialize I0 = {} {}".format(I_rb, model_parameters["I_unit"])
                )
                model_parameters["I0"] = "{}*{}".format(
                    I_rb, model_parameters["I_unit"]
                )

        if self.valid:
            resolve_parameter_dependencies(model_parameters)
            logger.debug("Valid model. Initialize parameters...")
            for key, val in model_parameters.items():
                if type(val) == str:
                    try:
                        model_parameters[key] = eval_parameter(val)
                    except:
                        raise ValueError("{}".format(val))
            logger.debug("Initialize parameters... done.")
            self.model_parameters = model_parameters
            self.N = self.simulation_parameters["N"]
            self.method = self.simulation_parameters["method"]
            self.refractory = self.simulation_parameters["refractory"]
            self.t = eval_parameter(
                self.simulation_parameters["t"], local_vars=self.model_parameters
            )

            if self.seed is not None:
                # manually sample random variables.
                # brian2 draws random numbers from a stream, which leads to
                # simulations not being reproducible.
                # setting the seed can lead to large usage of RAM!
                assert type(self.seed) is int
                self.rng = np.random.default_rng(seed=self.seed)
                self.dt = defaultclock.dt
                nt = int(np.ceil(eval(self.simulation_parameters["t_init"]) / self.dt))
                # nt = int(np.ceil((eval(self.simulation_parameters['t_init']) + self.t) / self.dt))
                self.n_rnd = self.eqs.count("randn")
                for i in range(self.n_rnd):
                    # replace 'randn' of brian2 with numbered arrays
                    r_str = "r{}".format(i)
                    r_var = r_str + "(t, i)"
                    self.eqs = self.eqs.replace("randn()", r_var, 1)
                    self.model_parameters[r_str] = TimedArray(
                        self.rng.normal(size=(nt, self.N)), dt=self.dt
                    )

            # brian2 reset
            self.reset = self.simulation_parameters.get("reset")

            self.spike_count_method = self.simulation_parameters["spike_count_method"]
            self.v_min_peak = eval_parameter(self.simulation_parameters["v_min_peak"])

            if self.spike_count_method == "threshold":
                if self.simulation_parameters["threshold"] is not None:
                    assert (">" in self.simulation_parameters["threshold"]) or (
                        "<" in self.simulation_parameters["threshold"]
                    )
                    self.threshold = self.simulation_parameters["threshold"]
                elif "VT" in self.model_parameters:
                    self.threshold = "v > {} * mV".format(
                        self.model_parameters["VT"] / b2.mV
                    )
                else:
                    raise ValueError(
                        "'spike_count_method' is 'threshold', but threshold could not be defined: simulation_parameters['threshold'] is None and 'VT' is not specified in model_parameters"
                    )
            else:
                assert self.spike_count_method in ["find_peaks"]
                self.threshold = self.simulation_parameters.get("threshold", None)

            logger.debug("Build model.")

            # initialise model
            logger.debug("Initialize model...")
            # run model depending on t_init
            t_init = self.simulation_parameters["t_init"]
            if (t_init is False) or (t_init == "False") or (t_init is None):
                self.t_init = 0 * b2.second
                self.initial_states = {}
            else:
                logger.debug("Run for t_init = {}".format(t_init))
                t_init = eval_parameter(t_init, local_vars=self.model_parameters)
                self.t_init = t_init
                G = b2.NeuronGroup(
                    self.N,
                    self.eqs,
                    threshold=self.threshold,
                    refractory=self.refractory,
                    method=self.method,
                    reset=self.reset,
                )
                net = b2.Network()
                net.add([G])

                if init_v:
                    G.v = self.model_parameters["E_l"]

                # seperate timed arrays into init and simulation parts
                tmp_model_parameters = deepcopy(self.model_parameters)
                if continuous_timed_array:
                    t_init_second = t_init / b2.second
                    for k, v in tmp_model_parameters.items():
                        if type(v) == b2.input.timedarray.TimedArray:
                            logger.info(
                                "Adjust timedArray in parameter '{}' to compensate t_init={} seconds".format(
                                    k, t_init_second
                                )
                            )
                            # tmp_model_parameters[k].values = v.values[:int(t_init / v.dt)]
                            # self.model_parameters[k].values = v.values[int(
                            #     t_init / v.dt):]
                            # values = v.values.copy()
                            dt = v.dt
                            N_init = int(t_init / dt)
                            tmp_model_parameters[k] = b2.TimedArray(
                                v.values[:N_init], dt=dt * b2.second
                            )
                            self.model_parameters[k] = b2.TimedArray(
                                v.values[N_init:], dt=dt * b2.second
                            )
                            # self.eqs.replace(
                            # '{}(t)'.format(k), '{}(t - {})'.format(k, self.simulation_parameters['t_init']))

                net.run(t_init, namespace=tmp_model_parameters)
                # self.t_offset = t_init
                self.initial_states = deepcopy(G.get_states())
                for val in ["N", "i", "t", "dt", "t_in_timesteps"]:
                    self.initial_states.pop(val)
                # for threshold based spike counting, it is important to reset
                # the time of the last spike. Otherwise the system will be in
                # refractory eventually.
                if "lastspike" in self.initial_states:
                    self.initial_states["lastspike"] -= t_init
                del G
                del net
            # create new network such that time starts from zero.
            # thus, predefined random numbers need smaller size as time
            # cannot be reset manually.
            self.G = b2.NeuronGroup(
                self.N,
                self.eqs,
                threshold=self.threshold,
                refractory=self.refractory,
                method=self.method,
                reset=self.reset,
            )
            self.net = b2.Network()
            self.net.add([self.G])
            if init_v & ("v" not in self.initial_states.keys()):
                self.initial_states["v"] = self.model_parameters["E_l"]
            self.G.set_states(self.initial_states)
            self.t_offset = 0 * b2.second
            logger.debug("Initialize model... done")

            # initialise recordings
            self.M = None
            self.spikes = None
            self.variables = []

            logger.debug("Add monitors.")
            self.add_monitor(variables, save_spikes)
            logger.debug("initialise_model done.")

    def remove_monitor(self):
        self.net.remove(self.M)
        if self.spike_count_method == "threshold":
            self.net.remove(self.spikes)
        self.M = None
        self.spikes = None

    def add_monitor(self, variables=[], spikes=True, spike_variable="v"):
        """Adds recording monitor

        Parameters
        ----------
        variables : list, optional
            List of variables to record, by default []
        spikes : bool, optional
            Whether to record spikes, by default True
        spike_variable : str, optional
            Which variable to record the spikes from, by default 'v'
        """
        import brian2 as b2

        if len(variables) > 0:
            if self.M is not None:
                log.warning("StateMonitor already exists.")
            else:
                if (
                    spikes
                    and (self.spike_count_method != "threshold")
                    and ("v" not in variables)
                ):
                    self.variables = variables + [spike_variable]
                else:
                    self.variables = variables
                self.M = b2.StateMonitor(self.G, variables, record=True)
                self.net.add(self.M)
        if spikes:
            if self.spikes is not None:
                log.warning("SpikeMonitor already exists.")
            else:
                # brian2 only supports threshold
                # other methods must be implemented by hand in get_variable
                if self.spike_count_method == "threshold":
                    self.spikes = b2.SpikeMonitor(self.G)
                    self.net.add(self.spikes)

    def run(self, t=None, namespace=None):
        """run network

        Args:
            t (float, optional): time. Defaults to None.
            namespace (dict, optional): model paramters. Defaults to None.
        """
        if t is None:
            t = self.t
        if namespace is None:
            namespace = self.model_parameters
        if self.seed is not None:
            nt = int(np.ceil(t / self.dt))
            for i in range(self.n_rnd):
                r_str = "r{}".format(i)
                namespace[r_str] = TimedArray(
                    self.rng.normal(size=(nt, self.N)), dt=self.dt
                )

        self.net.run(t, namespace=namespace)

    def reset_model(
        self, states={}, default=0, exclude=["N", "i", "t", "dt", "t_in_timesteps"]
    ):
        initial_states = self.G.get_states()
        for k in exclude:
            if k in initial_states.keys():
                initial_states.pop(k)
        for k, v in initial_states.items():
            if k in states.keys():
                initial_states[k] = states[k]
            else:
                if k == "v":
                    initial_states[k] = self.model_parameters["E_l"]
                elif k == "last_spike":
                    # set last spike time to some time far away
                    initial_states[k] = -100 * b2.ms
                else:
                    initial_states[k] = default * v
        self.G.set_states(initial_states)

    @staticmethod
    def get_rheobase(
        model_parameters,
        simulation_parameters,
        eqs,
        t_init="100*ms",
        t="500*ms",
        max_v_rest="-30*mV",
        I0=0,
        I1=0,
        p0=0.1,
        p1=0.01,
        init_with_I0=False,
        min_I0=-np.inf,
        max_I0=np.inf,
        max_fr=None,
        early_stop_negative_rheobase=False,
        logger=logging.getLogger(__name__),
    ):
        """get the rheobase for the corresponding model

        Args:
            model_parameters (dict): model parameters (see 'config.model_dynamics')
            simulation_parameters (dict): simulation parameters (see 'config.simulation')
            eqs (str): equations (see 'config.model_dynamics')
            t_init (str, optional): Time to initialize with 'I0' before increasing or decreasing 'I0' stepswise. Defaults to '100*ms'.
            t (str, optional): Time to stimulate each step input. Defaults to '500*ms'.
            max_v_rest (str, optional): Maximal voltage without a spike. If neuron exceeds 'max_v_rest' during stimulation without a spike, the parameters are considered invalid and 'np.nan' is returned. Defaults to '-30*mV'.
            I0 (int, optional): Input current to start search for rheobase. Defaults to 0.
            I1 (int, optional): Amplitude of sinusoidal. Defaults to 0.
            p0 (float, optional): Size of largest steps. Defaults to 0.1.
            p1 (float, optional): Size of smallest steps. Defaults to 0.01.
            init_with_I0 (bool, optional): Initialize with 'I0' before each step input. Defaults to False.
            min_I0 (float, optional): Lower bound of 'I0'. Defaults to -np.inf.
            max_I0 (float, optional): Upper bound of 'I0'. Defaults to np.inf.
            max_fr (float, optional): Upper bound of firing rate. Defaults to None.
            early_stop_negative_rheobase (bool, optional): Return 'np.nan' for negative rheobase. Defaults to False.
            logger (logging.logger, optional): logging object. Defaults to logging.getLogger(__name__).

        Returns:
            float: value of rheobase or 'np.nan' if model is invalid
        """
        import brian2 as b2

        model_params = deepcopy(model_parameters)
        simulation_params = deepcopy(simulation_parameters)

        t_init = eval_parameter(t_init)
        t = eval_parameter(t)
        max_v_rest = eval_parameter(max_v_rest)

        simulation_params["N"] = 1
        simulation_params["refractory"] = "False"

        # for rheobase, input current should be constant impuls
        # set amplitude of signal to 0
        if "I0" in model_params.keys():
            model_params["I0"] = I0 * b2.nA
        if "I1" in model_params.keys():
            model_params["I1"] = I1 * b2.nA  # "0*nA"
        # set amplitude of noise to 0
        if "s" in model_params.keys():
            model_params["s"] = 0 * b2.nA  # "0*nA"

        m = HH_model(model_params, simulation_params, eqs=eqs)

        if m.t_init == 0 * b2.second:
            logger.info(
                "model was not initialised. Run for {} seconds to initialize.".format(
                    t_init / b2.second
                )
            )
            m.run(t_init)

        m.add_monitor(["v"])

        def spiketest(I0, return_v_rest, max_v):
            m.reset_model(states=m.initial_states)
            m.model_parameters["I0"] = I0 * b2.nA
            if init_with_I0:
                m.run(t_init)
            t0 = m.M.t[-1] / b2.second if len(m.M.t) > 0 else 0.0
            m.run(t)
            t_s = m.get_variable("spikes")[0]
            t_s = t_s[t_s > t0]

            t_v, v = m.get_variable("v", unit="mV", mean=False)
            v = v[:, t_v > t0]
            if return_v_rest:
                v_max = v[0].max()
                if (v_max >= max_v / b2.mV) and (len(t_s) == 0):
                    v_rest = max_v
                else:
                    v_rest = v[0, -100:].mean() * b2.mV
                return t_s, v_rest
            else:
                return t_s

        # test for negative rheobase current
        if early_stop_negative_rheobase:
            t_s = spiketest(I0=0.0, return_v_rest=False, max_v=max_v_rest)
            if len(t_s) > 0:
                logger.info("Invalid run: rheobase current is negative.")
                v_rest = 0
                I_rh = np.float64("nan")
                return I_rh, v_rest

        # find lowest current for which model does not spike (ideally I0 -> loop only once)
        spiked = True
        while spiked:
            logger.info("Check if model does spike for I0={}.".format(I0))
            t_s = spiketest(I0, return_v_rest=False, max_v=max_v_rest)
            if len(t_s) > 0:
                if I0 < min_I0:
                    v_rest = 0
                    I_rh = np.float64("nan")
                    return I_rh, v_rest
                spiked = True
                I0 -= p0
                logger.info(
                    "\tmodel did spike ({} spikes). Setting I0={}.".format(len(t_s), I0)
                )
            else:
                spiked = False
                logger.info("\tmodel did not spike. Continue ...")

        logger.info("... start fine grade search.")
        # to find the rheobase, the search goes from coarse to fine, starting
        # with steps of p0 (e.g. p0=0.1=10**-1) and moving exponent-wise to
        # p1 (e.g. p1=0.001=10**-3) with an intermediate step of 10**-2
        exp_range = np.int(np.log10(p0) - np.log10(p1)) + 1
        for exp in range(exp_range):
            spiked = False
            step = 10 ** (np.log10(p0) - exp)
            logger.info("I0={}".format(I0))
            logger.info("step={}".format(step))
            # some voltage smaller than max_v_rest
            v_rest = max_v_rest - 0.1 * abs(max_v_rest)
            while (not spiked) and (v_rest < max_v_rest) and (I0 < max_I0):
                I0_tmp = I0 + step
                ts, v_rest = spiketest(I0_tmp, return_v_rest=True, max_v=max_v_rest)
                logger.info(
                    "\tI0={}; spikes={}; v_rest={}".format(I0_tmp, len(ts), v_rest)
                )
                if v_rest >= max_v_rest:
                    logger.info("v_rest >= max_v_rest = {}".format(max_v_rest))
                    I_rh = np.float64("nan")
                    return I_rh, v_rest

                if len(ts) > 0:
                    spiked = True
                else:
                    I0 = I0_tmp

        logger.info("Lowest voltage w/o spike: {}".format(v_rest))

        if spiked:
            if max_fr is not None:
                if len(ts) > 1:
                    fr = 1 / np.diff(ts)[-1]
                else:
                    fr = 0
                if fr <= max_fr:
                    I_rh = round(I0, int(-np.log10(p1)))
                else:
                    I_rh = np.float64("nan")
            else:
                I_rh = round(I0, int(-np.log10(p1)))
        else:
            I_rh = np.float64("nan")
        logger.info("I_rh = {}".format(I_rh))
        return I_rh, v_rest

    @staticmethod
    def get_I0_rate_based(
        model_parameters,
        simulation_parameters,
        eqs,
        t_init="100*ms",
        t="200*ms",
        max_v_rest="-30*mV",
        I0=0,
        I1=0,
        p0=0.1,
        p1=0.001,
        init_with_I0=False,
        min_I0=-np.inf,
        max_I0=np.inf,
        min_fr=0,
        max_fr=np.inf,
        initialization="I0",
        initialization_value=0,
        logger=logging.getLogger(__name__),
        target_fr=10,
        t_reinit="100*ms",
        I0_step=0.005,
    ):
        assert initialization in ["I0", "reset"]

        import brian2 as b2

        model_params = deepcopy(model_parameters)
        simulation_params = deepcopy(simulation_parameters)

        t_init = eval_parameter(t_init)
        t_reinit = eval_parameter(t_reinit)
        t = eval_parameter(t)
        max_v_rest = eval_parameter(max_v_rest)

        simulation_params["N"] = 1
        simulation_params["refractory"] = "False"
        simulation_params["t_init"] = t_init

        valid = True

        if "I0" in model_params.keys():
            model_params["I0"] = I0 * b2.nA
        if "I1" in model_params.keys():
            model_params["I1"] = I1 * b2.nA  # "0*nA"
        # set amplitude of noise to 0
        if "s" in model_params.keys():
            model_params["s"] = 0 * b2.nA  # "0*nA"

        m = HH_model(model_params, simulation_params, eqs=eqs)
        m.add_monitor(["v"])

        def fr_test(I0):
            # m.model_parameters['I0'] = initialization_value * b2.nA
            # m.run(t_reinit)
            m.reset_model(states=m.initial_states)
            t0 = m.M.t[-1] / b2.second if len(m.M.t) > 0 else 0.0
            m.model_parameters["I0"] = I0 * b2.nA
            m.run(t)
            t_s = m.get_variable("spikes")[0]
            t_s = t_s[t_s > t0]
            if len(t_s) > 1:
                fr = 1 / np.diff(t_s)[-1]
            else:
                fr = 0
            return fr

        logger.info("Start I0 search...")
        logger.info("Get fr for initial I0={:.3f}...".format(I0))
        fr = fr_test(I0)
        logger.info("...done. fr(I0={:.3f})={:.3f}.".format(I0, fr))

        # allow to increase I0 to inf in first round
        I0_max_step = np.inf
        I0_old = I0
        fr_old = 0.0
        if fr > target_fr:
            t0 = time.time()
            logger.info(
                "\tfr={:.3f} is larger than target_fr={}.\nStart decreasing I0...".format(
                    fr, target_fr
                )
            )
            while (fr > target_fr) and (I0 > min_I0):

                I0 -= p0
                fr = fr_test(I0)
                logger.info("fr(I0={:.3f})={:.3f}.".format(I0, fr))
            fr_old = fr
            I0_old = I0
            logger.info("...done. Took {:.1f} seconds".format(time.time() - t0))

        exp_range = np.int64(np.log10(p0) - np.log10(p1)) + 1
        for exp in range(exp_range):
            I0 = I0_old
            fr = fr_old
            step = 10 ** (np.log10(p0) - exp)
            logger.info("I0={}".format(I0))
            logger.info("Start step={}...".format(step))
            t0 = time.time()
            while (fr < target_fr) and (I0 < max_I0) and (I0 <= I0_max_step):
                fr_old = fr
                I0_old = I0
                I0 += step
                fr = fr_test(I0)
                logger.info("fr(I0={:.3f})={:.3f}.".format(I0, fr))
            # only allow I0 to increase until last stepsize
            I0_max_step = I0
            logger.info("...done. Took {:.1f} seconds".format(time.time() - t0))

        if fr > max_fr:
            logger.info(
                "fr={} is larger than max_fr={}. Model is invalid.".format(fr, max_fr)
            )
            valid = False
        if fr < min_fr:
            logger.info(
                "fr={} is smaller than min_fr={}. Model is invalid.".format(fr, min_fr)
            )
            valid = False
        if I0 > max_I0:
            logger.info(
                "I0={} is larger than max_I0={}. Model is invalid.".format(I0, max_I0)
            )
            valid = False
        if I0 < min_I0:
            logger.info(
                "I0={} is smaller than min_I0={}. Model is invalid.".format(I0, min_I0)
            )
            valid = False

        if valid:
            if abs(fr_old - target_fr) < abs(fr - target_fr):
                final_fr = fr_old
                final_I0 = I0_old
            else:
                final_fr = fr
                final_I0 = I0
            logger.info(
                "fr({:.3f})={:.3f} is closest to target_fr={}.".format(
                    final_I0, final_fr, target_fr
                )
            )

            return final_I0
        else:
            return np.float64("nan")


    def get_variable(
        self,
        variable,
        unit=None,
        mean=True,
        diff_method="central",
        spike_variable="v",
        hist_kw={},
        spike_count_kw={},
    ):
        """Get variables from model

        Args:
            variable (str): the variable to get. E.g. model variables recorded in the monitor ('v', 'I', ...) but also many equations (e.g. '-v*g_na' or 'g_na_inf_dot').
            unit (str, optional): Unit the variable. If 'None' efaults to unit from Monitor. Defaults to None.
            mean (bool, optional): Return mean. Defaults to True.
            diff_method (str, optional): 'central' for differentiating correct time points for 't[1:-1]' or 'forward' for values according to '(t[i] + t[i+1])/2'. Defaults to "central".
            spike_variable (str, optional): Variable to compute the spike times on. Defaults to "v".
            hist_kw (dict, optional): kwargs for histogram in 'psth'. Defaults to {}.
            spike_count_kw (dict, optional): kwargs for 'spike_count'. Defaults to {}.

        Returns:
            (t, v): time and vale for variable
        """
        import brian2 as b2

        assert diff_method in ["forward", "central"]
        if variable == "psth":
            t, _ = self.get_variable("spikes")
            t_h, h = psth(t, self.N, hist_kw)
            return (t_h, h)
        elif variable == "spikes":
            if self.spike_count_method == "threshold":
                assert self.spikes is not None
                t = (self.spikes.t - self.t_offset) / b2.second
                i = self.spikes.i[:]
            elif self.spike_count_method == "find_peaks":
                t_v, v = self.get_variable(spike_variable, mean=False, unit="mV")
                height = spike_count_kw.pop("height", self.v_min_peak / b2.mV)
                t = []
                i = []
                for i_v in range(len(v)):
                    peaks, peaks_dict = scipy.signal.find_peaks(
                        v[i_v], height=height, **spike_count_kw
                    )
                    t.append(t_v[peaks])
                    i.append([i_v] * len(peaks))
                t = np.concatenate(t)
                i = np.concatenate(i)
            else:
                raise NotImplementedError(
                    "spike method {} is not implemented in {}.get_variable".format(
                        self.spike_count_method, self
                    )
                )
            return (t, i)
        else:
            dot = False
            sign = +1
            t_val = np.array((self.M.t - self.t_offset) / b2.second)
            if variable.split("_")[-1] == "dot":  # get temporal derivatives
                variable = "_".join(variable.split("_")[:-1])
                dot = True
                t_val, val = self.get_variable(
                    variable, mean=False, diff_method=diff_method
                )
            elif variable[0] == "-":  # get sign
                sign = -1
                variable = variable[1:]
                t_val, val = self.get_variable(variable, mean=False)
            elif variable in self.variables:  # get recorded variable
                val = self.M.variables[variable].get_value().T  # SI units
                if unit is not None:
                    if type(unit) is str:
                        unit = eval(unit)
                    val = val * self.M.variables[variable].unit / unit
            elif variable.split("_")[0] in ["m", "n", "h"]:  # get gating variables
                var = variable.split("_")[0]  # gating variable
                assert len(variable.split("_")) == 2  # e.g. m_inf or m_tau
                attr = variable.split("_")[1]  # i.e. inf or tau
                if attr == "inf":
                    _, alpha = self.get_variable("alpha_{}".format(var), mean=False)
                    _, beta = self.get_variable("beta_{}".format(var), mean=False)
                    val = 1 / (beta / alpha + 1)
                elif attr == "tau":
                    _, alpha = self.get_variable("alpha_{}".format(var), mean=False)
                    _, beta = self.get_variable("beta_{}".format(var), mean=False)
                    val = 1 / (alpha + beta)
                else:
                    raise NotImplementedError(
                        "{}_{} is not implemented.".format(var, attr)
                    )
            elif variable.split("_")[0] == "g":  # conductances
                assert len(variable.split("_")) > 1
                var = variable.split("_")[1]  # e.g. Na or K
                if variable.split("_")[-1] == "inf":  # steady state
                    inf = "_inf"
                    variable = "_".join(variable.split("_")[:-1])
                else:
                    inf = ""
                if var == "na":
                    _, m = self.get_variable("m" + inf, mean=False)
                    _, h = self.get_variable("h" + inf, mean=False)
                    val = m**3 * h * self.model_parameters["g_na_bar"] / b2.siemens
                elif var == "kd":
                    _, n = self.get_variable("n" + inf, mean=False)
                    val = n**4 * self.model_parameters["g_kd_bar"] / b2.siemens
                elif var == "tot":
                    _, g_na = self.get_variable("g_na" + inf, mean=False)
                    _, g_kd = self.get_variable("g_kd" + inf, mean=False)
                    g_l = self.model_parameters["g_l"] / b2.siemens
                    val = g_na + g_kd + g_l  # / b2.second

            elif variable.split("_")[0] == "I":  # currents
                assert len(variable.split("_")) > 1
                var = variable.split("_")[1]
                assert var in ["na", "kd"]
                if variable.split("_")[-1] == "inf":
                    inf = "_inf"
                    variable = "_".join(variable.split("_")[:-1])
                else:
                    inf = ""
                _, g = self.get_variable("_".join(["g", var]) + inf)
                E = self.model_parameters["_".join(["E", var])] / b2.volt
                _, v = self.get_variable("v", mean=False)
                val = g * (E - v)
            elif variable == "tau_eff":
                _, g_tot = self.get_variable("g_tot", mean=False)
                C_m = self.model_parameters["Cm"] / b2.fara
                val = C_m / (g_tot)  # / b2.second
            else:
                raise ValueError("variable {} not implemented.".format(variable))

            if dot:
                if diff_method == "forward":
                    val = np.diff(val) / np.diff(t_val)
                    t_val = t_val[:-1] + np.diff(t_val)[0] / 2
                elif diff_method == "central":
                    diff = np.diff(val) / np.diff(t_val)
                    val = (diff[:, 1:] + diff[:, :-1]) / 2
                    t_val = t_val[1:-1]
            if mean:
                val = val.mean(0)
            val *= sign
            return (t_val, val.copy())

    def plot_phase_shift(
        self,
        v1,
        v2,
        i="mean",
        fig=None,
        ax1=None,
        ax2=None,
        i_max=1,
        fontsize=16,
        subplot=(1, 1, 1),
        diff_method="central",
        hist_kw={},
        t_from=None,
        t_to=None,
        every=1,
        **kwargs
    ):
        """Plot phase shift between two variables

        Args:
            v1 (str): Variable 1
            v2 (str): Variable 2
            i (str or int, optional): Index of variable or 'mean'. Defaults to "mean".
            fig (plt.Figure, optional): Figure object. Defaults to None.
            ax1 (plt.Axes, optional): Axes object for variable 1. Defaults to None.
            ax2 (plt.Axes, optional): Axes object for Variable 2. Defaults to None.
            i_max (int, optional): Index of maxima for vertical line. Defaults to 1.
            fontsize (int, optional): Fontsize of text. Defaults to 16.
            subplot (tuple, optional): Description of subplot of Figure that contains ax1 and ax2. Defaults to (1, 1, 1).
            diff_method (str, optional): Differentiation method to use for variables that contain temporal derivatives. Defaults to "central".
            hist_kw (dict, optional): kwargs for histogram. Defaults to {}.
            t_from (_type_, optional): Lower bound of time axis. Defaults to None.
            t_to (_type_, optional): Upper bound for time axis. Defaults to None.
            every (int, optional): Plot 'every'th sample. Defaults to 1.
        """
        kwargs = deepcopy(kwargs)
        if fig is None:
            assert ax1 is None
            assert ax2 is None
            fig, (ax1, ax2) = plt.subplots(2, figsize=(7.5, 10), sharex=True)
        if v1 in config.variable_dict.keys():
            unit = config.variable_dict[v1]["unit"]
        else:
            unit = None
        unit = None
        s1, label1 = self.resolve_variable(
            v1,
            i=i,
            unit=unit,
            diff_method=diff_method,
            every=every,
            hist_kw=hist_kw,
        )

        if v2 in config.variable_dict.keys():
            unit = config.variable_dict[v2]["unit"]
        else:
            unit = None
        unit = None

        if (v1 == "psth") and (kwargs.get("type_1", "line") == "bar"):
            if "plot_kw_1" not in kwargs.keys():
                kwargs["plot_kw_1"] = {"color": "w", "edgecolor": "k", "alpha": 0.5}
        if (v2 == "psth") and (kwargs.get("type_2", "line") == "bar"):
            if "plot_kw_2" not in kwargs.keys():
                kwargs["plot_kw_2"] = {"color": "w", "edgecolor": "k", "alpha": 0.5}

        s2, label2 = self.resolve_variable(
            v2,
            i=i,
            unit=unit,
            diff_method=diff_method,
            every=every,
            hist_kw=hist_kw,
        )

        if t_from is not None:
            idx = np.argwhere(s1[0] > t_from).reshape(-1)
            s1 = s1[0][idx], s1[1][idx]
            idx = np.argwhere(s2[0] > t_from).reshape(-1)
            s2 = s2[0][idx], s2[1][idx]
        if t_to is not None:
            idx = np.argwhere(s1[0] < t_to).reshape(-1)
            s1 = s1[0][idx], s1[1][idx]
            idx = np.argwhere(s2[0] < t_to).reshape(-1)
            s2 = s2[0][idx], s2[1][idx]

        plot_phase_shift(
            s1,
            s2,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            label1=label1,
            label2=label2,
            i_max=i_max,
            subplot=subplot,
            fontsize=fontsize,
            **kwargs
        )

        for i_v, l in enumerate([label1, label2]):
            ax = [ax1, ax2][i_v]
            ax.tick_params(labelsize=fontsize)
            ax.set_ylabel(l, size=fontsize)
        ax2.set_xlabel(get_label("t"), size=fontsize)

        return fig, (ax1, ax2)

    def resolve_variable(
        self,
        v,
        i="mean",
        unit=None,
        label=None,
        diff_method="central",
        every=1,
        hist_kw={},
    ):
        """Resolve variable and mathematical expressions

        Args:
            v (str): variable string, containing variables and model parameters, e.g. '3*<m>**3*<h>*<g_na_bar>'.
            i (str or int, optional): Index of variable or 'mean'. Defaults to "mean".
            unit (str, optional): brian2 unit of variable. Defaults to None.
            label (str, optional): Label of variable. Defaults to None.
            diff_method (str, optional): Differentiation method to use for variables that contain temporal derivatives. Defaults to "central".
            every (int, optional): Get 'every'th sample. Defaults to 1.
            hist_kw (dict, optional): kwargs for histogram. Defaults to {}.

        Returns:
            tuple: (t, v), label
        """
        mean = True if i == "mean" else False
        pattern = r"<(.*?)>"
        variables = re.findall(pattern, v)
        if len(variables) > 0:
            replacement = r"(\1)"
            v = re.sub(pattern, replacement, v)
            ns = {}
            t = None
            for i_v, var in enumerate(variables):
                if var in self.model_parameters:
                    ns[var] = asarray(self.model_parameters[var])
                else:
                    t_, val = self.get_variable(
                        var, mean=False, diff_method=diff_method
                    )
                    if t is None:
                        t = t_
                    else:
                        if len(t_) < len(t):
                            t = t_
                    ns[var] = val
            for var in variables:
                if (var not in self.model_parameters) and (ns[var].shape[-1] > len(t)):
                    assert (ns[var].shape[-1] - len(t)) % 2 == 0
                    dt = (ns[var].shape[-1] - len(t)) // 2
                    ns[var] = ns[var][:, dt:-dt]
            y = eval(v, globals(), ns)
            if mean:
                y = y.mean(0)
        else:
            t, y = self.get_variable(
                v, mean=mean, diff_method=diff_method, hist_kw=hist_kw
            )
            dot = 0
            while "dot" in v.split("_"):
                assert v.split("_")[-1] == "dot"
                dot += 1
                v = "_".join(v.split("_")[:-1])
        if (not mean) and (i != "all"):
            assert v != "psth"
            assert type(i) == int
            y = y[i]

        if (label is None) or (unit is None):
            if v in config.variable_dict.keys():
                if unit is None:
                    unit = config.variable_dict[v]["unit"]
                    if dot > 0:
                        unit = "{}/second**{}".format(unit, dot)
                if label is None:
                    prefix = []
                    if dot > 0:
                        prefix.append(r"\{}ot".format("d" * dot))
                    if mean:
                        prefix.append(r"\bar")
                    label = get_label(v, prefix=prefix, custom_unit=unit)
        if type(unit) is str:
            if unit in units.stdunits.__all__ + units.allunits.__all__:
                unit = eval(unit)
        if type(unit) is Unit:
            y /= asarray(unit)

        if (every > 1) and (v not in ["psth", "spikes"]):
            assert type(every) is int
            t = t[::every]
            y = y[::every]

        return (t, y), label

    def plot_variable(
        self,
        v,
        i="mean",
        label=None,
        unit=None,
        diff_method="central",
        i_from=None,
        i_to=None,
        hist_kw={},
        **kwargs
    ):
        """Plot variable

        Args:
            v (str): Variable name
            i (str or int, optional): Index of variable or 'mean'. Defaults to "mean".
            label (str, optional): Label of variable. Defaults to None.
            unit (str, optional): brian2 unit of variable. Defaults to None.
            diff_method (str, optional): Differentiation method to use for variables that contain temporal derivatives. Defaults to "central".
            i_from (int, optional): Start index to plot. Defaults to None.
            i_to (int, optional): Stop index to plot. Defaults to None.
            hist_kw (dict, optional): kwargs for histogram. Defaults to {}.

        Returns:
            tuple: plt.Figure, plt.Axes
        """

        (t, y), label_ = self.resolve_variable(
            v, i=i, unit=unit, label=label, diff_method=diff_method, hist_kw=hist_kw
        )

        if (i_from is not None) or (i_to is not None):
            if i_from is None:
                i_from = 0
            if i_to is None:
                i_to = len(y)
            assert type(i_from) is int
            assert type(i_to) is int
            t = t[i_from:i_to]
            y = y[i_from:i_to]

        if label is None:
            # overwrite with default
            label = label_

        return plot_variable(t, y, label=label, **kwargs)

    def plot_spikes(self, i="all", **kwargs):
        """Rasterplot of spikes

        Args:
            i (array or list or str, optional): Indices of neurons to plot. Defaults to "all".

        Returns:
            tuple: plt.Figure, plt.Axes
        """
        t_spike, i_spike = self.get_variable("spikes")
        if i != "all":
            if type(i) is int:
                i = [i]
            idxs = np.concatenate([np.argwhere(i_spike == i_) for i_ in i]).reshape(-1)
            t_spike = t_spike[idxs]
            i_spike = i_spike[idxs]

        return plot_spikes(t_spike, i_spike, **kwargs)

    def plot_vs(
        self,
        v1,
        v2,
        i="mean",
        every=1,
        zscore1=False,
        zscore2=False,
        unit1=None,
        unit2=None,
        label1=None,
        label2=None,
        ax=None,
        fontsize=16,
        figsize=(15, 7.5),
        diff_method="central",
        plot_kw={},
    ):
        """Plot variable vs variable

        Args:
            v1 (str): variable name
            v2 (str): variable name
            i (str or int, optional): Index of variable or 'mean'. Defaults to "mean".
            every (int, optional): Get 'every'th sample. Defaults to 1.
            zscore1 (bool, optional): Whether to zscore variable. Defaults to False.
            zscore2 (bool, optional): Whether to zscore variable. Defaults to False.
            unit1 (str, optional): Unit of variable. Defaults to None.
            unit2 (str, optional): Unit of variable. Defaults to None.
            label1 (str, optional): Label of variable. Defaults to None.
            label2 (str, optional): Label of variable. Defaults to None.
            ax (plt.Axes, optional): Axis to plot on. Defaults to None.
            fontsize (int, optional): Fontsize of text. Defaults to 16.
            figsize (tuple, optional): Size of figure. Defaults to (15, 7.5).
            diff_method (str, optional): Differentiation method to use for variables that contain temporal derivatives. Defaults to "central".
            plot_kw (dict, optional): kwargs for plt.Scatter. Defaults to {}.
        """
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)

        (t1, val1), label1 = self.resolve_variable(
            v1, i=i, unit=unit1, label=label1, diff_method=diff_method
        )
        (t2, val2), label2 = self.resolve_variable(
            v2, i=i, unit=unit2, label=label2, diff_method=diff_method
        )

        val1 = val1.reshape((-1, len(t1)))
        val2 = val2.reshape((-1, len(t2)))

        if (v1 == "psth") and (v2 != "psth"):
            # assert val2 != 'psth'
            dt = np.diff(t2)[0]
            idx = np.array((t1 / dt).round(), dtype=int) - int(t2[0] / dt)
            t2 = t2[idx]
            val2 = val2[:, idx]
        elif (v2 == "psth") and (v1 != "psth"):
            idx = np.array((t2 / np.diff(t1)[0]).round(), dtype=int) - int(t2[0] / dt)
            t1 = t1[idx]
            val1 = val1[:, idx]

        # indexing for d/dt
        Nt = min(len(t1), len(t2))
        if len(t1) > Nt:
            assert (len(t1) - Nt) % 2 == 0
            dt = (len(t1) - Nt) // 2
            val1 = val1[:, dt:-dt]
        if len(t2) > Nt:
            assert (len(t2) - Nt) % 2 == 0
            dt = (len(t2) - Nt) // 2
            val2 = val2[:, dt:-dt]

        val1 = val1.reshape(-1)
        val2 = val2.reshape(-1)

        if zscore1:
            val1 = zscore(val1)
        if zscore2:
            val2 = zscore(val2)

        ax.scatter(val1[::every], val2[::every], **plot_kw)
        ax.set_xlabel(label1, size=fontsize)
        ax.set_ylabel(label2, size=fontsize)

    def plot_gating_variables(self, **kwargs):
        """Plot gating variables
            **kwargs: viz.plot_gating_variables

        Returns:
            list: list of plt.Lines
        """
        return plot_gating_variables(
            model_parameters=self.model_parameters, eqs=self.eqs, **kwargs
        )

