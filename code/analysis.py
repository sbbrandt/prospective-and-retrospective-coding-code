import sys, os
import numpy as np
import pickle
import json
import logging
from copy import deepcopy

from joblib import Parallel, delayed

from viz import phase_shift, psth
import config
import logging
from utils import get_prior_from_sbi_id

from tqdm import tqdm



def collect_run_ids(save_dir,
                    required_files=['variables.p', 'augmented_parameters.json'],
                    logger=logging.getLogger(__name__)):
    """Utility function to collect run ids from a save directory.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    required_files : list of str, optional
        List of file endings that should be found in 'save_dir', e.g.
        '<run_id>_<required_file>'. If one of the files is missing, the 'run_id'
        will be accounted as invalid.
        By default ['variables.p', 'augmented_parameters.json'].
    logger : logger object

    Returns
    -------
    list of str
        List containing all valid 'run_ids' found in 'save_dir'.
    """
    files = os.listdir(save_dir)
    run_ids = []
    invalid_ids = []
    for f in files:
        run_id = f.split('_')[0]
        if not run_id in run_ids:
            valid_id = True
            for file_name in required_files:
                file_str = os.path.join(save_dir, '_'.join([run_id, file_name]))
                if not os.path.exists(file_str):
                    valid_id = False
                    if not run_id in invalid_ids:
                        invalid_ids.append(run_id)
                    logger.debug('{} not found. {} invalid ids.'.format(file_str, len(invalid_ids)))
            if valid_id:
                run_ids.append(run_id)
    return run_ids

def get_augmented_parameters(save_dir, file_pre_str=''):
    """Utility function to get the augmented parameters of an experiment.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    file_pre_str : str, optional
        String that is joined with 'augmented_parameters.json' by '_' to form
        the name of the file. Usually the 'run_id'. By default ''.

    Returns
    -------
    dict
        Dictionary containing the augmented parameters.
    """
    file_str = '_'.join([file_pre_str, 'augmented_parameters.json'])
    file_path = os.path.join(save_dir, file_str)
    with open(file_path) as tmp:
        params = json.load(tmp)
    return params

def get_variables(save_dir, file_pre_str=''):
    """Utility function to get the variables of an experiment.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    file_pre_str : str, optional
        String that is joined with 'variables.p' by '_' to form
        the name of the file. Usually the 'run_id'. By default ''.

    Returns
    -------
    dict
        Dictionary containing the variables as saved by the experiment.
    """
    file_str = '_'.join([file_pre_str, 'variables.p'])
    file_path = os.path.join(save_dir, file_str)
    with open(file_path, 'rb') as tmp:
        variables = pickle.load(tmp)
    return variables

def collect_phases(save_dir, provided_run_ids=None,
                   required_files=['variables.p', 'augmented_parameters.json'],
                   v1={'psth': 'all'}, v2={'I': 'mean'},
                   N=10000, hist_kw={}, phase_shift_kw={},
                   num_workers=1, logger=logging.getLogger(__name__)):
    """Get the phases from a save directory.

    Args:
        save_dir (str): The path to the save directory.
        provided_run_ids (list of str, optional): List containing 'run_ids' found in 'save_dir'. If None, output of 'collect_run_ids' is taken. By default None.
        required_files (list of str, optional): List of file endings that should be found in 'save_dir'. Passed to 'collect_run_ids' if 'provided_run_ids' is None and else not used. By default ['variables.p', 'augmented_parameters.json'].
        v1 (dict, optional): variable and index, e.g. {'psth': 'all'}. Defaults to {'psth': 'all'}.
        v2 (dict, optional):  variable and index, e.g. {'I': 'mean'}. Defaults to {'I': 'mean'}.
        N (int, optional): Number of trials in simulation. Defaults to 10000.
        hist_kw (dict, optional): Kwargs to pass to 'np.hist'. Defaults to {}.
        phase_shift_kw (dict, optional): kwargs for 'phase_shift'. Defaults to {}.
        num_workers (int, optional): Number of parallel workers. Defaults to 1.
        logger (logging object, optional): Defaults to logging.getLogger(__name__).

    Returns:
        list of float: A list of phases in the order of 'run_ids'.
    """
    if type(save_dir) == str:
        save_dir = [save_dir]
    phases = []
    for i_sd, sd in enumerate(save_dir):
        if provided_run_ids is None:
            run_ids = collect_run_ids(sd, required_files=required_files)
        else:
            if len(save_dir) > 1:
                run_ids = provided_run_ids[i_sd]
            else:
                run_ids = provided_run_ids
        if num_workers == 1:
            tmp = []
            for run_id in tqdm(run_ids):
                logger.debug('Compute phase for {} ...'.format(run_id))
                tmp.append(
                    get_phase_from_save_dir(
                        sd, file_pre_str=run_id, file_str='variables.p',
                        v1=v1, v2=v2, hist_kw=hist_kw,
                        N=N, phase_shift_kw=phase_shift_kw
                    )
                )
                logger.debug('... compute phase done.')
        else:
            tmp = Parallel(n_jobs=num_workers)(delayed(get_phase_from_save_dir)(
                sd, file_pre_str=run_id, file_str='variables.p',
                v1=v1, v2=v2, hist_kw=hist_kw,
                N=N, phase_shift_kw=phase_shift_kw
            ) for run_id in run_ids)

        phases += tmp
    return phases

def theta_from_augmented_parameters(augmented_parameters, valid_types=['model_parameters', 'eqs']):
    """Get the scaling factors theta from the 'augmented_parameters' dictionary.

    Parameters
    ----------
    augmented_parameters : dict
         Dictionary containing the augmented parameters.
    valid_types : list
        List with types of parameters to consider. I.e. one or both of ['model_parameters', 'eqs']. Defaults to ['model_parameters', 'eqs'].

    Returns
    -------
    list of str
        List containing the scaling factors as str.
    """
    theta = []
    for parameter_type, _ in augmented_parameters.items():
        if parameter_type in valid_types:
            for _, value in augmented_parameters[parameter_type].items():
                theta.append(float(value.split('*')[0]))
    return theta


def collect_thetas(save_dir, provided_run_ids=None, required_files=['variables.p', 'augmented_parameters.json'], valid_types=['model_parameters', 'eqs'], logger=logging.getLogger(__name__)):
    """Utility function to get thetas from a save directory.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    provided_run_ids : list, optional
        List containing 'run_ids' found in 'save_dir'. If None, output of
        'collect_run_ids' is taken. By default None.
    required_files : list of str, optional
        List of file endings that should be found in 'save_dir'. Passed to
        'collect_run_ids' if 'provided_run_ids' is None and else not used.
        By default ['variables.p', 'augmented_parameters.json'].
    valid_types : list
        List with types of parameters to consider. I.e. one or both of ['model_parameters', 'eqs']. Defaults to ['model_parameters', 'eqs'].


    Returns
    -------
    list of lists
        List of thetas in the order of 'run_ids'.
    """
    if type(save_dir) == str:
        save_dir = [save_dir]
    if len(save_dir) > 1:
        assert len(save_dir) == len(run_ids)
    thetas = []
    for i_sd, sd in enumerate(save_dir):
        if provided_run_ids is None:
            run_ids = collect_run_ids(sd, required_files=required_files)
        else:
            if len(save_dir) > 1:
                run_ids = provided_run_ids[i_sd]
            else:
                run_ids = provided_run_ids
        for run_id in tqdm(run_ids):
            augmented_parameters = get_augmented_parameters(sd, run_id)
            theta = theta_from_augmented_parameters(
                augmented_parameters, valid_types=valid_types)
            thetas.append(theta)
    return thetas

def collect_rounds(save_dir, max_rounds=10):
    """Utility function to get the summaries of the rounds of an sbi experiment.

    Parameters
    ----------
    save_dir : str
        The path to the save directory.
    max_rounds : int, optional
        Maximum number of rounds to iterate through. By default 10.

    Returns
    -------
    tuple of lists
        For each, thetas and phases, the round results in a list.
    """
    thetas = []
    phases = []
    if type(save_dir) != list:
        save_dir = [save_dir]
    for s_dir in save_dir:
        print(s_dir)
        for r in range(max_rounds):
            file_str = 'simulation_results_round_{}.p'.format(r)
            file_path = os.path.join(s_dir, file_str)
            if os.path.exists(file_path):
                print('\tround {}: found.'.format(r))
                with open(file_path, 'rb') as tmp:
                    results = pickle.load(tmp)
                thetas.append(results['theta'])
                phases.append(results['x'])
            else:
                print('\tround {}: not found.'.format(r))
    return thetas, phases

def get_variable_from_save_dir(v, i, save_dir, file_pre_str='',
                               file_str='variables.p', hist_kw={'bins': 100}, dt=1e-4, N=None):
    """Utility function to quickly get a variable from a save directory.

    Parameters
    ----------
    v : str
        Variable name.
    i : str, int
        Can be either 'mean', 'all' or the index of the neuron.
    save_dir : str
        The path to the save directory.
    file_pre_str : str, optional
        String that is joined with 'variables.p' by '_' to form
        the name of the file. Usually the 'run_id'. By default ''.
    file_str : str, optional
        Name of the file where the variables are saved (pickled).
        By default 'variables.p'.
    hist_kw : dict, optional
        Kwargs to pass to 'np.hist'. Recommended is '{'bins': 100}'.
        By default {}.
    N : int, None, optional
        To estimate the spike rate of the neurons, the number of neurons is
        required. If not provided, the number of Neurons that spiked is taken.
        By default None.

    Returns
    -------
    tuple
        A tuple of two arrays or tensors with the first being time and the
        second the variable.
    """
    try:
        variables = get_variables(save_dir, file_pre_str)
        if v == 'psth':
            if (len(hist_kw) == 0) and ('psth' in variables.keys()):
                    t = variables['psth']['t']
                    y = variables['psth']['all']
            else:
                assert 'spikes' in variables.keys()
                t_spikes = variables['spikes']['t']
                i_spikes = variables['spikes']['all']
                if len(t_spikes) > 0:
                    if N is None:
                        N = len(np.unique(i_spikes))  # best approximation to count neurons that spiked
                    t, y = psth(t_spikes, N, hist_kw)
                else:
                    t = y = np.float64('nan')
        else:
            assert v in variables.keys()
            y = variables[v][i]
            if 't' in variables[v].keys():
                t = variables[v]['t']
            else:
                t = np.arange(0, round(len(y)*dt, int(np.log10(1/dt))), dt)

    except pickle.UnpicklingError:
        print('UnpicklingError; file not loaded')
        t = np.float64('nan')
        y = np.float64('nan')
    return t, y

def get_phase_from_save_dir(save_dir, file_pre_str='', file_str='variables.p',
                            v1={'psth': 'all'}, v2={'I': 'mean'}, hist_kw={},
                            N=10000, phase_shift_kw={}, initialize=True,
                            logger=logging.getLogger(__name__)):
    """Utility function to quickly get a phase from a variable file.

    Parameters
    ----------
    save_dir : str, optional
        The path to the save directory. By default '.'.
    file_pre_str : str, optional
        String that is joined with 'variables.p' by '_' to form
        the name of the file. Usually the 'run_id'. By default ''.
    file_str : str, optional
        Name of the file where the variables are saved (pickled).
        By default 'variables.p'.
    v1 : dict, optional
        The name of the first variable for the phase estimation. E.g. a positive
        phase means v1 is phase advanced to v2. Expected form is
        '{<variable_name>: <i>}, where '<i>' can be the index of a neuron, 'all'
        or 'mean' (recommended). If '<variable_name>="psth"', '<i>' should be
        'all'. By default {'psth': 'all'}.
    v2 : dict, optional
        The name of the second variable for the phase estimation. E.g. a
        positive phase means v1 is phase advanced to v2. Expected form is
        '{<variable_name>: <i>}, where '<i>' can be the index of a neuron, 'all'
        or 'mean' (recommended). If '<variable_name>="psth"', '<i>' should be
        'all'. By default {'I': 'mean'}.
    hist_kw : dict, optional
        Kwargs to pass to 'np.hist'. Recommended is '{'bins': 100}'.
        By default {}.
    N : int, None, optional
        To estimate the spike rate of the neurons, the number of neurons is
        required. If not provided, the number of Neurons that spiked is taken.
        By default None.
    phase_shift_kw : dict, optional
        Kwargs that are passed to the phase shift function. Here, the frequency
        of the sinusoidal input is crucial for a proper phase estimation.
        E.g. '{'f_0': 10}' initialises the fit function at the (correct)
        frequency of 10 Hz. By default {}.

    Returns
    -------
    float
        The phase shift between v1 and v2
    """
    s = []
    for v_ in [v1, v2]:
        v = list(v_.keys())[0]
        i = v_[v]
        s.append(get_variable_from_save_dir(v, i, save_dir, file_pre_str, file_str, hist_kw, N))

    # print(s)
    if (s[0][1] is None) or (s[1][1] is None):
        shift = float('nan')
    elif np.any(np.isnan(s[0][1])) or np.any(np.isnan(s[1][1])):
        shift = float('nan')
    else:
        phase_shift_kw['initialize'] = initialize
        shift = phase_shift(s[0], s[1], logger=logger, **phase_shift_kw)
    return shift


def collect_save_dirs(experiment_id, experiment_name='sbi', include_config=False, save_dir=None):
    # experiment ids as list
    experiment_ids = [experiment_id]
    # include experiment ids that start from other experiments (include their samples from posterior, but are identical)
    if include_config:
        assert experiment_name == 'sbi'
        for exp_id, exp_params in config.experiment[experiment_name].items():
            if 'rounds_from_id' in exp_params:
                rounds_from_id = deepcopy(exp_params)['rounds_from_id']
                if type(rounds_from_id) is str:
                    rounds_from_id = [rounds_from_id]
                if experiment_id in rounds_from_id:
                    experiment_ids.append(exp_id)
    
    # get all save directories
    save_dirs = []
    for exp_id in experiment_ids:
        if save_dir is None:
            save_dir = config.experiment[experiment_name][exp_id]['save_dir']
        exp_dir = os.path.join(save_dir, experiment_name)
        if exp_dir[0] == '~':
            assert exp_dir[1] == '/'
            exp_dir = os.path.join(
                os.environ.get('HOME'), exp_dir[2:])
        for tmp_id in os.listdir(exp_dir):
            if tmp_id.split('_')[0] == exp_id:
                save_dirs.append(os.path.join(exp_dir, tmp_id))
    return save_dirs


def get_posterior(experiment_id, experiment_dir=None, experiment_ids='all', return_estimator=False, method='SNPE', from_rounds=True, log10=False, SNPE_kw={}):
    import sbi
    from sbi import utils
    from sbi import inference
    import torch
    if experiment_ids == 'all':
        save_dirs = collect_save_dirs(
            experiment_id, include_config=True, save_dir=experiment_dir)
    else:
        assert type(experiment_ids) is list
        for exp_dir in experiment_ids:
            assert os.path.exists(exp_dir)
        save_dirs = experiment_ids
    if from_rounds:
        thetas, phases = collect_rounds(save_dirs)
    else:
        thetas = []
        phases = []
        for save_dir in save_dirs:
            run_ids = collect_run_ids(save_dir)
            thetas += collect_thetas(save_dir, provided_run_ids=run_ids)
            phases += collect_phases(save_dir, provided_run_ids=run_ids)
        phases = [torch.tensor(phases, dtype=torch.float32).reshape(-1, 1)]
        thetas = torch.tensor(thetas, dtype=torch.float32)
        if log10:
            thetas = torch.log10(thetas)
        thetas = [thetas]
    n_theta = thetas[0].shape[1]

    prior = get_prior_from_sbi_id(experiment_id)

    if method == 'SNPE':
        inference = sbi.inference.SNPE(prior=prior, **SNPE_kw)
    elif method == 'SNLE':
        inference = sbi.inference.SNLE(prior=prior)
    restriction_estimator = sbi.utils.RestrictionEstimator(prior=prior)
    
    proposal = prior
    for r in range(len(thetas)):
        estimator = inference.append_simulations(
            thetas[r], phases[r], proposal=proposal)
        restriction_estimator.append_simulations(thetas[r], phases[r])
        restriction_estimator.train()
        proposal = restriction_estimator.restrict_prior()

    # all_theta, all_x, _ = restriction_estimator.get_simulations()

    if method == 'SNPE':
        # build inference object
        
        # add simulations to inference object and train
        # estimator = inference.append_simulations(all_theta, all_x).train()
        estimator = inference.train()

        # build posterior
        posterior = inference.build_posterior()
    elif method == 'SNLE':
        inference = sbi.inference.SNLE()
        # estimator = inference.append_simulations(all_theta, all_x).train()
        estimator = inference.train()

        class posterior_wrapper:
            def __init__(self, estimator, prior):
                self.estimator = estimator
                self.prior = prior
                self.x_o = None

            def get_posterior(self, x_o=None):
                if x_o is None:
                    assert self.x_o is not None
                    x_o = self.x_o
                potential_fn, parameter_transform = sbi.inference.likelihood_estimator_based_potential(
                    self.estimator, self.prior, x_o
                )
                posterior = sbi.inference.MCMCPosterior(
                    potential_fn, proposal=self.prior, theta_transform=parameter_transform
                )
                return posterior

            def sample(self, n, x_o=None, **kwargs):
                posterior = self.get_posterior(x_o)
                return posterior.sample(n, **kwargs)

            def set_default_x(self, x_o):
                self.x_o = x_o

            def log_prob(self, x):
                posterior = self.get_posterior()
                return posterior.log_prob(x)

        posterior = posterior_wrapper(estimator, prior)

    if return_estimator:
        return posterior, estimator
    else:
        return posterior


def get_conditional_posterior(posterior_estimator, prior, x_o, condition):
    import sbi
    import sbi.utils
    import sbi.inference
    import sbi.analysis
    # from sbi.analysis import conditional_potential

    potential_fn, theta_transform = sbi.inference.posterior_estimator_based_potential(
        posterior_estimator, prior=prior, x_o=x_o
    )

    # there is a typo in sbi for "conditonal_potential" that might be updated in a later package! took me 1 day... :facepalm:
    conditioned_potential_fn, restricted_tf, restricted_prior = sbi.analysis.conditonal_potential(
        potential_fn=potential_fn,
        theta_transform=theta_transform,
        prior=prior,
        condition=condition,  # the first three values are arbitrary and are ignored internally
        dims_to_sample=[1, 2, 3],
    )

    mcmc_posterior = sbi.inference.MCMCPosterior(
        potential_fn=conditioned_potential_fn,
        theta_transform=restricted_tf,
        proposal=restricted_prior,
        num_workers=72,
        method='slice_np_vectorized'
    )

    return mcmc_posterior

