from brian2 import *
import config

def zscore(x):
    out = x.copy()
    out -= out.mean()
    out /= out.std()
    return out

def resolve_parameter_dependencies(parameters):
    """Utility function to resolve dependencies between parameters.

    A dependence of a parameter should be denoted by smaller/greater than signs.
    E.g. '<param>'.

    Parameters
    ----------
    parameters : dict
        Dictionary containing the parameters.
    """
    for p1, v1 in parameters.items():
        p1_var = '<{}>'.format(p1)
        for p2, v2 in parameters.items():
            if type(v2) is not str:
                v2 = str(v2)
            if p1_var in v2:
                parameters[p2] = v2.replace(p1_var, "({})".format(v1))


def eval_parameter(parameter, local_vars={}):
    """Utility function to evaluate a parameter if it is a str.

    Parameters
    ----------
    parameter : str, any
        The parameter to evaluate.

    Returns
    -------
    any
        The evaluated parameter.
    """
    if type(parameter) is str:
        if len(local_vars) > 0:
            return eval(parameter, globals(), local_vars)
        else:
            return eval(parameter)
    else:
        return parameter


def get_evaluated_parameters(parameter=None, model_type=None, model_name=None, model_parameters=None, unit=False):
    if model_parameters is None:
        assert model_type is not None
        assert model_name is not None
        model_parameters = config.model_dynamics[model_type][model_name]['parameters']
    
    mp = model_parameters.copy()
    resolve_parameter_dependencies(mp)
    for k_, v_ in mp.items():
        mp[k_] = eval_parameter(v_)

    if not unit:
        for k, v in mp.items():
            mp[k] = asarray(v)
    if parameter is None:
        return mp
    else:
        return mp[parameter]


def get_label(variable, unit=True, custom_unit=None, prefix=None):
    unit_ = custom_unit
    if variable in config.variable_dict.keys():
        label = config.variable_dict[variable]['tex'].strip('$')
        if custom_unit is None:
            unit_ = config.variable_dict[variable]['unit']
    else:
        label = variable
    if prefix is not None:
        if prefix == 'hat':
            prefix = '\hat'
        elif prefix == 'dot':
            prefix = '\dot'
        elif prefix == 'bar':
            prefix = '\bar'
        if type(prefix) is not list:
            prefix = [prefix]
            
        l_ = label.split('_')[0]#[1:]
        for p in prefix:
            l_ = p + '{' + l_ + '}'
        # label = r'$'+ l_ + '_' + label.split('_')[1]
        label = '_'.join([l_] + label.split('_')[1:])
        # label = r'$'+ prefix + '{' + label.split('_')[0][1:] + '}_' + label.split('_')[1]
        
    if unit and (type(unit_) is not int) and (unit_ is not None) and (unit_ != '1'):
        label = label + r'\,{\rm [' + unit_ + r']}'
    label = r'$' + label + r'$'
    return label


def get_unit_prefix(variable):
    if variable in config.variable_dict.keys():
        unit = config.variable_dict[variable]['unit']
        prefix = asarray(eval(unit))
    else:
        prefix = 1
    return prefix


def augment_parameters(model_type=None, model_name=None, simulation_name=None, augmented_model_params={},
                       augmented_eqs={}, augmented_simulation_params={},
                       fixed_model_params={}, fixed_eqs={}, fixed_simulation_params={}):
    """Change default parameters according to provided augmentations.

    For the model and simulation parameters, the parameters in the dictionary is
    replaced by its augmented value. For the equations, the variables are
    replaced in the default string. See 'merge_eqs' for more info.

    Parameters
    ----------
    model_name : str
        The name of the model, which parameters should be used as default
        parameters. See 'config.model_dynamics' for more info.
    simulation_name : str
        The name of the simulation which should be used as default parameters.
        See 'config.simulation' for more info.
    augmented_model_params : dict, optional
        Contains parameter value pairs of the model parameters to change.
        By default {}.
    augmented_eqs : dict, optional
        Contains variable equation pairs of the value equations to change.
        By default {}.
    fixed_simulation_params : dict, optional
        Contains parameter value pairs of the simulation parameters to change.
        By default {}.

    Returns
    -------
    (dict, str, dict)
        The augmented model parameters, equations and simulation parameters.
    """

    if model_name is not None:
        model_params = config.model_dynamics[model_type][model_name]['parameters'].copy()
        for k, v in fixed_model_params.items():
            assert k not in augmented_model_params.keys()
            model_params[k] = v
        for k, v in augmented_model_params.items():
            model_params[k] = v
        eqs = config.model_dynamics[model_type][model_name]['eqs']
        for k, v in fixed_eqs.items():
            assert k not in augmented_eqs.keys()
            eqs = merge_eqs('{}={}'.format(k, v), eqs, [k])
        for k, v in augmented_eqs.items():
            eqs = merge_eqs('{}={}'.format(k, v), eqs, [k])
    else:
        model_params = None
        eqs = None

    if simulation_name is not None:
        simulation_params = config.simulation[simulation_name].copy()
        for k, v in fixed_simulation_params.items():
            assert k not in augmented_simulation_params.keys()
            simulation_params[k] = v
        for k, v in augmented_simulation_params.items():
            simulation_params[k] = v
    else:
        simulation_params = None

    return model_params, eqs, simulation_params


def eqs_to_dict(eqs, indent='\n', return_unit=False):
    """_summary_

    Args:
        eqs (str): equation string
        indent (str, optional): indent between equations. Defaults to '\n'.
        return_unit (bool, optional): return units dictionary. Defaults to False.

    Returns:
        _type_: _description_
    """
    eqs = eqs.replace(" ", "")
    eqs = eqs.split(indent)
    eqs_dict= {}
    unit_dict = {}
    for eq in eqs:
        if len(eq) > 0:
            v, e = eq.split('=')
            # if v == 'alpha_m':
            #     1/0
            eqs_dict[v] = e.split(':')[0]
            unit_dict[v] = e.split(':')[1]
    if return_unit:
        return eqs_dict, unit_dict
    else:
        return eqs_dict


def merge_eqs(eqs_from, eqs_to, variables, indent='\n'):
    """Utility function to merge equations

    Parameters
    ----------
    eqs_from : str
        equations to merge from
    eqs_to : str
        equations to merge to
    variables : list of str
        variables to merge (e.g. 'dv/dt' or 'alpha_m')
    indent : str, optional
        which indent is used to separate equations, by default '\n'

    Returns
    -------
    str
        resulting equations
    """
    # indent = '\n            '
    eqs_from = eqs_from.replace(" ", "")
    eqs_from = eqs_from.split(indent)

    eqs_to = eqs_to.replace(" ", "")
    eqs_to = eqs_to.split(indent)

    vars_from = []
    for eq in eqs_from:
        vars_from.append(eq.split('=')[0])

    vars_to = []
    for eq in eqs_to:
        vars_to.append(eq.split('=')[0])

    for var in variables:
        assert var in vars_from
        assert var in vars_to
        i_from = vars_from.index(var)
        i_to = vars_to.index(var)
        eqs_to[i_to] = eqs_from[i_from]

    eqs = indent.join(eqs_to)
    return eqs


def get_prior(prior_distribution, prior_distribution_kw, num_dim, low=None, high=None):
    import torch
    import sbi.utils

    def update_prior_distribution_parameter(parameters, prior_distribution_kw, num_dim):
        assert type(parameters) is list
        assert len(parameters) > 0
        for p in parameters:
            assert p in prior_distribution_kw.keys()
            v = prior_distribution_kw.pop(p)
            if type(v) is list:
                assert len(v) == num_dim
            else:
                if type(v) is int:
                    v = float(v)
                assert type(v) == float
                v = [v] * num_dim
            prior_distribution_kw[p] = torch.tensor(v)

    # get proper prior parameters
    # currently only works with uniform
    if prior_distribution == 'uniform':
        # initially only uniform supported. low and high arguments still accepted.
        if low is not None:
            assert high is not None
            assert 'low' not in prior_distribution_kw
            assert 'high' not in prior_distribution_kw
            prior_distribution_kw['low'] = low
            prior_distribution_kw['high'] = high
        update_prior_distribution_parameter(
            ['low', 'high'], prior_distribution_kw, num_dim)
        prior = sbi.utils.BoxUniform(**prior_distribution_kw)
    elif prior_distribution == 'normal':
        update_prior_distribution_parameter(
            ['mean', 'std'], prior_distribution_kw, num_dim)
        loc = prior_distribution_kw.pop('mean')
        covariance_matrix = torch.eye(
            num_dim) * prior_distribution_kw.pop('std')
        prior = torch.distributions.MultivariateNormal(
            loc=loc, covariance_matrix=covariance_matrix, **prior_distribution_kw)
    else:
        raise NotImplementedError(
            "'prior_distribution'={} not implemented".format(prior_distribution))

    return prior


def get_prior_from_sbi_id(sbi_id):
    experiment_params = config.experiment['sbi'][sbi_id].copy()
    prior_distribution = experiment_params.pop('prior_distribution', 'uniform')
    prior_distribution_kw = experiment_params.pop('prior_distribution_kw', {}).copy()

    augmented_model_params = experiment_params.pop(
        'augmented_model_params', [])
    augmented_eqs = experiment_params.pop('augmented_eqs', [])
    augmented_simulation_params = experiment_params.pop(
        'augmented_simulation_params', {})

    num_dim = len(augmented_model_params + augmented_eqs) + \
        len(augmented_simulation_params)

    low = experiment_params.pop('low', None)
    high = experiment_params.pop('high', None)
    prior = get_prior(prior_distribution=prior_distribution,
                      prior_distribution_kw=prior_distribution_kw,
                      num_dim=num_dim, low=low, high=high)
    return prior

