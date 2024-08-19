# information about some parameters to allow easy and nice plotting
variable_dict = {
    "t": {"name": "time", "tex": r"$t$", "unit": "s"},
    "psth": {
        "name": "PSTH",
        "tex": r"$r_{\rm out}$",
        "unit": "Hz",
    },
    "v": {
        "name": "Membrane Voltage",
        "tex": r"$V_{\rm m}$",
        "unit": "mV",
    },
    "I": {
        "name": "Input Current",
        "tex": r"$I_{\rm in}$",
        "unit": "nA",
    },
    "g_na": {
        "name": "Sodium Conductance",
        "tex": r"$g_{\rm Na}$",
        "unit": "nS",  # 'siemens',
    },
    "g_na_bar": {
        "name": "Maxmimal Sodium Conductance",
        "tex": r"$\bar{g}_{\rm Na}$",
        "unit": "nS",  # 'siemens',
    },
    "g_na_inf": {
        "name": "Sodium Conductance Steady State",
        "tex": r"$g_{\rm Na}^\infty$",
        "unit": "nS",  # 'siemens',
    },
    "g_kd": {
        "name": "Potassium Conductance",
        "tex": r"$g_{\rm K}$",
        "unit": "nS",  # 'siemens',
    },
    "g_kd_bar": {
        "name": "Maximal Potassium Conductance",
        "tex": r"$\bar{g}_{\rm K}$",
        "unit": "nS",  # 'siemens',
    },
    "I_na": {
        "name": "Sodium Current",
        "tex": r"$I_{\rm Na}$",
        "unit": "nA",
    },
    "I_kd": {"name": "Potassium Current", "tex": r"$I_{\rm K}$", "unit": "nA"},
    "I_kd_inf": {
        "name": "Steady State Potassium Current",
        "tex": r"$I_{\rm K}^\infty$",
        "unit": "nA",
    },
    "tau_eff": {
        "name": "Effective Time Constant",
        "tex": r"$\tau_{\rm eff}$",
        "unit": "ms",
    },
    "h_inf": {"name": "$h_\infty$", "tex": r"$h_\infty$", "unit": "1"},
    "h": {"name": "h", "tex": r"$h$", "unit": "1"},
    "u_pro": {"name": r"$V+\tau(V)\dot{V}$", "tex": r"$\tilde{u}$", "unit": "mV"},
    "g_l": {"name": "Leakage Conductance", "tex": r"$g_\ell$", "unit": "siemens"},
    "E_l": {"name": "Leakage Potential", "tex": r"$E_\ell$", "unit": "mV"},
    "Cm": {"name": "Membrane Capacitance", "tex": r"$C_{\rm m}$", "unit": "farad"},
    "alpha_m": {"name": "Opening Rate m", "tex": r"$\alpha_{\rm m}$", "unit": 1},
    "beta_m": {"name": "Closing Rate m", "tex": r"$\beta_{\rm m}$", "unit": 1},
    "m_inf": {"name": "Steady State m", "tex": r"$m_\infty$", "unit": 1},
    "m_inf^3": {"name": "m^3", "tex": r"$m_\infty^3$", "unit": 1},
    "tau_m": {"name": "Time Constant m", "tex": r"$\tau_m$", "unit": "ms"},
    "alpha_n": {"name": "Opening Rate n", "tex": r"$\alpha_{\rm n}$", "unit": 1},
    "beta_n": {"name": "Closing Rate n", "tex": r"$\beta_{\rm n}$", "unit": 1},
    "n_inf": {"name": "Steady State n", "tex": r"$n_\infty$", "unit": 1},
    "tau_n": {"name": "Time Constant n", "tex": r"$\tau_n$", "unit": "ms"},
    "alpha_h": {"name": "Opening Rate h", "tex": r"$\alpha_{\rm h}$", "unit": 1},
    "beta_h": {"name": "Closing Rate h", "tex": r"$\beta_{\rm h}$", "unit": 1},
    "h_inf": {"name": "Steady State h", "tex": r"$h_\infty$", "unit": 1},
    "tau_h": {"name": "Time Constant h", "tex": r"$\tau_h$", "unit": "ms"},
    "lam_alpha_m": {
        "tex": r"$\lambda_{\alpha_m}$",
        "unit": 1,
    },
    "lam_beta_h": {
        "tex": r"$\lambda_{\beta_h}$",
        "unit": 1,
    },
    "lam_g_l": {
        "tex": r"$\lambda_{g_\ell}$",
        "unit": 1,
    },
}

# simulation parameters
simulation = {
    "brian2": {
        "N": 10000,
        "method": "exponential_euler",
        "spike_count_method": "find_peaks",
        "threshold": None,
        "refractory": "20*ms",
        "t_init": "1000*ms",  # run for t_init without measurements
        "t": "500 * ms",
        "v_min_peak": "-10*mV",
    },
}

# model parameters and equations
model_dynamics = {
    # model_dynamics[model_type][model_name]['parameters'/'eqs']
    "HH_model": {
        "hippocampus": {
            "parameters": {  # https://link.springer.com/content/pdf/10.1007/s10827-007-0038-6.pdf
                # hh parameters
                "area": "20000*umetre**2",
                "Cm": "1*ufarad/cm**2*<area>",
                "g_l": "5e-5*siemens/cm**2*<area>",
                "E_l": "-60*mV",
                "E_kd": "-90*mV",
                "E_na": "50*mV",
                "g_na_bar": "100*msiemens/cm**2*<area>",
                "g_kd_bar": "30*msiemens/cm**2*<area>",
                "VT": "-63*mV",
                # input parameters
                "I0": "0 * nA",
                "I1": "0 * nA",
                "f": "10**<f_exp> * Hz",
                "f_exp": 1,
                "s": "0 * nA",
                "tau_noise": "10 * ms",
                "I_unit": "nA",
            },
            "eqs": """
                dv/dt = (g_l*(E_l-v) - g_na_bar*(m*m*m)*h*(v-E_na) - g_kd_bar*(n*n*n*n)*(v-E_kd) + int(not_refractory)*I)/Cm : volt
                I = I0 + I1 * sin(2 * pi * f * t) + I_noise : ampere 
                dI_noise/dt = - I_noise/tau_noise + s * sqrt(2/dt/tau_noise) * randn() : ampere
                dm/dt = alpha_m*(1-m)-beta_m*m : 1
                dn/dt = alpha_n*(1-n)-beta_n*n : 1
                dh/dt = alpha_h*(1-h)-beta_h*h : 1
                alpha_m = 0.32*(mV**-1)*4*mV/exprel((13*mV-v+VT)/(4*mV))/ms : Hz
                beta_m = 0.28*(mV**-1)*5*mV/exprel((v-VT-40*mV)/(5*mV))/ms : Hz
                alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
                beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
                alpha_n = 0.032*(mV**-1)*5*mV/exprel((15*mV-v+VT)/(5*mV))/ms : Hz
                beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
                """,
        },
        "cortex": {  # https://neuronaldynamics.epfl.ch/online/Ch2.S2.html
            "parameters": {
                # hh parameters
                "area": "20000*umetre**2",
                "Cm": "1*ufarad/cm**2*<area>",
                "g_l": "0.3*msiemens/cm**2*<area>",
                "E_l": "-65*mV",
                "E_kd": "-77*mV",
                "E_na": "55*mV",
                "g_na_bar": "40*msiemens/cm**2*<area>",
                "g_kd_bar": "35*msiemens/cm**2*<area>",
                # input parameters
                "I0": "0 * nA",
                "I1": "0 * nA",
                "f": "(10**(<f_exp>)) * Hz",
                "f_exp": "1",
                "s": "0 * nA",
                "tau_noise": "10 * ms",
                "I_unit": "nA",
            },
            "eqs": """
                dv/dt = (g_l*(E_l-v) - g_na_bar*(m*m*m)*h*(v-E_na) - g_kd_bar*(n*n*n*n)*(v-E_kd) + int(not_refractory)*I)/Cm : volt
                I = I0 + I1 * sin(2 * pi * f * t) + I_noise : ampere
                dI_noise/dt = - I_noise/tau_noise + s * sqrt(2/dt/tau_noise) * randn() : ampere
                dm/dt = alpha_m*(1-m)-beta_m*m : 1
                dn/dt = alpha_n*(1-n)-beta_n*n : 1
                dh/dt = alpha_h*(1-h)-beta_h*h : 1
                alpha_n = 0.02*(mV**-1)*9*mV/exprel((25*mV-v)/(9*mV))/ms : Hz
                beta_n = 0.002*(mV**-1)*9*mV/exprel((v-25*mV)/(9*mV))/ms : Hz
                alpha_m = 0.182*(mV**-1)*9*mV/exprel(-(35*mV+v)/(9*mV))/ms : Hz
                beta_m = 0.124*(mV**-1)*9*mV/exprel((35*mV+v)/(9*mV))/ms : Hz
                alpha_h = 0.25*exp(-(v+90*mV)/(12*mV))/ms : Hz
                beta_h = 0.25*exp((v+62*mV)/(6*mV))/exp((v+90*mV)/(12*mV))/ms : Hz
                """,
        },
    },
}

# experiment_parameters
experiment = {
    "sbi": {
        "001": {  # 9 parameter model
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "augmented_model_params": [
                "g_l",
                "g_na_bar",
                "g_kd_bar",
            ],
            "augmented_eqs": [
                "alpha_m",
                "beta_m",
                "alpha_h",
                "beta_h",
                "alpha_n",
                "beta_n",
            ],
            "fixed_simulation_params": {"threshold": "m>.5", "refractory": "m>.5"},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "low": None,
            "high": None,
            "num_rounds": 5,
            "simulations_per_round": 1000,
            "num_workers": 32,
            "log10": True,
            "prior_distribution": "normal",
            "prior_distribution_kw": {"mean": 0, "std": 0.1},
            "save_dir": "../data/",
            "fixed_model_params": {
                "I0": "rate_based",
                "I1": "0.01*nA",
                "s": "0.02*nA",
                "rate_based_kw": {
                    "target_fr": 10,
                    "min_fr": 5,
                    "max_fr": 15,
                    "min_I0": -10,
                    "max_I0": 10,
                },
            },
        },
        "002": {  # 3 parameter model
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "augmented_model_params": ["g_l"],
            "augmented_eqs": ["alpha_m", "beta_h"],
            "fixed_simulation_params": {"threshold": "m>.5", "refractory": "m>.5"},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "low": None,
            "high": None,
            "num_rounds": 5,
            "simulations_per_round": 1000,
            "num_workers": 32,
            "log10": True,
            "prior_distribution": "normal",
            "prior_distribution_kw": {"mean": 0, "std": 0.1},
            "save_dir": "../data/",
            "fixed_model_params": {
                "I0": "rate_based",
                "I1": "0.01*nA",
                "s": "0.02*nA",
                "rate_based_kw": {
                    "target_fr": 10,
                    "min_fr": 5,
                    "max_fr": 15,
                    "min_I0": -10,
                    "max_I0": 10,
                },
            },
        },
        "003": {  # continuing from 002
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "augmented_model_params": ["g_l"],
            "augmented_eqs": ["alpha_m", "beta_h"],
            "fixed_simulation_params": {"threshold": "m>.5", "refractory": "m>.5"},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "low": None,
            "high": None,
            "num_rounds": 5,
            "simulations_per_round": 1000,
            "num_workers": 32,
            "log10": True,
            "prior_distribution": "normal",
            "prior_distribution_kw": {"mean": 0, "std": 0.1},
            "save_dir": "../data/",
            "fixed_model_params": {
                "I0": "rate_based",
                "I1": "0.01*nA",
                "s": "0.02*nA",
                "rate_based_kw": {
                    "target_fr": 10,
                    "min_fr": 5,
                    "max_fr": 15,
                    "min_I0": -10,
                    "max_I0": 10,
                },
            },
            "rounds_from_id": ["002"],
        },
    },
    "posterior_check": {
        "001": {  # 050
            "sbi_experiment_id": "002",
            "posterior_save_path": "../data/sbi/003/posterior.p",
            "x_o": 0.01,
            "n_simulations": 2,
            "num_workers": 36,
            "save_dir": "../data/",
        },
        "002": {  # 051
            "sbi_experiment_id": "002",
            "posterior_save_path": "../data/sbi/003/posterior.p",
            "x_o": 0.0,
            "n_simulations": 1000,
            "num_workers": 36,
            "save_dir": "../data/",
        },
        "003": {  # 052
            "sbi_experiment_id": "002",
            "posterior_save_path": "../data/sbi/003/posterior.p",
            "x_o": -0.01,
            "n_simulations": 1000,
            "num_workers": 36,
            "save_dir": "../data/",
        },
    },
    "frequency_response": {
        "001": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "f_0": 0.4,  # lower bound frequency range
            "f_1": 40,  # upper bound frequency range
            "f_N": 50,  # number of frequency evaluations
            "f_dist": "log10",  # frequency distribution ('linear' or 'log10')
            "fixed_simulation_params": {
                "N": 10000,
                "refractory": "m>.5",
            },
            "fixed_model_params": {"I0": "0.082*nA", "I1": "0.01*nA", "s": "0.02*nA"},
            "fixed_eqs": {},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "save_dir": "../data/",
            "n_jobs": 1,
            "seed": 20221107,
        },
        "002": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "hippocampus",
            "f_0": 0.4,  # lower bound frequency range
            "f_1": 40,  # upper bound frequency range
            "f_N": 50,  # number of frequency evaluations
            "f_dist": "log10",  # frequency distribution ('linear' or 'log10')
            "fixed_simulation_params": {
                "N": 10000,
                "refractory": "m>.5",
            },
            "fixed_model_params": {"I0": "-0.014*nA", "I1": "0.01*nA", "s": "0.02*nA"},
            "fixed_eqs": {},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "save_dir": "../data",
            "n_jobs": 1,
            "seed": 20221107,
            "adapt_init": True,
        },
        "003": {  # cortex with I0 s.t. mean r0 is 20 Hz
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "f_0": 0.4,  # lower bound frequency range
            "f_1": 40,  # upper bound frequency range
            "f_N": 50,  # number of frequency evaluations
            "f_dist": "log10",  # frequency distribution ('linear' or 'log10')
            "fixed_simulation_params": {
                "N": 10000,
                "refractory": "m>.5",
            },
            "fixed_model_params": {"I0": "0.187*nA", "I1": "0.01*nA", "s": "0.05*nA"},
            "fixed_eqs": {},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "save_dir": "../data/",
            "n_jobs": 1,
            "seed": 20221107,
        },
        "004": {  # cortex with I0 s.t. mean r0 is 30 Hz
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "f_0": 0.4,  # lower bound frequency range
            "f_1": 40,  # upper bound frequency range
            "f_N": 50,  # number of frequency evaluations
            "f_dist": "log10",  # frequency distribution ('linear' or 'log10')
            "fixed_simulation_params": {
                "N": 10000,
                "refractory": "m>.5",
            },
            "fixed_model_params": {"I0": "0.344*nA", "I1": "0.01*nA", "s": "0.05*nA"},
            "fixed_eqs": {},
            "save_variables": {
                "I": {"mean": True, "i": 0},
                "v": {"mean": True, "i": 0},
            },
            "save_spikes": True,
            "save_dir": "../data/",
            "n_jobs": 1,
            "seed": 20221107,
        },
    },
    "sinusoidals": {
        "001": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "frequencies": [0.5, 2, 3],
            "amplitudes": [-1, -0.5, 0.1],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_dependent_variables": {"g_na": {"mean": True}},
            "save_spikes": True,
            "num_workers": 16,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "0.082*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
        "002": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "frequencies": [1.5, 2.4, 3.3, 4.4, 4.7],
            "amplitudes": [0.5, 0.4, 0.3, 0.4, 0.5],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_spikes": True,
            "num_workers": 32,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "0.082*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
        "003": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "cortex",
            "frequencies": [0.5, 2, 3, 11],
            "amplitudes": [-1, -0.5, 0.1, 0.02],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_dependent_variables": {"g_na": {"mean": True}},
            "save_spikes": True,
            "num_workers": 16,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "0.082*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
        "004": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "hippocampus",
            "frequencies": [0.5, 2, 3],
            "amplitudes": [-1, -0.5, 0.1],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_dependent_variables": {"g_na": {"mean": True}},
            "save_spikes": True,
            "num_workers": 16,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "-0.014*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
        "005": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "hippocampus",
            "frequencies": [1.5, 2.4, 3.3, 4.4, 4.7],
            "amplitudes": [0.5, 0.4, 0.3, 0.4, 0.5],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_dependent_variables": {"g_na": {"mean": True}},
            "save_spikes": True,
            "num_workers": 16,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "-0.014*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
        "006": {
            "model_type": "HH_model",
            "simulation_name": "brian2",
            "model_name": "hippocampus",
            "frequencies": [0.5, 2, 3, 11],
            "amplitudes": [-1, -0.5, 0.1, 0.02],
            "save_variables": {
                "I": {"mean": True},
                "v": {"mean": True},
                "m": None,
                "h": None,
            },
            "save_dependent_variables": {"g_na": {"mean": True}},
            "save_spikes": True,
            "num_workers": 16,
            "num_simulations": 2000,
            "save_dir": "/scratch/snx3000/sbrandt/experiments/",
            "fixed_model_params": {
                "I0": "-0.014*nA",
                "I1": "0.01*nA",
                "s": "0.02*nA",
            },
            "fixed_simulation_params": {
                "threshold": "m>.5",
                "refractory": "m>.5",
                "t": "1*second",
                "N": 5000,
            },
        },
    },
}
