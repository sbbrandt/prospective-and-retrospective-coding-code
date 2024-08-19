# prospective-and-retrospective-coding-code
Code for paper 'Prospective and retrospective coding in cortical neurons'

The code is organized in the following way:
- 'code/' contains the code base
- 'notebooks/' contains jupyter lab files to generate the figures of the manuscript
    - Some of the smaller simulations have to be run within the manuscript
    - For the larger simulations, the data summaries are provided in 'data/'
    - Each file contains the description of how to generate the data from scratch
- 'data/' contains some of the data and its summaries required to generate the figures

## Setting up virtual environment and install requirements
1. Setting up environment:
```bash
python -m venv venv
```

2. Activate environment
```bash
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Running minimal example
A minimal example of the simulation can be found in 'notebooks/Fig-1.ipynb'. The notebook contains the initial simlations for the two parameter models of the cortex and hippocampus. The cortex model is the one used throughout the manuscript, except of the comparison of the two models in the frequency response figure.

## Running experiments
Some of the experiments are more extensive and require more time or are better suited for a cluster.

Experiments can be run by executing
```bash
python run_experiment.py --experiment_name <experiment_name> --experiment_id <experiment_id>
```

Data summaries for these experiments are provided in 'data/<experiment_name>/<experiment_id>/'
