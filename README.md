PoolD
==============================

## Python library for Optimistic Online Learning under Delay 

This library implements algorithms for online learning under delay using optimistic hints. The core of the 
library is the learners module. Learners can be instantiated using the `create` method:
```
import poold
model_list = ["model1", "model2"]
learner = poold.create(learner="dormplus", model_list=model_list, groups=None, T=None)
```
This method initializes an online learning object with an online learning `History` object, that performs the
record keeping of plays and losses observed during online learning. 

At each round  `t`, the learner expects to receive a list of loss feedbacks `losses_fb` (tuples of feedback times and loss dictionaries) and an optimistic hint `hint`, and will produce a play `w`:
```
    w_t = learner.update_and_play(losses_fb, hint=h)
```

These losses and hints should be provided to the the learner. For ease of use, the library defines the
`Environment` and `Hinter` abstract classes, that outline a standard interface for providing loss
functions and hints to the learner respectively.

An example of online learning for subseasonal climate forecasting, as presented in Flaspohler et al. can 
be found in the `examples` directory.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands to setup environment and run testing code
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── examples           <- Example code, including subseasonal forecasting data
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── poold              <- Source code for use in this project.
     tox.ini            <- tox file with settings for running tox; see tox.testrun.org
    │   ├── __init__.py    <- Makes poold a Python module
    │   │
    │   ├── learners       <- Scripts implementing online learning algorithms AdaHedgeD, DUB, DORM and DORM+
    │   │   └── __init__.py
    │   │   └── learners.py
    │   │
    │   ├── environment <- Scripts outlining the basic online learning environment interface
    │   │   └── __init__.py
    │   │   └── environment.py
    │   │
    │   ├── hinters <- Scripts outlining the basic online optimistic hinting interface
    │   │   └── __init__.py
    │   │   └── hinters.py
    │   │
    │   └── utils <- Scripts implementing basic utilities, including online learning play history
    │       └── __init__.py
    │       └── general_utils.py
    │       └── history.py
    │       └── loss_utils.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
    └── setup.py           <- makes project pip installable (pip install -e .) so poold can be imported
--------

## Install project requriements 
Project requirements are managed by pipenv. 
```
pipenv install # to install the dependencies from the Pipfile
pipenv shell # to activate the environment
```