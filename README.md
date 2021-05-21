PoolD
==============================

## Python library for Optimistic Online Learning under Delay 

This library implements algorithms for online learning under delay using optimistic hints. The core of the 
library is the learners module. Learners can be instantiated using the `create` method:
```
import poold
model_list = ["model1", "model2"]
learner, history = poold.create(learner="dormplus", model_list=model_list, partition=None, T=None)
```
This method initializes an online learning and online learning `History` object, that performs the
record keeping of plays and losses observed during online learning. 

At each round  `t`, the learner expects to receive a set of loss feedbacks `losses_fb` at times `times_fb`,
and an optimistic hint `hint`, and will product a play `w`:
```
    w = learner.update(t, times_fb, losses_fb, hint)
```

These losses and hints should be provided to the the learner. For ease of use, the library defines the
`Environemnt` and `Hinter` abstract classes, that outline a standard interface for providing loss
functions and hints to the learner respecitvely.

An example of online learning for subseasonal climate forecasting, as presented in [CITE] can 
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
    ├── setup.py           <- makes project pip installable (pip install -e .) so poold can be imported
    ├── poold              <- Source code for use in this project.
    │   ├── __init__.py    <- Makes poold a Python module
    │   │
    │   ├── learners       <- Scripts implementing online learning algorithms AdaHedgeD, DUB, DORM and DORM+
    │   │   └── __init__.py
    │   │   └── learners.py
    │   │
    │   ├── learners       <- Scripts outlining the basic online learning environment interface
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
--------

## Install project requriements 
`pip install -r requirements.txt`

## To run tests, run
`py.test tests`