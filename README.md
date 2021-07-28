PoolD
==============================
## Python library for Optimistic Online Learning under Delay 
This Python package implements algorithms for online learning under delay using optimistic hints. More details on the algorithms and their regret properties can be found in the manuscript [Online Learning with Optimism and Delay](https://arxiv.org/abs/2106.06885).

```

@InProceedings{pmlr-v139-flaspohler21a,
  title = 	 {Online Learning with Optimism and Delay},
  author =       {Flaspohler, Genevieve E and Orabona, Francesco and Cohen, Judah and Mouatadid, Soukayna and Oprescu, Miruna and Orenstein, Paulo and Mackey, Lester},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {3363--3373},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/flaspohler21a/flaspohler21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/flaspohler21a.html},
  abstract = 	 {Inspired by the demands of real-time climate and weather forecasting, we develop optimistic online learning algorithms that require no parameter tuning and have optimal regret guarantees under delayed feedback. Our algorithms—DORM, DORM+, and AdaHedgeD—arise from a novel reduction of delayed online learning to optimistic online learning that reveals how optimistic hints can mitigate the regret penalty caused by delay. We pair this delay-as-optimism perspective with a new analysis of optimistic learning that exposes its robustness to hinting errors and a new meta-algorithm for learning effective hinting strategies in the presence of delay. We conclude by benchmarking our algorithms on four subseasonal climate forecasting tasks, demonstrating low regret relative to state-of-the-art forecasting models.}
}
```
The core of the library is the learners module. Learners can be instantiated using the `create` method:
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

## Install  
The PoolD pacakge can be installed [via pip](https://pypi.org/project/poold/):
```
pip install poold
```
