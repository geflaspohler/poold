
## Subseasonal Climate Forecasting Example
1. Download the data and model forecast files from https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IOCFCY.

2. Unzip the `data` and `models` directories and place in `examples/s2s_forecast` directory.

To run the example code for the precipitation task at the 56w horizon:
```
cd examples/s2s_forecast
python run_learner_and_hinter.py contest_precip 56w --alg dormplus --hint recent_g --re 1
```

See `src/s2s_environment` and `src/s2s_hints` for examples of the Environment and Hinter classes.