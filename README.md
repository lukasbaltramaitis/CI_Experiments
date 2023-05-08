# CI_Experiments

This is the evaluation tool for the application of Causal Inference algorithms (from Ylearn tool:
https://ylearn.readthedocs.io/en/latest/index.html)
on the business process event logs that has defined outcome for the case and Case ID, 
Activity and Timestamp variables.

## Setup

### Setup the environment and Git-LFS

1. Install conda from: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
2. Then open terminal and execute these commands:
```
conda create -n ci_experiments python=3.8
conda activate ci_experiments
```
3. Download and install git lfs from: https://git-lfs.github.com/ 
   Git lfs is used for huge data files -- they should be in `data` folder.

4. In the terminal execute command (once per user):
```
git lfs install
```
5. In the terminal execute command to download the git LFS objects:
```
git lfs fetch
```
### Work with the environment

Save new dependencies
```
conda env export --from-history > environment.yaml
```

Apply new changes
```
conda env update --file environment.yml --name ci_experiments
```

### Execute encoding unit tests
1. In terminal go to the project parent folder.
2. Execute the command:
```
python -m unittest CI_Experiments.tests.encoding_test
```
3. Tests can be found in CI_Experiments/tests/encoding_test.py file.

## Reproduce the experiments

### Data preparation

For data preparation execute all cells in one (based on selected dataset) of the Jupyter notebooks in 
CI_Experiments/preparation folder

### Experiments

For experiments reproduction execute all cell in one (based on selected dataset) of the jupyter notebooks in
CI_Experiments/experiments folder.

### Results

Results of the experiment can be found in the CI_Experiments/experiments/results folder.

## Repository structure

* CI_Experiments - root folder.
* data:
    * other_data - prepared data from other redundant experiments.
    * prepared_process_logs - data prepared for the experiments.
    * unprepared_process_logs - unprepared process logs used for the experiment.
* experiments - the folder of Jupyter notebooks with Causal Inference experiments.
    * results/activity_time: - each dataset results. In every data there is:
        * discovery_result.csv - result of Causal Discovery algorithms.
        * 2/discovery_result.csv - results of second Causal Discovery tryout with shuffled variables.  
        * estimation_result.csv - result of Causal Estimation algorithms.
        * qini_compare.png - Uplift modeling treatment and models comparison graph.
        * qini_subplots.png - every Uplift model treatment and model qini curve graph.
        * qinis.csv - results of qini curve scores. We draw Qini curves based on these values. 
        * test.csv - test data received from Data Preparation step in the experiment.
        * train.csv - train data received from Data Preparation step in the experiment.
      (note: results/BPIC2017 has numeric/estimation.txt file for the results from the results with Bozorgi's et al. study:
          https://arxiv.org/abs/2009.01561).
* other_tryouts - other redundant tryouts and their versions that were done initially.
* pipeline - the source code for the Data preparation and Causal Inference steps:
    * causal_discovery.py - code for Causal Discovery step.
    * causal_estimation.py - code for Causal Estimation step and Uplift modeling.
    * pipeline.py - end-to-end class that has all needed methods.
    * preparation.py - code for Data Preparation step.
* preparation - Jupyter notebooks for the initial data preparation (Case ID, Activity, Timestamp) for the event log 
  configuration for the experiments.
* tests/encoding_test.py - contains unit tests for encoding method.
* utilities/qinis_graphs_painter.ipynb - has qinis curve draw tool which could be more configurable to have better graphs.
* config.py - file that has the path to root folder: CI_Experiments. 

    
    
          

        