import pandas as pd
import numpy as np

from CI_Experiments.pipeline.causal_discovery import CausalDiscovery
from CI_Experiments.pipeline.causal_estimation import CausalEstimation
from CI_Experiments.pipeline.preparation import Preparation

ESTIMATION_ALGORITHMS = [
    'slearner',
    'tlearner',
    'xlearner',
    'tree',
    'grf',
    'bound',
    'iv',
    'div',
    'dr',
    'dml'
]

DISCOVERY_ALGORITHMS = {
    'ylearn': ['notears'],
    'gcastle': [
        'ANMNonlinear',
        'GES',
        'DirectLiNGAM',
        'ICALiNGAM',
        'PC',
        'Notears',
        'DAG_GNN',
        'RL',
        'CORL',
        'NotearsNonlinear',
        'GOLEM',
        'MCSL',
        'GAE'
    ],
    'pgm': [
        'ExhaustiveSearch',
        'HillClimbSearch',
        'TreeSearch',
        'MmhcEstimator',
        'PC'
    ]
}


class Experiment:
    """
    A class used to represent to perform experiments.
    ...

    Attributes
    ----------
    preparation : Preparation
        an object that stores the state and methods for data preparation (encoding-aggregation) step.
    discovery : CausalDiscovery
        an object that stores the state and methods for Causal Discovery step.
    estimation : CausalEstimation
        an object that stores the state and methods for Causal Estimation step.
    train_data : pd.DataFrame
        a reference to prepared (in the Preparation step) train data DataFrame.
    test_data : pd.DataFrame
        a reference to prepared (in the Preparation step) test data DataFrame.
    discovery_result : pd.DataFrame
        a reference to Causal Discovery step results.
    estimation_result: tuple[pd.DataFrame, pd.DataFrame]
        a reference to Causal Estimation and Uplift modeling step results.
        estimation_result[0] - Causal Estimation step results.
        estimation_result[1] - Uplift modeling Qini scores.
    Methods
    -------
    prepare_data(data, save_path=None, encode_type='onehottime', test_size=0.2, random_state=42)
        Encodes and splits data for the experiments.
    causal_discovery(
        data, save_path=None, discrete=True, random_state=42, algorithms=DISCOVERY_ALGORITHMS,
        timeout=1800, shuffle=False)
        Performs Causal Discovery step.
    compare_causal_discovery_results(exp)
        Compares Causal Discovery results of two experiments by Consistency metric.
    causal_estimation(train_data, test_data, treatment, adjustment, covariate, instrument,
        treat=None, control=None, estimators_options={}, discrete_treatment=True, adjust_treatment=False,
        target_outcome=None, reverse_treatment=False, path=None, random_state=42, algorithms=ESTIMATION_ALGORITHMS,
        timeout=1800
        Performs Causal Estimation step.
    plot_qini_curves(path, qinis=None)
        Plots qini curves.
    """

    def __init__(self):
        self.preparation = Preparation()
        self.discovery = CausalDiscovery()
        self.estimation = CausalEstimation()
        self.train_data = None
        self.test_data = None
        self.discovery_result = None
        self.estimation_result = None

    def prepare_data(
            self,
            data: pd.DataFrame,
            save_path=None,
            encode_type='onehottime',
            test_size=0.2,
            random_state=42
    ):

        """
        Encodes and splits data for the experiments.

        Parameters
        ----------
        data : pd.DataFrame
            data to prepare. Data should have Case ID, Activity, Timestamp, Outcome columns
        save_path : str, optional
            path to the folder where to save train.csv and test.csv files, default is None.
        encode_type: str, optional
            encoding method type - currently only onehottime is evaluated, default is 'onehottime'.
        test_size: float, optional
            proportion of test part against the data parameter.
         random_state: int, optional
            random state.

        Returns
        -------
        dict[str, pd.DataFrame] - 'train' key stores train dataset, 'test' key stores test dataset.
        """
        prepared_data = self.preparation.prepare_data(data, save_path, encode_type, test_size, random_state)
        self.train_data = prepared_data['train']
        self.test_data = prepared_data['test']
        return prepared_data

    def causal_discovery(
            self,
            data: pd.DataFrame,
            save_path=None,
            discrete=True,  # discrete dataset
            random_state=42,
            algorithms=DISCOVERY_ALGORITHMS,
            timeout=1800,
            shuffle=False
    ):
        """
        Performs Causal Discovery step.

        Parameters
        ----------
        data : pd.DataFrame
            train data for the discovery algorithm. Should have Outcome variable.
        save_path : str, optional
            path to the folder where to save Causal Discovery results, default is None.
        discrete: bool, optional
            if the dataset is discrete, default is True.
        random_state: int, optional
            random state, default is 42.
        algorithms: dict, optional
            Ylearn Causal Discovery algorithms to use in the experiment, default DISCOVERY_ALGORITHMS.
        timeout: int, optional
            number of seconds to timeout one Causal Discovery algorithm, default is 1600.
        shuffle: bool, optional
            if True, then performs random shuffle of columns in data before the Causal Discovery, to check the
            consistency, default is False.

        Returns
        -------
        pd.DataFrame - every row the result of each Causal Discovery algorithm that successfully finished the job.
        Columns: identifier_name (algorithm identification name),treatment (discovered treatment variables names),
            adjustment (discovered adjustment variables names), covariate (discovered covariate variables names),
            instrument (discovered instrument variables names), duration (algorithm execution duration in seconds),
            causal_graph (discovered causal graph as adjacency matrix).
        """
        if data is None:
            data = self.train_data
        discovery_result = self.discovery.discover(
            data,
            save_path,
            discrete,
            random_state,
            algorithms,
            timeout,
            shuffle
        )
        self.discovery_result = discovery_result
        return discovery_result

    def compare_causal_discovery_results(self, exp):
        """
        Compares Causal Discovery results of two experiments by Consistency metric.

        Parameters
        ----------
        exp : Experiment
            another experiment object with performed Causal Discovery step.
        Returns
        -------
        pd.DataFrame - the joined dataframe between two Causal Discovery results. The join is done on
        identifier_name column. The consistency=1.0 if the treatment is equal in both experiments, and consistency=0.0
        if the treatment is not equal in both experiments.
        """
        result_1 = self.discovery_result
        result_2 = exp.discovery_result
        results = pd.merge(result_1, result_2, on='identifier_name', how='outer', suffixes=('_1', '_2'))
        results['Consistency'] = np.where(
            results.apply(lambda row: sorted(row['treatment_1']) == sorted(row['treatment_2']), axis=1), 1.0, 0.0)
        return results

    def causal_estimation(
            self,
            train_data,
            test_data,
            treatment,
            adjustment,
            covariate,
            instrument,
            treat=None,
            control=None,
            estimators_options={},
            discrete_treatment=True,
            adjust_treatment=False,
            target_outcome=None,
            reverse_treatment=False,
            path=None,
            random_state=42,
            algorithms=ESTIMATION_ALGORITHMS,
            timeout=1800
    ):
        """
        Performs Causal Discovery step.

        Parameters
        ----------
        train_data : pd.DataFrame
            train data for the estimation algorithm fit substep.
        test_data : pd.DataFrame
            test data for the estimation algorithm estimate substep.
        treatment: list[str]
            treatment variable(-s) name(-s). All treatment columns should be in the train_data and test_data.
        adjustment: list[str]
            adjustment variable(-s) name(-s). All adjustment columns should be in the train_data and test_data.
        covariate: list[str]
            covariate variable(-s) name(-s). All covariate columns should be in the train_data and test_data.
        instrument: list[str]
            instrument variable(-s) name(-s). All instrument columns should be in the train_data and test_data.
        treat: list, optional
            treat values for the treatment, optional - can be inferred from discrete treatment, default is None.
        control: lsit, optional
            control values for the treatment, optional - can be inferred from discrete treatment, default is None.
        estimators_options: dict, optional
            individual options for each estimator.
        discrete_treatment: bool, optional
            if True, then the treatment is discrete and Uplift modeling can be applied, default is True.
        adjust_treatment: bool, optional
            if True, then the continuous treatment is converted to discrete by the rule t > 0.0 => 1.0 and t == 0.0 => 0.0
            Uplift modeling can be applied, default is True.
        target_outcome: str, optional
            the label of target outcome for the Uplift modeling, default is None (inferred from the data).
        reverse_treatment: bool, optional
            if True, then binary treatment is reversed 1.0 => 0.0, 0.0 => 1.0, default is False.
        path: str, optional
            path to save the estimation_result.csv
        random_state: int, optional
            random state,
        algorithms: dict, optional
            Ylearn Causal Estimation algorithms to use in the experiment, default is ESTIMATION_ALGORITHMS.
        timeout: int, optional
            number of seconds to timeout one Causal Discovery algorithm, default is 1600.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]:
        [0] - Causal Estimation step results.
        [1] - Uplift modeling Qini scores.
        """
        if train_data is None:
            train_data = self.train_data
        if test_data is None:
            test_data = self.test_data

        estimation_result = self.estimation.estimate_and_uplift(
            train_data=train_data,
            test_data=test_data,
            treatment=treatment,
            adjustment=adjustment,
            covariate=covariate,
            instrument=instrument,
            treat=treat,
            control=control,
            estimators_options=estimators_options,
            discrete_treatment=discrete_treatment,
            adjust_treatment=adjust_treatment,
            target_outcome=target_outcome,
            reverse_treatment=reverse_treatment,
            path=path,
            random_state=random_state,
            algorithms=algorithms,
            timeout=timeout
        )
        self.estimation_result = estimation_result
        return estimation_result

    def plot_qini_curves(self, path, qinis=None):
        """
        Plots qini curves.
        Parameters
        ----------
        path: str
            path to save the qini_subplots.png and qini_compare.png files.
        qinis: pd.DataFrame, optional
            qinis scores to draw the curves. If None, then it tries to draw qini curves from the inner result,
            default is None

        Returns
        -------
        None
        """
        self.estimation.plot_qini_curves(path, qinis)
