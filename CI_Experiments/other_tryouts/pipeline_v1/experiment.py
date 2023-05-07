"""

Pipeline (v1). Redundant.

"""


import pandas as pd
import os
import CI_Experiments.other_tryouts.pipeline_v1.constants as C
from datetime import datetime
from CI_Experiments.other_tryouts.pipeline_v1.datasets import ApproachDataset, Dataset
from CI_Experiments.other_tryouts.pipeline_v1.params import DiscoverParams, EstimateParams

# EXPERIMENT:


class Experiment:
    def _make_results_dir(self, path_to_results_dir: str, dataset_name: str):
        # experiment result dir
        now_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path_to_result_dir = f'{path_to_results_dir}/{dataset_name}/{now_string}'
        os.mkdir(path_to_result_dir)

        # approaches other_results dirs
        self.path_to_result_dir = path_to_result_dir
        os.mkdir(f'{path_to_result_dir}/{C.ACTIVITIES}')
        os.mkdir(f'{path_to_result_dir}/{C.ACTIVITIES_POS_NEG}')
        os.mkdir(f'{path_to_result_dir}/{C.ACTIVITIES_NUMERIC}')
        os.mkdir(f'{path_to_result_dir}/{C.ACTIVITIES_NUMERIC_POS_NEG}')

    def _get_true_graph(self, path_to_dataset: str):
        path_to_true_graph = f'{path_to_dataset}/{C.TRUE_GRAPH}.{C.CSV}'
        try:
            return pd.read_csv(path_to_true_graph)
        except OSError as e:
            print(f"True graph file in path {path_to_true_graph} was not found: {e}")
            return None

    def _init_approach_dataset(self, path_to_dataset: str, approach: str) -> ApproachDataset:
        path_to_approach_dataset = f'{path_to_dataset}/{approach}'
        train_data = pd.read_csv(f'{path_to_approach_dataset}/activity/{C.TRAIN}.{C.CSV}')
        test_data = pd.read_csv(f'{path_to_approach_dataset}/activity/{C.TEST}.{C.CSV}')
        return ApproachDataset(approach, train_data, test_data)

    def _init_dataset(self, path_to_datasets_dir: str, dataset_name: str):
        path_to_dataset = f'{path_to_datasets_dir}/{dataset_name}'
        true_graph = self._get_true_graph(path_to_dataset)

        activities_dataset = self._init_approach_dataset(path_to_dataset, C.ACTIVITIES)
        activities_pos_neg_dataset = activities_dataset.make_pos_neg()

        activities_numeric_dataset = self._init_approach_dataset(path_to_dataset, C.ACTIVITIES_NUMERIC)
        activities_numeric_pos_neg_dataset = activities_numeric_dataset.make_pos_neg()

        self.dataset = Dataset(
            dataset_name,
            activities_dataset,
            activities_pos_neg_dataset,
            activities_numeric_dataset,
            activities_numeric_pos_neg_dataset,
            true_graph
        )

    def __init__(
            self,
            dataset_name: str,
            path_to_datasets_dir: str,
            path_to_results_dir: str,
    ) -> None:
        self._make_results_dir(path_to_results_dir, dataset_name)
        self._init_dataset(path_to_datasets_dir, dataset_name)

    def discover(self, discover_params: DiscoverParams):
        discover_params.path_to_result_dir = self.path_to_result_dir
        self.dataset.discover(discover_params)

    def estimate(self, estimate_params: EstimateParams):
        estimate_params.path_to_result_dir = self.path_to_result_dir
        self.dataset.estimate(estimate_params)
