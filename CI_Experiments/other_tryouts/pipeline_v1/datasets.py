"""

Pipeline (v1). Redundant.

"""


import pandas as pd
import networkx as nx
import CI_Experiments.other_tryouts.pipeline_v1.constants as C
import matplotlib.pyplot as plt
from CI_Experiments.other_tryouts.preparation_v1.preparation import replace_outcome_with_pos_neg
from cdt.metrics import precision_recall, SHD, SID
from ylearn import Why
from ylearn.causal_discovery import CausalDiscovery
from CI_Experiments.other_tryouts.pipeline_v1.results import ApproachResults, DiscoverResults, EstimateResults, \
    UpliftResults
from CI_Experiments.other_tryouts.pipeline_v1.params import DiscoverParams, ApproachEstimateParams, EstimateParams


# DATASETS:

class ApproachDataset:
    def __init__(
            self,
            approach: str,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
    ):
        self.approach = approach
        self.headers = train_data.columns
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        self.results = ApproachResults(
            discover_results=DiscoverResults(),
            estimate_results=EstimateResults()
        )
        self.outcome = None
        self.treatment = None
        self.why = None
        self.policy_interpreter = None
        self.uplift_model = None

    def make_pos_neg(self):
        approach = f'{self.approach}_pos_neg'
        train_data = self.train_data.copy()
        train_data = replace_outcome_with_pos_neg(train_data)
        test_data = self.test_data.copy()
        test_data = replace_outcome_with_pos_neg(test_data)
        return ApproachDataset(approach, train_data, test_data)

    def _prepare_headers(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        cols = data.columns
        data.columns = [str(i) for i in range(len(cols))]
        return data

    def _save_discover_results(self, discover_params: DiscoverParams):
        path_to_approach_dir = f'{discover_params.path_to_result_dir}/{self.approach}'
        self.results.discover_results.save(self.headers, path_to_approach_dir, discover_params.print_results)

    def discover(self, discover_params: DiscoverParams, true_graph: pd.DataFrame):
        cd = CausalDiscovery(hidden_layer_dim=discover_params.hidden_layer_dim)
        train_data = self._prepare_headers(self.train_data)
        graph = cd(train_data, threshold=discover_params.treshold)
        self.results.discover_results.graph = graph

        if discover_params.compare_with_true_graph:
            self._compare_with_true_graph(true_graph)

        self._save_discover_results(discover_params)

    def _save_txt(self, path_to_approach_dir: str):
        metrics_path = f'{path_to_approach_dir}/{C.SUMMARY}.{C.TXT}'

        with open(metrics_path, 'w') as f:
            f.write('METRICS:\n')
            f.write('********************************************************************\n')
            f.write(f'rloss: {self.results.estimate_results.rloss}\n')
            f.write(f"ate: {self.results.estimate_results.ate['mean']}\n")
            f.write('ESTIMATION:\n')
            f.write('********************************************************************\n')
            f.write(f'Outcome: {self.outcome}\n')
            f.write(f'Treatment: {", ".join(self.treatment)}\n')

    def save_estimation_results(self, path_to_result: str):
        path_to_approach_dir = f'{path_to_result}/{self.approach}'

        # policy tree
        fig = plt.figure(self.approach + "_graph", figsize=(40, 40), dpi=216)
        self.policy_interpreter.plot()
        fig.savefig(f'{path_to_approach_dir}/{C.TREE_MODEL}.{C.PNG}')

        # other_data estimation other_results
        self.results.estimate_results.uplift_results.id = self.approach
        self.results.estimate_results.save(path_to_approach_dir)
        self._save_txt(path_to_approach_dir)

    def estimate(self, params: ApproachEstimateParams):
        self.outcome = params.outcome
        self.treatment = params.treatment
        self.why = Why()
        self.why.fit(self.train_data, outcome=self.outcome, treatment=self.treatment)
        self.treatment = self.why.treatment_

        # ate
        ate = self.why.causal_effect(test_data=self.test_data, return_detail=True)
        self.results.estimate_results.ate = ate

        # ite
        ite = self.why.individual_causal_effect(test_data=self.test_data)
        self.results.estimate_results.ite = ite

        # rloss
        rloss = self.why.score(test_data=self.test_data, target_outcome=self.outcome, scorer='rloss')
        self.results.estimate_results.rloss = rloss

        # policy interpreter
        self.policy_interpreter = self.why.policy_interpreter(test_data=self.test_data)

        # uplift model
        self.uplift_model = self.why.uplift_model(test_data=self.test_data, treatment=self.treatment)
        self.results.estimate_results.uplift_results = UpliftResults(self.uplift_model)

    def _compare_with_true_graph(self, true_g: pd.DataFrame):
        # TODO: improve this
        def filter_node(node):
            return node.replace('Activity_', '') in true_g.columns

        graph = self.results.discover_results.graph
        discovered_g = graph.copy()
        discovered_g.columns = self.headers

        T = nx.from_pandas_adjacency(true_g, create_using=nx.DiGraph)
        D = nx.from_pandas_adjacency(discovered_g, create_using=nx.DiGraph)
        Ds = nx.subgraph_view(D, filter_node=filter_node)
        T_ = nx.DiGraph()
        T_.add_nodes_from(sorted(T.nodes(data=True)))
        T_.add_edges_from(T.edges(data=True))
        D_ = nx.DiGraph()
        D_.add_nodes_from(sorted(Ds.nodes(data=True)))
        D_.add_edges_from(Ds.edges(data=True))
        aupr, curve = precision_recall(T_, D_)
        shd = SHD(T_, D_, double_for_anticausal=False)
        sid = SID(T_, D_)
        discover_metrics = {'aupr': aupr, 'curve': curve, 'shd': shd, 'sid': sid}
        self.results.discover_results.metrics = discover_metrics


class Dataset:
    def __init__(
            self,
            name: str,
            activities_dataset: ApproachDataset = None,
            activities_pos_neg_dataset: ApproachDataset = None,
            activities_numeric_dataset: ApproachDataset = None,
            activities_numeric_pos_neg_dataset: ApproachDataset = None,
            true_graph: pd.DataFrame = None,
    ):
        self.name = name
        self.activities_dataset = activities_dataset
        self.activities_pos_neg_dataset = activities_pos_neg_dataset
        self.activities_numeric_dataset = activities_numeric_dataset
        self.activities_numeric_pos_neg_dataset = activities_numeric_pos_neg_dataset
        self.true_graph = true_graph

    def discover(self, discover_params: DiscoverParams):
        self.activities_dataset.discover(discover_params, self.true_graph)
        self.activities_pos_neg_dataset.discover(discover_params, self.true_graph)
        self.activities_numeric_dataset.discover(discover_params, self.true_graph)
        self.activities_numeric_pos_neg_dataset.discover(discover_params, self.true_graph)

    def _save_estimation_results(self, path_to_result_dir: str):
        self.activities_dataset.save_estimation_results(path_to_result_dir)
        self.activities_pos_neg_dataset.save_estimation_results(path_to_result_dir)
        self.activities_numeric_dataset.save_estimation_results(path_to_result_dir)
        self.activities_numeric_pos_neg_dataset.save_estimation_results(path_to_result_dir)

    def estimate(self, params: EstimateParams):
        self.activities_dataset.estimate(params.activities_params)
        self.activities_pos_neg_dataset.estimate(params.activities_pos_neg_params)
        self.activities_numeric_dataset.estimate(params.activities_numeric_params)
        self.activities_numeric_pos_neg_dataset.estimate(params.activities_numeric_pos_neg_params)
        self._save_estimation_results(params.path_to_result_dir)
