from typing import List
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from ylearn import Why, uplift as L
from ylearn.causal_discovery import CausalDiscovery
import os
from datetime import datetime
from preparation_code.preparation import replace_outcome_with_pos_neg
from cdt.metrics import precision_recall, SHD, SID

#--------------------------------------------------------------------------
# CONSTANTS:

ACTIVITIES = 'activities'
ACTIVITIES_NUMERIC = 'activities_numeric'
ACTIVITIES_NUMERIC_POS_NEG = 'activities_numeric_pos_neg'
ACTIVITIES_POS_NEG = 'activities_pos_neg'
ATE = 'ate'
AUUC_SCORE = 'auuc_score'
CAUSAL_GRAPH = 'causal_graph'
CSV = 'csv'
CUMLIFT = 'cumlift'
EFFECTS = 'effects'
GAIN = 'gain'
GAIN_TOP_POINT = 'gain_top_point'
GRAPH = 'graph'
GRAPH_TABLE_CELL_LOC = 'left'
GRAPH_TABLE_COL_WIDTHS = [0.2, 0.8]
GRAPH_TABLE_HEADERS = ['Node nr', 'Name']
GRAPH_TABLE_HEADERS_COLORS = ['y', 'y']
ITE = 'ite'
METRICS = 'metrics'
NEATO = 'neato'
PNG = 'png'
QINI = 'qini'
QINI_SCORE = 'qini_score'
QINI_TOP_POINT = 'qini_top_point'
TEST = 'test'
TRAIN = 'train'
TREE_MODEL = 'tree_model'
TRUE_GRAPH = 'true_graph'
TXT = 'txt'

#--------------------------------------------------------------------------
# PARAMS:

class DiscoverParams:
    def __init__(
            self, 
            print_results: bool = False, 
            hidden_layer_dim: List[int] = [3], 
            treshold: float = 0.01, 
            compare_with_true_graph: bool = False,
            path_to_result_dir: str = None
        ):
        self.print_results = print_results,
        self.hidden_layer_dim = hidden_layer_dim
        self.treshold = treshold
        self.compare_with_true_graph = compare_with_true_graph
        self.path_to_result_dir = path_to_result_dir


class ApproachEstimateParams:
    def __init__(
            self,
            outcome: str,
            treatment: List[str] = None
    ):
        self.outcome = outcome
        self.treatment = treatment


class EstimateParams:
    def __init__(
            self, 
            activities_params: ApproachEstimateParams,
            activities_pos_neg_params: ApproachEstimateParams,
            activities_numeric_params: ApproachEstimateParams,
            activities_numeric_pos_neg_params: ApproachEstimateParams,
            path_to_result_dir: str = None
        ):
        self.activities_params = activities_params
        self.activities_pos_neg_params = activities_pos_neg_params
        self.activities_numeric_params = activities_numeric_params
        self.activities_numeric_pos_neg_params = activities_numeric_pos_neg_params
        self.path_to_result_dir = path_to_result_dir

#--------------------------------------------------------------------------
# RESULTS:

class UpliftResults:
    def __init__(
            self,
            uplift_model: L.UpliftModel
        ):
        self.uplift_model = uplift_model
        self.cumlift = uplift_model.get_cumlift()
        self.gain = uplift_model.get_gain()
        self.qini = uplift_model.get_qini()
        self.gain_top_point = uplift_model.gain_top_point()
        self.qini_top_point = uplift_model.qini_top_point()
        self.auuc_score = uplift_model.auuc_score()
        self.qini_score = uplift_model.qini_score()


    def save(self, path_to_approach_dir: str):
        # qini
        fig = plt.figure(2)
        self.uplift_model.plot_qini()
        fig.savefig(f'{path_to_approach_dir}/{QINI}.{PNG}')
        self.qini.to_csv(f'{path_to_approach_dir}/{QINI}.{CSV}')
        if self.qini_top_point is not None:
            self.qini_top_point.to_csv(f'{path_to_approach_dir}/{QINI_TOP_POINT}.{CSV}')
        if self.qini_score is not None:
            self.qini_score.to_csv(f'{path_to_approach_dir}/{QINI_SCORE}.{CSV}')

        # gain
        fig = plt.figure(3)
        self.uplift_model.plot_gain()
        fig.savefig(f'{path_to_approach_dir}/{GAIN}.{PNG}')
        self.gain.to_csv(f'{path_to_approach_dir}/{GAIN}.{CSV}')
        if self.gain_top_point is not None:
            self.gain_top_point.to_csv(f'{path_to_approach_dir}/{GAIN_TOP_POINT}.{CSV}')

        # cumlift
        fig = plt.figure(4)
        self.uplift_model.plot_cumlift()
        fig.savefig(f'{path_to_approach_dir}/{CUMLIFT}.{PNG}')
        self.cumlift.to_csv(f'{path_to_approach_dir}/{CUMLIFT}.{CSV}')

        # auuc score
        if self.auuc_score is not None:
            self.auuc_score.to_csv(f'{path_to_approach_dir}/{AUUC_SCORE}.{CSV}')


class DiscoverResults:
    def __init__(
            self,
            graph: pd.DataFrame = None,
            metrics: dict = None
        ):
        self.graph = graph
        self.metrics = metrics


    def _formulate_headers_as_table_row(self, headers: List[str]):
        return list(map(lambda i_header: [str(i_header[0]), i_header[1].strip()], enumerate(headers)))


    def save(self, headers: List[str], path_to_approach_dir: str, print: bool=False):
        # graph
        path_to_graph = f'{path_to_approach_dir}/{CAUSAL_GRAPH}.{CSV}'
        self.graph.to_csv(path_to_graph)

        # graph image
        G = nx.from_pandas_adjacency(self.graph, create_using=nx.DiGraph)
        nodes_legend_table = self._formulate_headers_as_table_row(headers)
        fig = plt.figure(0, figsize=(12,12))
        nx.draw_networkx(G, pos=nx.nx_agraph.graphviz_layout(G, NEATO), arrows=True, with_labels=True)
        plt.table(
            nodes_legend_table,
            cellLoc=GRAPH_TABLE_CELL_LOC,
            colColours=GRAPH_TABLE_HEADERS_COLORS,
            colWidths=GRAPH_TABLE_COL_WIDTHS,
            colLabels=GRAPH_TABLE_HEADERS
            )
        path_to_graph_image = f'{path_to_approach_dir}/{CAUSAL_GRAPH}.{PNG}'
        fig.savefig(path_to_graph_image)
        if print:
            plt.show()


class EstimateResults:
    def __init__(
            self,
            ate: pd.DataFrame = None,
            ite: pd.DataFrame = None,
            rloss: float = None,
            uplift_results: UpliftResults = None
        ):
        self.ate = ate
        self.ite = ite
        self.rloss = rloss
        self.uplift_results = uplift_results


    def save(self, path_to_approach_dir: str):
        self.ate.to_csv(f'{path_to_approach_dir}/{ATE}.{CSV}')
        self.ite.to_csv(f'{path_to_approach_dir}/{ITE}.{CSV}')
        self.uplift_results.save(path_to_approach_dir)


class ApproachResults:
    def __init__(
            self,
            discover_results: DiscoverResults,
            estimate_results: EstimateResults
        ):
        self.discover_results = discover_results
        self.estimate_results = estimate_results

#--------------------------------------------------------------------------
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


    def _check_treatment(self):
        if self.treatment is None:
            columns = self.headers
            columns.remove(self.outcome)
            self.treatment = columns


    def _save_txt(self, path_to_approach_dir: str):
        metrics_path = f'{path_to_approach_dir}/{METRICS}.{TXT}'

        with open(metrics_path, 'w') as f:
            f.write('METRICS:/n')
            f.write('********************************************************************/n')
            f.write(f'rloss: {self.results.estimate_results.rloss}/n')
            f.write('ESTIMATION:/n')
            f.write('********************************************************************/n')
            f.write(f'Outcome: {self.outcome}/n')
            f.write(f'Treatment: {", ".join(self.treatment)}/n')

    
    def save_estimation_results(self, path_to_result: str):
        path_to_approach_dir = f'{path_to_result}/{self.approach}'

        # policy tree
        fig = plt.figure(1, figsize=(12, 12))
        self.policy_interpreter.plot()
        fig.savefig(f'{path_to_approach_dir}/{TREE_MODEL}.{PNG}')

        # other estimation results
        self.results.estimate_results.save(path_to_approach_dir)
        self._save_txt(path_to_approach_dir)


    def estimate(self, params: ApproachEstimateParams):
        self.outcome = params.outcome
        self.treatment = params.treatment
        self._check_treatment()
        self.why = Why()
        self.why.fit(self.train_data, outcome=self.outcome, treatment=self.treatment)
        
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
        self.policy_interpreter = self.why.policy_interpreter(test_data=self.test_data, target_outcome=self.outcome)
        
        # uplift model
        self.uplift_model = self.why.uplift_model(test_data=self.test_data, treatment=self.treatment, target_outcome=self.outcome)
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
            activities_dataset: ApproachDataset=None,
            activities_pos_neg_dataset: ApproachDataset=None,
            activities_numeric_dataset: ApproachDataset=None,
            activities_numeric_pos_neg_dataset: ApproachDataset=None,
            true_graph: pd.DataFrame=None,  
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
        self.activities_numeric_pos_neg_dataset.estimate(params.activities_pos_neg_params)
        self.activities_numeric_dataset.estimate(params.activities_numeric_params)
        self.activities_numeric_pos_neg_dataset.estimate(params.activities_numeric_pos_neg_params)
        self._save_estimation_results(params.path_to_result_dir)
            
#--------------------------------------------------------------------------
# EXPERIMENT:

class Experiment:
    def _make_results_dir(self, path_to_results_dir: str, dataset_name: str):
        # exepriment result dir
        now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        path_to_result_dir = f'{path_to_results_dir}/{dataset_name}/{now_string}'
        os.mkdir(path_to_result_dir)

        # approaches results dirs
        self.path_to_result_dir = path_to_result_dir
        os.mkdir(f'{path_to_result_dir}/{ACTIVITIES}')
        os.mkdir(f'{path_to_result_dir}/{ACTIVITIES_POS_NEG}')
        os.mkdir(f'{path_to_result_dir}/{ACTIVITIES_NUMERIC}')
        os.mkdir(f'{path_to_result_dir}/{ACTIVITIES_NUMERIC_POS_NEG}')

    
    def _get_true_graph(self, path_to_dataset: str):
        path_to_true_graph = f'{path_to_dataset}/{TRUE_GRAPH}.{CSV}'
        try:
            return pd.read_csv(path_to_true_graph)
        except OSError as e:
            print(f"True graph file in path {path_to_true_graph} was not found: {e}")
            return None
    
    def _init_approach_dataset(self, path_to_dataset: str, approach: str) -> ApproachDataset:
        path_to_approach_dataset = f'{path_to_dataset}/{approach}'
        train_data = pd.read_csv(f'{path_to_approach_dataset}/{TRAIN}.{CSV}')
        test_data = pd.read_csv(f'{path_to_approach_dataset}/{TEST}.{CSV}')
        return ApproachDataset(approach, train_data, test_data)

    
    def _init_dataset(self, path_to_datasets_dir: str, dataset_name: str):
        path_to_dataset = f'{path_to_datasets_dir}/{dataset_name}'
        true_graph = self._get_true_graph(path_to_dataset)

        activities_dataset = self._init_approach_dataset(path_to_dataset, ACTIVITIES)
        activities_pos_neg_dataset = activities_dataset.make_pos_neg()

        activities_numeric_dataset = self._init_approach_dataset(path_to_dataset, ACTIVITIES_NUMERIC)
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
        self.dataset.discover(discover_params)
        
    
    def estimate(self, estimate_params: EstimateParams):
        self.dataset.estimate(estimate_params)
