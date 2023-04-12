from typing import List
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from ylearn import Why
from ylearn.causal_discovery import CausalDiscovery
import os
from datetime import datetime
from preparation_code.preparation import replace_outcome_with_pos_neg
from cdt.metrics import precision_recall, SHD, SID

# Constants:

DATASET_FILE_NAME = 'train'
CSV = 'csv'
TXT = 'txt'
PNG = 'png'
NEATO = 'neato'

GRAPH_TABLE_CELL_LOC = 'left'
GRAPH_TABLE_HEADERS_COLORS = ['y', 'y']
GRAPH_TABLE_COL_WIDTHS = [0.2, 0.8]
GRAPH_TABLE_HEADERS = ['Node nr', 'Name']
GRAPH = 'graph'
EFFECTS = 'effects'
SUMMARY = 'summary'
ACTIVITIES = 'activities'
ACTIVITIES_POS_NEG = 'activities_pos_neg'
ACTIVITIES_NUMERIC = 'activities_numeric'
ACTIVITIES_NUMERIC_POS_NEG = 'activities_numeric_pos_neg'
TRUE_GRAPH = 'true_graph'



class Experiment:
    
    def _init_empty(self):
        self.path_to_graph = '-'
        self.path_to_graph_image = '-'
        self.path_to_effects = '-'
        self.outcome = '-'
        self.treatment = '-'


    def _make_results_dir(self, path_to_results_dir: str):
        result_dir_path = f'{path_to_results_dir}/{self.dataset_name}/{self.now_string}'
        os.mkdir(result_dir_path)
        self.path_to_result_dir = result_dir_path

    
    def _init_true_graph(self, path_to_datasets_dir: str):
        self.path_to_true_graph = f'{path_to_datasets_dir}/{self.dataset_name}/{TRUE_GRAPH}.{CSV}'
        self.true_graph_adj = pd.read_csv(self.path_to_true_graph)

    
    def _init_datasets(self, path_to_datasets_dir: str):
        self.path_to_activities_dataset = f'{path_to_datasets_dir}/{self.dataset_name}/{ACTIVITIES}/{DATASET_FILE_NAME}.{CSV}'
        self.activities_dataset = pd.read_csv(self.path_to_activities_dataset)

        self.activities_pos_neg_dataset = replace_outcome_with_pos_neg(self.activities_dataset)

        self.path_to_activities_numeric_dataset = f'{path_to_datasets_dir}/{self.dataset_name}/{ACTIVITIES_NUMERIC}/{DATASET_FILE_NAME}.{CSV}'
        self.activities_numeric_dataset = pd.read_csv(self.path_to_activities_numeric_dataset)

        self.activities_numeric_pos_neg_dataset = replace_outcome_with_pos_neg(self.activities_numeric_dataset)


    def _prepare_headers(self):
        self.activities_headers = self.activities_dataset.columns
        self.activities_dataset.columns = [str(i) for i in range(len(self.activities_dataset.columns))]
        
        self.activities_pos_neg_headers = self.activities_pos_neg_dataset.columns
        self.activities_pos_neg_dataset.columns = [str(i) for i in range(len(self.activities_pos_neg_dataset.columns))]

        self.activities_numeric_headers = self.activities_numeric_dataset.columns
        self.activities_numeric_dataset.columns = [str(i) for i in range(len(self.activities_numeric_dataset.columns))]

        self.activities_numeric_pos_neg_headers = self.activities_numeric_pos_neg_dataset.columns
        self.activities_numeric_pos_neg_dataset.columns = [str(i) for i in range(len(self.activities_numeric_pos_neg_dataset.columns))]


    def __init__(self, dataset_name: str, path_to_datasets_dir: str, path_to_results_dir: str) -> None:
        self._init_empty()

        self.dataset_name = dataset_name
        self.now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self._make_results_dir(path_to_results_dir)

        self._init_true_graph(path_to_datasets_dir)

        self._init_datasets(path_to_datasets_dir)

        self._prepare_headers()
        

    def _formulate_headers_as_table_row(self, headers: List[str]):
        return list(map(lambda i_header: [str(i_header[0]), i_header[1].strip()], enumerate(headers)))


    def _compare_with_true_graph(self, est: pd.DataFrame, headers: List[str]) -> dict:
        def filter_node(node):
            return node.replace('Activity_', '') in self.true_graph_adj.columns
        discovered = est.copy()
        true = self.true_graph_adj
        discovered.columns = headers
        
        T = nx.from_pandas_adjacency(true, create_using=nx.DiGraph)
        D = nx.from_pandas_adjacency(discovered, create_using=nx.DiGraph)
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
        results_dict = {'aupr': aupr, 'curve': curve, 'shd': shd, 'sid': sid}
        return results_dict


    def _discover(self, dataset: pd.DataFrame, headers: List[str], graph_name: str, print_results=False, hidden_layer_dim=[3], treshold=0.01):
        cd = CausalDiscovery(hidden_layer_dim=hidden_layer_dim)
        est = cd(dataset, threshold=treshold)
        metrics_dict = self._compare_with_true_graph(est, headers)
        G = nx.from_pandas_adjacency(est, create_using=nx.DiGraph)
        nodes_legend_table = self._formulate_headers_as_table_row(headers)
        fig = plt.figure(0,figsize=(12,12))
        nx.draw_networkx(G, pos=nx.nx_agraph.graphviz_layout(G, NEATO), arrows=True, with_labels=True)
        plt.table(
            nodes_legend_table,
            cellLoc=GRAPH_TABLE_CELL_LOC,
            colColours=GRAPH_TABLE_HEADERS_COLORS,
            colWidths=GRAPH_TABLE_COL_WIDTHS,
            colLabels=GRAPH_TABLE_HEADERS
            )
        path_to_graph = f'{self.path_to_result_dir}/{graph_name}_{GRAPH}.{CSV}'
        est.to_csv(self.path_to_graph)
        path_to_graph_image = f'{self.path_to_result_dir}/{graph_name}_{GRAPH}.{PNG}'
        fig.savefig(path_to_graph_image)
        if print_results:
            plt.show()
        return est, path_to_graph, path_to_graph_image, metrics_dict


    def discover(self, print_results=False, hidden_layer_dim=[3], treshold=0.01):
        self.activities_est, self.path_to_activities_graph, self.path_to_activities_image, self.activities_graph_metrics = self._discover(
            self.activities_dataset, 
            self.activities_headers, 
            ACTIVITIES, 
            print_results, 
            hidden_layer_dim, 
            treshold)
        
        self.activities_pos_neg_est, self.path_to_activities_pos_neg_graph, self.path_to_activities_pos_neg_image, self.activities_pos_neg_graph_metrics = self._discover(
            self.activities_pos_neg_dataset, 
            self.activities_pos_neg_headers, 
            ACTIVITIES_POS_NEG, 
            print_results, 
            hidden_layer_dim, 
            treshold)
        
        self.activities_numeric_est, self.path_to_activities_numeric_graph, self.path_to_activities_numeric_image, self.activities_numeric_graph_metrics = self._discover(
            self.activities_numeric_dataset, 
            self.activities_numeric_headers, 
            ACTIVITIES_NUMERIC, 
            print_results, 
            hidden_layer_dim, 
            treshold)
        
        self.activities_numeric_pos_neg_est, self.path_to_activities_numeric_pos_neg_graph, self.path_to_activities_numeric_pos_neg_image, self.activities_numeric_pos_neg_graph_metrics = self._discover(
            self.activities_numeric_pos_neg_dataset, 
            self.activities_numeric_pos_neg_headers, 
            ACTIVITIES_NUMERIC_POS_NEG, 
            print_results, 
            hidden_layer_dim, 
            treshold)


    def _rename_columns(self) -> pd.DataFrame: 
        return self.dataset.rename(columns={col: str(col) for col in self.dataset.columns})

    
    def _check_treatment(self, columns, treatment):
        if treatment is None:
            columns.remove(self.outcome)
            return columns
        else:
            return treatment


    def estimate(self, outcome: str, treatment=None):
        self.outcome = outcome
        data = self._rename_columns()
        treatment = self._check_treatment(data.columns, treatment)
        self.treatment = treatment
        why = Why()
        why.fit(data, outcome=outcome, treatment=treatment)
        self.effects = why.causal_effect(return_detail=True)
        self.path_to_effects = f'{self.path_to_result_dir}/{EFFECTS}.{CSV}'
        self.effects.to_csv(self.path_to_effects)


    def save_summary(self):
        summary_path = f'{self.path_to_result_dir}/{SUMMARY}.{TXT}'

        with open(summary_path, 'w') as summary_file:
            summary_file.write('SUMMARY:/n')
            summary_file.write('********************************************************************/n')
            summary_file.write(f'Date: {self.now_string}/n')
            summary_file.write(f'Dataset name: {self.dataset_name}/n')
            summary_file.write(f'Dataset path: {self.path_to_dataset}/n')
            summary_file.write(f'Headers path:{self.path_to_headers}/n')
            summary_file.write('DISCOVERY:/n')
            summary_file.write('********************************************************************/n')
            summary_file.write(f'Graph path: {self.path_to_graph}/n')
            summary_file.write(f'Graph image path: {self.path_to_graph_image}/n')
            summary_file.write('ESTIMATION:/n')
            summary_file.write('********************************************************************/n')
            summary_file.write(f'Effects path: {self.path_to_effects}/n')
            summary_file.write(f'Outcome: {self.outcome}/n')
            summary_file.write(f'Treatment: {", ".join(self.treatment)}/n')

