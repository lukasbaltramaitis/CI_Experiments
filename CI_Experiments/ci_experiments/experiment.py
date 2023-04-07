import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from ylearn import Why
from ylearn.causal_discovery import CausalDiscovery
import os
from datetime import datetime

# Constants:

DATASET_FILE_NAME = 'log'
HEADERS_FILE_NAME = 'headers'

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



class Experiment:
    def __init__(self, dataset_name: str, path_to_datasets_dir: str, path_to_results_dir: str) -> None:
        self.path_to_graph = '-'
        self.path_to_graph_image = '-'
        self.path_to_effects = '-'
        self.outcome = '_'
        self.treatment = '_'


        self.dataset_name = dataset_name
        
        now_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.now_string = now_string

        result_dir_path = f'{path_to_results_dir}/{dataset_name}/{now_string}'
        os.mkdir(result_dir_path)
        self.path_to_result_dir = result_dir_path

        self.path_to_dataset = f'{path_to_datasets_dir}/{dataset_name}/{DATASET_FILE_NAME}.{CSV}'
        self.dataset = pd.read_csv(self.path_to_dataset, header=None)

        self.path_to_headers = f'{path_to_datasets_dir}/{dataset_name}/{HEADERS_FILE_NAME}.{TXT}'
        with open(self.path_to_headers, 'r') as headers_file:
            self.headers = headers_file.readlines()


    def _formulate_headers_as_table_row(self):
        return list(map(lambda i_header: [str(i_header[0]), i_header[1].strip()], enumerate(self.headers)))

    
    def discover(self, print_results=False, hidden_layer_dim=[3], treshold=0.01):
        cd = CausalDiscovery(hidden_layer_dim=hidden_layer_dim)
        self.est = cd(self.dataset, threshold=treshold)
        G = nx.from_pandas_adjacency(self.est, create_using=nx.DiGraph)
        nodes_legend_table = self._formulate_headers_as_table_row()
        fig = plt.figure(0,figsize=(12,12))
        nx.draw_networkx(G, pos=nx.nx_agraph.graphviz_layout(G, NEATO), arrows=True, with_labels=True)
        plt.table(
            nodes_legend_table,
            cellLoc=GRAPH_TABLE_CELL_LOC,
            colColours=GRAPH_TABLE_HEADERS_COLORS,
            colWidths=GRAPH_TABLE_COL_WIDTHS,
            colLabels=GRAPH_TABLE_HEADERS
            )
        self.path_to_graph = f'{self.path_to_result_dir}/{GRAPH}.{CSV}'
        self.est.to_csv(self.path_to_graph)
        self.path_to_graph_image = f'{self.path_to_result_dir}/{GRAPH}.{PNG}'
        fig.savefig(self.path_to_graph_image)
        if print_results:
            plt.show()


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

