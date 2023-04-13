from typing import List
from ylearn import uplift as L
import matplotlib.pyplot as plt
import constants as C
import pandas as pd
import networkx as nx

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
        fig.savefig(f'{path_to_approach_dir}/{C.QINI}.{C.PNG}')
        self.qini.to_csv(f'{path_to_approach_dir}/{C.QINI}.{C.CSV}')
        if self.qini_top_point is not None:
            self.qini_top_point.to_csv(f'{path_to_approach_dir}/{C.QINI_TOP_POINT}.{C.CSV}')
        if self.qini_score is not None:
            self.qini_score.to_csv(f'{path_to_approach_dir}/{C.QINI_SCORE}.{C.CSV}')

        # gain
        fig = plt.figure(3)
        self.uplift_model.plot_gain()
        fig.savefig(f'{path_to_approach_dir}/{C.GAIN}.{C.PNG}')
        self.gain.to_csv(f'{path_to_approach_dir}/{C.GAIN}.{C.CSV}')
        if self.gain_top_point is not None:
            self.gain_top_point.to_csv(f'{path_to_approach_dir}/{C.GAIN_TOP_POINT}.{C.CSV}')

        # cumlift
        fig = plt.figure(4)
        self.uplift_model.plot_cumlift()
        fig.savefig(f'{path_to_approach_dir}/{C.CUMLIFT}.{C.PNG}')
        self.cumlift.to_csv(f'{path_to_approach_dir}/{C.CUMLIFT}.{C.CSV}')

        # auuc score
        if self.auuc_score is not None:
            self.auuc_score.to_csv(f'{path_to_approach_dir}/{C.AUUC_SCORE}.{C.CSV}')


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
        path_to_graph = f'{path_to_approach_dir}/{C.CAUSAL_GRAPH}.{C.CSV}'
        self.graph.to_csv(path_to_graph)

        # graph image
        G = nx.from_pandas_adjacency(self.graph, create_using=nx.DiGraph)
        nodes_legend_table = self._formulate_headers_as_table_row(headers)
        fig = plt.figure(0, figsize=(12,12))
        nx.draw_networkx(G, pos=nx.nx_agraph.graphviz_layout(G, C.NEATO), arrows=True, with_labels=True)
        plt.table(
            nodes_legend_table,
            cellLoc=C.GRAPH_TABLE_CELL_LOC,
            colColours=C.GRAPH_TABLE_HEADERS_COLORS,
            colWidths=C.GRAPH_TABLE_COL_WIDTHS,
            colLabels=C.GRAPH_TABLE_HEADERS
            )
        path_to_graph_image = f'{path_to_approach_dir}/{C.CAUSAL_GRAPH}.{C.PNG}'
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
        self.ate.to_csv(f'{path_to_approach_dir}/{C.ATE}.{C.CSV}')
        self.ite.to_csv(f'{path_to_approach_dir}/{C.ITE}.{C.CSV}')
        self.uplift_results.save(path_to_approach_dir)


class ApproachResults:
    def __init__(
            self,
            discover_results: DiscoverResults,
            estimate_results: EstimateResults
        ):
        self.discover_results = discover_results
        self.estimate_results = estimate_results
