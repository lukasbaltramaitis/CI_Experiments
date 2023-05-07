import logging
import pandas as pd
import networkx as nx
import random
from ylearn import Why
import matplotlib.pyplot as plt
import time as t
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

from CI_Experiments.config import PROJECT_DIR

DATASETS_PATH = PROJECT_DIR / 'data/prepared_process_logs'
RESULTS_PATH = PROJECT_DIR / 'other_results/activities_discovery_results'
OUTCOME = 'Outcome'
ALL_ALGORITHMS = {
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

WORKING_ALGORITHMS = {
    'ylearn': ['notears'],
    'gcastle': [
        'DirectLiNGAM',
        'ICALiNGAM',
        'PC',
        'Notears',
        'NotearsNonlinear',
        'GOLEM'
    ],
    'pgm': [
        'PC',
        'TreeSearch',
    ]
}

LOGICAL_ALGORITHMS = {
    'ylearn': ['notears'],
    'gcastle': [
        'ICALiNGAM',
        'PC'
    ],
    'pgm': [
        'HillClimbSearch',
        'TreeSearch',
    ]
}

GRAPH_TABLE_CELL_LOC = 'left'
GRAPH_TABLE_COL_WIDTHS = [0.2, 0.8]
GRAPH_TABLE_HEADERS = ['Node nr', 'Name']
GRAPH_TABLE_HEADERS_COLORS = ['y', 'y']


class CausalDiscovery:
    def __init__(self):
        self.random_state = None
        self.data = None
        self.result = None
        self.discrete = None
        self.timeout = None
        self.path = None
        self.draw_graph = None

    def _formulate_headers_as_table_row(self):
        return list(map(lambda i_header: [str(i_header[0]), i_header[1].strip()], enumerate(self.data.columns)))

    def _draw_graph(self, graph, identifier_name):
        G = nx.from_numpy_matrix(graph, create_using=nx.DiGraph)
        nodes_legend_table = self._formulate_headers_as_table_row()
        fig = plt.figure(identifier_name, figsize=(12, 16))
        nx.draw_networkx(G, arrows=True, with_labels=True, pos=nx.circular_layout(G))
        plt.table(
            nodes_legend_table,
            cellLoc=GRAPH_TABLE_CELL_LOC,
            colColours=GRAPH_TABLE_HEADERS_COLORS,
            colWidths=GRAPH_TABLE_COL_WIDTHS,
            colLabels=GRAPH_TABLE_HEADERS
        )
        path_to_graph_image = f'{self.path}/{identifier_name}_causal_graph.png'
        fig.savefig(path_to_graph_image)

    def _causal_graph(self, why: Why, identifier_name):
        try:
            cg = why.causal_graph()
            am = cg.to_adj_matrix()
            if am is not None and self.draw_graph:
                self._draw_graph(am, identifier_name)
            return am
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            print("Unexpected error in causal graph drawing, returning:", error)
            return None

    def _log(self, msg: str, identifier_name: str, duration: float):
        print(
            f'**************************\n{msg}\nidentifier: {identifier_name}\nduration: {duration}\n')

    def _create_identifier_name(self, identifier: str, identifier_options=None) -> str:
        if identifier_options is not None:
            learner_name = identifier_options.get('learner')
            return f'{identifier}_{learner_name}'
        else:
            return str(identifier)

    def _add_result(self, identifier_name, cd_result, causal_graph, duration):
        treatment = cd_result[0]
        adjustment = cd_result[1]
        covariate = cd_result[2]
        instrument = cd_result[3]
        data = {
            'identifier_name': [identifier_name],
            'treatment': [treatment],
            'adjustment': [adjustment],
            'covariate': [covariate],
            'instrument': [instrument],
            'duration': [duration],
            'causal_graph': [causal_graph]
        }
        row = pd.DataFrame(data, dtype=object)
        if self.result is not None:
            self.result = pd.concat([self.result, row], ignore_index=True)
        else:
            self.result = row

    def _identify(self, identifier, identifier_options):
        why = Why(identifier=identifier, identifier_options=identifier_options, random_state=self.random_state)
        data = self.data.copy()
        start = t.time()
        result = why.identify(data, OUTCOME)
        end = t.time()
        duration = end - start
        identifier_name = self._create_identifier_name(identifier, identifier_options)
        cg = self._causal_graph(why, identifier_name)
        self._log('Identify finished', identifier_name, duration)
        self._add_result(identifier_name, result, causal_graph=cg, duration=duration)

    def _identify_with_timeout(self, identifier, identifier_options):
        try:
            func_timeout(self.timeout, self._identify, args=(identifier, identifier_options))
        except FunctionTimedOut:
            identifier_name = self._create_identifier_name(identifier, identifier_options)
            print(f"{identifier_name} identify could not complete within {self.timeout} seconds and was terminated.\n")

    def _identify_with_error_except(self, identify_func, identifier, identifier_options):
        try:
            identify_func(identifier, identifier_options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logging.exception("Unexpected error!")

    def _init_result(self, size: int):
        data = {
            'identifier_name': np.full(size, np.nan),
            'treatment': np.full(size, np.nan),
            'adjustment': np.full(size, np.nan),
            'covariate': np.full(size, np.nan),
            'instrument': np.full(size, np.nan),
            'duration': np.full(size, np.nan),
            'causal_graph': np.full(size, np.nan)
        }
        self.result = pd.DataFrame(data, dtype=object)

    def _create_pc_identifier_options(self):
        ci_test = 'chi_square'
        if not self.discrete:
            ci_test = 'pearsonr'
        return {
            'learner': 'PC',
            'ci_test': ci_test,
            'max_cond_vars': 50
        }

    def _create_pgm_identifier_options(self, alg):
        if alg == 'PC':
            return self._create_pc_identifier_options()
        else:
            return {'learner': alg}

    def _discover(self, lib: str, alg: str):
        if lib == 'gcastle':
            identifier_options = {'learner': alg}
            self._identify_with_error_except(self._identify_with_timeout, lib, identifier_options)
        elif lib == 'pgm':
            identifier_options = self._create_pgm_identifier_options(alg)
            self._identify_with_error_except(self._identify_with_timeout, lib, identifier_options)
        else:
            self._identify_with_error_except(self._identify_with_timeout, alg, None)

    def _discover_algs(self, algorithms):
        for lib, algs in algorithms.items():
            for alg in algs:
                self._discover(lib, alg)

    def _save_result(self, path):
        if path is not None and self.result is not None:
            result_path = f"{path}/discovery_result.csv"
            self.result.to_csv(result_path, index=False)

    def _shuffle_variables(self, data, shuffle, random_state):
        if shuffle and data is not None:
            columns = list(data.columns)
            columns.remove(OUTCOME)
            outcome = data[OUTCOME]
            random.Random(random_state).shuffle(columns)
            data = data[columns]
            data[OUTCOME] = outcome
            return data
        else:
            return data

    def discover(
            self,
            data: pd.DataFrame,
            path=None,
            discrete=True,
            random_state=42,
            algorithms=LOGICAL_ALGORITHMS,
            timeout=1800,
            shuffle=False,
            draw_graph=False
    ):
        self.data = self._shuffle_variables(data, shuffle, random_state)
        self.discrete = discrete
        self.random_state = random_state
        self.timeout = timeout
        self.path = path
        self.draw_graph = draw_graph
        self._discover_algs(algorithms)
        self._save_result(path)
        return self.result

    def filter_results(self):
        if self.result is not None:
            cols = ['treatment', 'adjustment', 'covariate', 'instrument']
            result = self.result[cols]
            filtered_result = result.drop_duplicates()
            res = self.result[filtered_result.index]
            res = res.reset_index()
            return res.copy()
