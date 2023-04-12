from typing import List
import pandas as pd
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import numpy as np


CASE_ID = 'Case ID'
ACTIVITY = 'Activity'
END = 'End'
SEP = ';'
POS = 'Pos'
NEG = 'Neg'
OUTCOME = 'Outcome'


def replace_outcome_with_pos_neg(data: pd.DataFrame) -> pd.DataFrame:
     data = data.copy()
     data[POS] = data[OUTCOME]
     data[NEG] = 1 - data[OUTCOME]

     data = data.drop(columns=[OUTCOME])
     
     return data


def read_csv(path: str, sep=SEP) -> pd.DataFrame:
    return pd.read_csv(path, sep)


def save_csv(data: pd.DataFrame, path: str):
        data.to_csv(path, index=False)


class Preparation:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.graph_set = set()
        self.graph = None


    def _join_to_case(self, case: pd.DataFrame):
        del case[CASE_ID]
        return case.max()


    def encode(self, columns_to_encode: List[str]):
        if self.data is not None:
            self.encoded_data = self.data.copy()
            for column in columns_to_encode:
                ohe = OneHotEncoder()
                encoded = pd.DataFrame(ohe.fit_transform(self.encoded_data[[column]]).toarray(), columns=ohe.get_feature_names_out())
                self.encoded_data = self.encoded_data.join(encoded)
            self.encoded_data = self.encoded_data.drop(columns=columns_to_encode)
            self.encoded_data = self.encoded_data.groupby(CASE_ID).apply(self._join_to_case)
    

    def _add_activities_to_columns(self, case: pd.DataFrame) -> pd.DataFrame:
            if self.new_cols is not None:
                for i in range(case.shape[0]):
                    case.iloc[i][self.new_cols[i]] = case.iloc[i][ACTIVITY]
            return case


    def encode_with_activity_seq(self, columns_to_encode: List[str]):
        nr_of_cols = self.data[CASE_ID].value_counts().max()
        self.new_cols = []
        for i in range(nr_of_cols):
            column = f'N{i}'
            self.new_cols.append(column)
            self.data[column] = END

        
        self.data = self.data.groupby(CASE_ID).apply(self._add_activities_to_columns)
        self.data = self.data.drop(columns=[ACTIVITY])
        columns_to_encode.remove(ACTIVITY)
        columns_to_encode += self.new_cols
        self.encode(columns_to_encode)


    def _save_graph(self, path):
        G = nx.from_edgelist(self.graph_set)
        self.graph = G
        adj_matrix = nx.to_pandas_adjacency(G)
        save_csv(adj_matrix, path)


    def create_activity_graph(self, path):
        activity_data = self.data[[CASE_ID, ACTIVITY]]
        activity_data['case_id_1'] = activity_data[CASE_ID].shift(-1)
        activity_data['activity_effect'] = activity_data[ACTIVITY].shift(-1)
        activity_data = activity_data.iloc[:-1]
        activity_data['edge'] = list(zip(activity_data[ACTIVITY], activity_data['activity_effect']))
        activity_data = activity_data[activity_data[CASE_ID] == activity_data['case_id_1']]
        self.graph_set = set(activity_data['edge'].tolist())
        self._save_graph(path)
