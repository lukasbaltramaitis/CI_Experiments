from typing import List
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


CASE_ID = 'Case ID'
ACTIVITY = 'Activity'
END = 'End'
SEP = ';'


def read_csv(path: str, sep=SEP) -> pd.DataFrame:
    return pd.read_csv(path, sep)


def save_csv(data: pd.DataFrame, path: str):
        data.to_csv(path, index=False)


class Preparation:
    def __init__(self, data: pd.DataFrame):
        self.data = data


    def encode(self, columns_to_encode: List[str]):
        if self.data is not None:
            data = data.copy()
            for column in columns_to_encode:
                ohe = OneHotEncoder()
                encoded = pd.DataFrame(ohe.fit_transform(self.data[[column]]).toarray())
                self.data = self.data.join(encoded)
            self.data = self.data.drop(columns=columns_to_encode)
    

    def _add_activities_to_columns(self, case: pd.DataFrame) -> pd.DataFrame:
            if self.new_cols is not None:
                for i in range(case.shape[0]):
                    case.iloc[i][self.new_cols[i]] = case.iloc[i][ACTIVITY]
            return case


    def encode_with_activity_seq(self, columns_to_encode: List[str]):
        def get_case_length(case: pd.DataFrame) -> int:
            return case.shape[0]
        
        nr_of_cols = max(self.data.groupby(CASE_ID).apply(get_case_length))
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
        edges = [(key, val) for key, vals in self.graph.items() for val in vals]
        df = pd.DataFrame(edges)
        adj_matrix = pd.crosstab(df[0], df[1])
        save_csv(adj_matrix, path)

    def create_activity_graph(self, path):
        def add_relationships_from_the_case(case: pd.DataFrame):
            nr_of_rows = case.shape[0]
            for i in range(nr_of_rows - 1):
                activity_cause = case.iloc[i][ACTIVITY]
                activity_effect = case.iloc[i+1][ACTIVITY]
                graph_values = self.graph.get(activity_cause, None)
                if graph_values is not None:
                    graph_values.add(activity_effect)
                    self.graph[activity_cause] = graph_values
                else:
                    self.graph[activity_cause] = set(activity_effect)

        activity_data = self.data[[CASE_ID, ACTIVITY]]
        self.graph = {}
        activity_data.groupby(ACTIVITY).apply(add_relationships_from_the_case)
        self._save_graph(path)
