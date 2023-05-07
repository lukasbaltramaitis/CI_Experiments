import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

ACTIVITY = 'Activity'
CASE_ID = 'Case ID'
TIMESTAMP = 'Timestamp'
OUTCOME = 'Outcome'

COLUMNS_TO_DROP = [CASE_ID, ACTIVITY, TIMESTAMP]
COLUMNS = [ACTIVITY, CASE_ID, TIMESTAMP, OUTCOME]


class Preparation:
    def __init__(self):
        self.data = None
        self.encoded_data = None
        self.new_columns = None
        self.encoded_activities = None
        self.train = None
        self.test = None

    def _check_columns(self):
        for col in COLUMNS:
            if col not in self.data.columns:
                raise Exception(f"{col} not found in data.")

    def _simple_join_to_case(self, case: pd.DataFrame):
        case = case.drop(columns=COLUMNS_TO_DROP)
        return case.max()

    def _add_seq(self):
        data = self.data.copy()
        data['event_nr'] = self.data.groupby(CASE_ID, group_keys=False).cumcount()
        self.data[ACTIVITY] = data['event_nr'].astype('str') + '_' + self.data[ACTIVITY]

    def _add_empty_matrix(self, data: pd.DataFrame):
        for new_col in self.new_columns:
            data[new_col] = 0.0
        return data

    def _seq_encode(self):
        data = self.data.copy()
        data['event_nr'] = data.groupby(CASE_ID, group_keys=False).cumcount()
        data[ACTIVITY] = data['event_nr'].astype('str') + '_' + data[ACTIVITY]
        self.new_columns = data[ACTIVITY].unique()
        data = self._add_empty_matrix(data)
        for i, act in enumerate(data[ACTIVITY]):
            data.at[i, act] = 1.0
        data.drop(columns=['event_nr'])
        self.encoded_data = data

    def _time_join_to_case(self, case: pd.DataFrame):
        new_cols = self.new_columns
        case[TIMESTAMP] = pd.to_datetime(case[TIMESTAMP])
        case['Timestamp2'] = case[TIMESTAMP].shift(-1)
        case['Duration'] = (case['Timestamp2'] - case[TIMESTAMP]) / pd.Timedelta(seconds=1)
        case['Duration'] = case['Duration'].fillna(0.1)
        case[new_cols] = case[new_cols].mul(case['Duration'], axis=0)
        columns_to_drop = COLUMNS_TO_DROP + ['Timestamp2', 'Duration']
        case = case.drop(columns=columns_to_drop)
        grouped_case = case[new_cols].sum()
        outcome = case[OUTCOME].max()
        grouped_case[OUTCOME] = outcome
        return grouped_case

    def _aggregate(self, encode_type='onehot'):
        if encode_type == 'onehottime':
            self.encoded_data = self.encoded_data.groupby(CASE_ID).apply(self._time_join_to_case)
        else:
            self.encoded_data = self.encoded_data.groupby(CASE_ID).apply(self._simple_join_to_case)

    def _target_one_hot_encode(self):
        if self.train is not None and self.test is not None:
            encoder = TargetEncoder()
            encoded_column_name = f'encoded_activity'
            self.train[encoded_column_name] = encoder.fit_transform(
                self.train[ACTIVITY],
                self.train[OUTCOME])

            self.test[encoded_column_name] = encoder.transform(self.test[ACTIVITY])

            ohe = OneHotEncoder()
            temp_data = ohe.fit_transform(self.train[[ACTIVITY]]).toarray()
            self.new_cols = ohe.get_feature_names_out()
            encoded = pd.DataFrame(temp_data, columns=self.new_cols)
            self.train = self.train.join(encoded)

            temp_data_ = ohe.transform(self.test[ACTIVITY]).toarray()
            encoded = pd.DataFrame(temp_data_, columns=self.new_cols)
            self.test = self.test.join(encoded)

            self.train[self.new_cols] = \
                self.train[self.new_cols].to_numpy() * self.train[[encoded_column_name]].to_numpy()
            self.train = self.train.drop(columns=[encoded_column_name])

            self.test[self.new_cols] = \
                self.test[self.new_cols].to_numpy() * self.test[[encoded_column_name]].to_numpy()
            self.test = self.test.drop(columns=[encoded_column_name])

            self.train = self.train.groupby(CASE_ID).apply(self._simple_join_to_case)
            self.test = self.test.groupby(CASE_ID).apply(self._simple_join_to_case)

    def _one_hot_encode(self):
        if self.data is not None:
            encoded_data = self.data.copy()
            ohe = OneHotEncoder()
            enc = ohe.fit_transform(encoded_data[[ACTIVITY]]).toarray()
            self.new_columns = ohe.get_feature_names_out()
            self.encoded_activities = self.new_columns
            encoded = pd.DataFrame(enc, columns=self.new_columns)
            self.encoded_data = encoded_data.join(encoded)

    def _sort_by_timestamp(self):
        self.data = self.data.sort_values(by=TIMESTAMP)

    def _encode(self, encode_type):
        if encode_type == 'onehotseq':
            self._seq_encode()
        else:
            self._one_hot_encode()
        print("Encoded")
        self._aggregate(encode_type)
        print('Aggregated')

    def _split(self, test_size, random_state):
        if self.encoded_data is not None:
            x_train, x_test = train_test_split(self.encoded_data, test_size=test_size, random_state=random_state)
            self.train = x_train
            self.test = x_test

    def _target_split(self, test_size, random_state):
        if self.data is not None:
            case_id = self.data[CASE_ID].unique()
            x_train_cases, x_test_cases = train_test_split(case_id, test_size=test_size, random_state=random_state)

            self.train = self.data[self.data[CASE_ID].isin(x_train_cases.values)]
            self.test = self.data[self.data[CASE_ID].isin(x_test_cases.values)]

    def _save(self, path):
        if path is not None and self.train is not None and self.test is not None:
            train_path = f"{path}/train.csv"
            test_path = f"{path}/test.csv"
            self.train.to_csv(train_path, index=False)
            self.test.to_csv(test_path, index=False)

    def _remove_constant_value_features(self, df):
        return [e for e in df.columns if df[e].nunique() == 1]

    def _filter_single_class(self):
        if self.encoded_data is not None:
            self.encoded_data = self.encoded_data.reset_index(drop=True)
            cols_to_drop = self._remove_constant_value_features(self.encoded_data)
            new_df_columns = [e for e in self.encoded_data.columns if e not in cols_to_drop]
            self.encoded_data = self.encoded_data[new_df_columns]

    def prepare_data(
            self,
            data: pd.DataFrame,
            save_path=None,
            encode_type='onehottime',
            test_size=0.2,
            random_state=42
    ) -> dict:
        self.data = data.copy()
        self._check_columns()
        self._sort_by_timestamp()
        if encode_type == 'target':
            self._target_split(test_size, random_state)
            self._target_one_hot_encode()
        else:
            self._encode(encode_type)
            self._filter_single_class()
            self._split(test_size, random_state)
        self._save(save_path)
        return {'train': self.train, 'test': self.test}
