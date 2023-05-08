import unittest
import pandas as pd

from CI_Experiments.pipeline.preparation import Preparation


def prepare_artificial_data():
    data = {
        'Case ID': [0, 0, 0, 0, 1, 1, 1],
        'Activity': ['1', '3', '1', '4', '1', '2', '5'],
        'Timestamp': ['2023-01-01 09:51:16', '2023-01-01 09:51:17', '2023-01-01 09:51:19', '2023-01-01 09:51:22',
                      '2023-01-01 09:53:10', '2023-01-01 09:53:15', '2023-01-01 09:53:25'],
        'Outcome': [1, 1, 1, 1, 0, 0, 0]
    }
    return pd.DataFrame(data)


class EncodingTestCase(unittest.TestCase):
    def test_outcome_aggregation(self):
        # given the data with Case ID, Activity, Timestamp variables
        data = prepare_artificial_data()
        # and preparation step class
        preparation = Preparation()
        preparation.data = data
        # when data is encoded
        preparation._encode('onehottime')
        encoded_data = preparation.encoded_data
        # then encoded data Outcome should be equal to 1 and 0
        encoded_data_outcome_values = encoded_data['Outcome'].values
        expected_result = [1, 0]
        self.assertEqual(encoded_data_outcome_values[0], expected_result[0])
        self.assertEqual(encoded_data_outcome_values[1], expected_result[1])

    def test_encoded_new_columns(self):
        # given the data with Case ID, Activity, Timestamp variables
        data = prepare_artificial_data()
        # and preparation step class
        preparation = Preparation()
        preparation.data = data
        # when data is encoded
        preparation._encode('onehottime')
        encoded_data = preparation.encoded_data
        # then encoded data should have Activity_1, Activity_2, ..., Activity_5 columns
        columns = encoded_data.columns
        self.assertIn('Activity_1', columns)
        self.assertIn('Activity_2', columns)
        self.assertIn('Activity_3', columns)
        self.assertIn('Activity_4', columns)
        self.assertIn('Activity_5', columns)

    def test_encoded_activity_values(self):
        # given the data with Case ID, Activity, Timestamp variables
        data = prepare_artificial_data()
        # and preparation step class
        preparation = Preparation()
        preparation.data = data
        # when data is encoded
        preparation._encode('onehottime')
        encoded_data = preparation.encoded_data
        # then encoded data of Activity_1 should be equal to [4, 5],
        activity_1 = encoded_data['Activity_1']
        expected_result = [4, 5]
        self.assertEqual(activity_1[0], expected_result[0])
        self.assertEqual(activity_1[1], expected_result[1])


if __name__ == '__main__':
    unittest.main()
