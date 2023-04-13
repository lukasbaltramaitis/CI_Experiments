from typing import List

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