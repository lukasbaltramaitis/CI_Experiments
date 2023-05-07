import logging

from ylearn import Why
import pandas as pd
import time as t
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt

OUTCOME = 'Outcome'
ALL_ALGORITHMS = [
    'slearner',
    'tlearner',
    'xlearner',
    'tree',
    'grf',
    'bound',
    'iv',
    'div',
    'dr',
    'dml'
]
WORKING_ALGORITHMS = [
    'slearner',
    'tlearner',
    'tree',
    'grf',
    'dr'
]
ALGORITHMS = [
    'tlearner',
    'tree',
    'grf'
]


class CausalEstimation:
    def __init__(self):
        self.random_state = None
        self.train_data = None
        self.test_data = None
        self.result = None
        self.timeout = None
        self.treatment = None
        self.adjustment = None
        self.covariate = None
        self.instrument = None
        self.estimators_options = {}
        self.discrete_treatment = None
        self.adjust_treatment = None
        self.target_outcome = None
        self.qinis = None
        self.treat = None
        self.control = None
        self.reverse_treatment = None

    def _init_why(self, discrete_treatment, estimator, estimator_options):
        return Why(
            discrete_outcome=True,
            discrete_treatment=discrete_treatment,
            estimator=estimator,
            estimator_options=estimator_options,
            random_state=self.random_state
        )

    def _fit(self, why, train_data, treatment, adjustment, covariate, instrument, estimator):
        if why is not None:
            if estimator == 'div' or estimator == 'iv':
                why.fit(
                    data=train_data,
                    outcome=OUTCOME,
                    treatment=treatment,
                    adjustment=adjustment,
                    covariate=covariate,
                    instrument=instrument
                )
            else:
                if instrument is not None:
                    covariate = covariate + instrument
                why.fit(
                    data=train_data,
                    outcome=OUTCOME,
                    treatment=treatment,
                    adjustment=adjustment,
                    covariate=covariate
                )
            return why

    def _causal_effect(
            self,
            why,
            test_data,
            target_outcome=None
    ):
        if why is not None:
            return why.causal_effect(
                test_data,
                treat=self.treat,
                control=self.control,
                target_outcome=target_outcome,
                return_detail=True
            )

    def _log(self, msg: str, estimator_name: str, duration: float):
        print(
            f'**************************\n{msg}\nestimator: {estimator_name}\nduration: {duration}\n')

    def _add_result(self, estimator_name, effect, duration):
        effect['estimator_name'] = estimator_name
        effect['duration'] = duration
        if self.result is None:
            self.result = effect
        else:
            self.result = pd.concat([self.result, effect], ignore_index=True)

    def _uplift(self, why, treatment, estimator):
        try:
            um = why.uplift_model(
                test_data=self.test_data,
                treatment=treatment,
                target_outcome=self.target_outcome,
                name=f"{estimator}__{treatment}",
                random=f"{estimator}__{treatment}__random")
            return um.get_qini()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logging.exception("Unexpected error!")
            return None

    def _uplift_treatments(self, why, estimator):
        treatment = self.treatment
        for tr in treatment:
            qini = self._uplift(why, tr, estimator)
            if self.qinis is None:
                self.qinis = qini
            else:
                self.qinis = pd.concat([self.qinis, qini], axis=1)

    def _estimate_and_uplift(
            self,
            estimator,
            estimator_options

    ):
        why = self._init_why(self.discrete_treatment, estimator, estimator_options)
        start = t.time()
        why = self._fit(why, self.train_data, self.treatment, self.adjustment, self.covariate, self.instrument,
                        estimator)
        end = t.time()
        duration = end - start
        if len(self.treatment) > 1 and (self.treat is not None or self.control is not None):
            print("Warning: treatment len > 1 and treat value is given! It could result in inaccuracy of other_results")
        effect = self._causal_effect(why, self.test_data, self.target_outcome)
        if self.discrete_treatment:
            self._uplift_treatments(why, estimator)
            self._log('Estimation finished', estimator, duration)
            self._add_result(estimator, effect, duration=duration)

    def _save_result(self, path):
        if path is not None:
            if self.result is not None:
                result_path = f"{path}/estimation_result.csv"
                self.result.to_csv(result_path, index=False)
            if self.qinis is not None:
                result_path = f"{path}/qinis.csv"
                self.qinis.to_csv(result_path, index=False)

    def _estimate_and_uplift_with_timeout(self, estimator_name, estimator_options):
        try:
            func_timeout(self.timeout, self._estimate_and_uplift, args=(
                estimator_name,
                estimator_options))
        except FunctionTimedOut:
            print(f"{estimator_name} fit/estimate could not complete within {self.timeout} seconds and was terminated."
                  f"\n")

    def _estimate_and_uplift_with_error_except(self, estimate_and_uplift_func, estimator_name, estimator_options):
        try:
            estimate_and_uplift_func(estimator_name, estimator_options)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            logging.exception("Unexpected error!")

    def _prepare_discrete_treatment(self):
        new_treatment = []
        treatment_val = 1.0
        control_val = 0.0
        if self.reverse_treatment:
            treatment_val = 0.0
            control_val = 1.0
        for treatment in self.treatment:
            mean = np.mean(self.train_data[self.train_data[treatment] > 0.0][treatment])
            new_treatment_col = f"{treatment}_treatment"
            self.train_data[new_treatment_col] = \
                np.where((0.0 < self.train_data[treatment]) & (self.train_data[treatment] <= mean), treatment_val,
                         control_val)
            self.test_data[new_treatment_col] = \
                np.where((0.0 < self.test_data[treatment]) & (self.test_data[treatment] <= mean), treatment_val,
                         control_val)
            new_treatment.append(new_treatment_col)
        self.treatment = new_treatment
        self.discrete_treatment = True

    def _estimate_and_uplift_algs(self, algorithms):
        if (not self.discrete_treatment) and self.adjust_treatment:
            self._prepare_discrete_treatment()
        for alg in algorithms:
            estimator_options = self.estimators_options.get(alg, {})
            self._estimate_and_uplift_with_error_except(
                self._estimate_and_uplift_with_timeout,
                estimator_name=alg,
                estimator_options=estimator_options)

    def estimate_and_uplift(
            self,
            train_data: pd.DataFrame,
            test_data: pd.DataFrame,
            treatment,
            adjustment,
            covariate,
            instrument,
            treat=None,
            control=None,
            estimators_options={},
            discrete_treatment=True,
            adjust_treatment=False,
            target_outcome=None,
            reverse_treatment=False,
            path=None,
            random_state=42,
            algorithms=ALGORITHMS,
            timeout=1800
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.treatment = treatment
        self.adjustment = adjustment
        self.covariate = covariate
        self.instrument = instrument
        self.treat = treat
        self.control = control
        self.estimators_options = estimators_options
        self.discrete_treatment = discrete_treatment
        self.adjust_treatment = adjust_treatment
        self.target_outcome = target_outcome
        self.reverse_treatment = reverse_treatment
        self.random_state = random_state
        self.timeout = timeout
        self._estimate_and_uplift_algs(algorithms)
        self._save_result(path)
        self.plot_qini_curves(path)
        return self.result, self.qinis

    def _plot_qini_subplots(self, qinis, path):
        nr_of_qinis = int(len(qinis.columns) / 2)
        nr_of_models = len(set([col.split('__')[0] for col in qinis.columns]))
        nr_of_treatments = int(nr_of_qinis / nr_of_models)
        fig, axs = plt.subplots(nr_of_treatments, nr_of_models, figsize=(9, 9))
        for i in range(nr_of_treatments):
            for j in range(nr_of_models):
                ax = axs[i][j]
                index = (i + j * nr_of_treatments) * 2
                x_axis_name = qinis.columns[index].split('__')[0]
                y_axis_name = str(qinis.columns[index].split('__')[-1])\
                    .replace('_treatment', '')\
                    .replace('Activity_', '')
                qini = qinis[qinis.columns[index]]
                random = qinis[qinis.columns[index + 1]]
                ax.plot(qini.index, qini, label='estimator')
                ax.plot(random.index, random, label='random')
                ax.legend()
                ax.set(xlabel=x_axis_name, ylabel=y_axis_name)
                ax.label_outer()
        fig.savefig(f"{path}/qini_subplots.png")

    def _plot_qini_comparison(self, qinis, path):
        nr_of_qinis = int(len(qinis.columns) / 2)
        nr_of_models = len(set([col.split('__')[0] for col in qinis.columns]))
        nr_of_treatments = int(nr_of_qinis / nr_of_models)
        fig, axs = plt.subplots(1, nr_of_treatments, figsize=(6 * nr_of_treatments, 5))
        for i in range(nr_of_treatments):
            ax = axs[i]
            for j in range(nr_of_models):
                index = (j * nr_of_treatments + i) * 2
                qini = qinis[qinis.columns[index]]
                name = qini.name.split('__')[0]
                ax.plot(qini.index, qini, label=name)
            random_idx = 2 * i + 1
            random = qinis[qinis.columns[random_idx]]
            title = random.name.split('__')[1]
            ax.set_title(title)
            ax.plot(random.index, random, label='random')
            ax.set(xlabel='Population', ylabel='Number of incremental outcome')
            ax.legend(title='Estimators')
        fig.savefig(f"{path}/qini_compare.png")

    def plot_qini_curves(self, path, qinis=None):
        if qinis is None:
            qinis = self.qinis
        if qinis is not None:
            self._plot_qini_subplots(qinis, path)
            self._plot_qini_comparison(qinis, path)
