import numpy as np
from numpy import ndarray
from scipy.stats import ttest_rel
from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import \
    balanced_accuracy_score  # we can also check accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate
import pandas as pd
from typing import List

from utils import Logger


class Comparator:
    N_SPLITS = 5
    N_REPEATS = 2
    N_COMPONENTS = 2    # number of features left after PCA operation
    ALFA = 0.05

    def __init__(self, classifier, dataset: ndarray, logger: Logger, dataset_name: str) -> None:
        self._classifier = classifier
        self._dataset = dataset
        self._logger = logger
        self._dataset_name = dataset_name
        self._rskf = RepeatedStratifiedKFold(n_splits=self.N_SPLITS, n_repeats=self.N_REPEATS, random_state=42)
        self._pca = PCA(n_components=self.N_COMPONENTS)

    @ignore_warnings(category=ConvergenceWarning)
    def calculate_accuracy(self, oversampling, reverse_pca) -> ndarray:
        self._logger.write(f"Settings[ reverse_pca: {reverse_pca}, oversampling: {oversampling.__class__.__name__}]")
        X = self._dataset[:, :-1]
        Y = self._dataset[:, -1]
        scores = np.zeros(self.N_SPLITS*self.N_REPEATS)

        for fold_id, (train, test) in enumerate(self._rskf.split(X, Y)):
            clf = clone(self._classifier)

            x_train = self._pca.fit_transform(X[train])
            x_test = X[test] if reverse_pca else self._pca.transform(X[test])

            # * oversampling - SMOTE/ROS
            x_train, y_train = oversampling.fit_resample(x_train, Y[train])
            if reverse_pca:
                x_train = self._pca.inverse_transform(x_train)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            scores[fold_id] = balanced_accuracy_score(Y[test], y_pred)

        acc_score = np.copy(scores)
        self._logger.write("\n Balanced acc score values:")
        self._logger.write(str(acc_score))
        return acc_score

    def get_acc_mean(self, acc_score):
        return round(np.mean(acc_score), 2)

    def do_statystical_analysis(self, all_acc_score: List[ndarray]) -> None:
        print("--------ALL ACC score-----------")
        print(all_acc_score)

        # * init tables
        shape = (4, 4)
        t_statistic = np.zeros(shape)
        p_value = np.zeros(shape)
        advantage = np.zeros(shape)
        significance = np.zeros(shape)

        # * t-student test
        for i in range(shape[0]):
            for j in range(shape[1]):
                t_statistic[i, j], p_value[i, j] = ttest_rel(all_acc_score[i], all_acc_score[j])

        # * create tables for statistics
        headers = ["SMOTE-REVERSE", "SMOTE-NO-REVERSE", "ROS-REVERSE", "ROS-NO-REVERSE"]
        names_column = np.array([[header] for header in headers])
        t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
        t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
        p_value_table = np.concatenate((names_column, p_value), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        self._logger.write(f"t-statistic:\n {t_statistic_table} \n\np-value:\n {p_value_table}")

        advantage[t_statistic > 0] = 1
        advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
        self._logger.write(f"Advantage:\n {advantage_table}")

        significance[p_value <= self.ALFA] = 1
        significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
        self._logger.write(f"Statistical significance (alpha = 0.05):\n {significance_table}")

        stat_better = significance * advantage
        print(stat_better.shape)
        print(stat_better)
        stat_better_table = tabulate(np.concatenate((names_column, stat_better), axis=1), headers)


        df = pd.DataFrame(data = [{
            'dataset': self._dataset_name,
            headers[0]: np.array2string(np.where(stat_better[0][:] == 1)[0]),
            headers[1]: np.array2string(np.where(stat_better[1][:] == 1)[0]),
            headers[2]: np.array2string(np.where(stat_better[2][:] == 1)[0]),
            headers[3]: np.array2string(np.where(stat_better[3][:] == 1)[0]),
        }])
        self._logger.write(f"Statistically significantly better:\n {stat_better_table}")
        self._logger.write("\n----------------------------------------\n")
        return df
