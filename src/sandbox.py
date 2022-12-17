import numpy as np
from numpy import ndarray
from scipy.stats import ttest_rel
from sklearn import clone, tree
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate

from utils import Logger


CLASSIFIRES = {
    'MLP_A': MLPClassifier(
        hidden_layer_sizes = (30, 30, 30), max_iter=400,
        random_state = 75, solver = 'sgd'
    ),
    'CART': tree.DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

N_COMPONENTS = 2

N_SPLITS = 5
N_REPEATS = 2

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
)
pca = PCA(n_components=N_COMPONENTS)

def experiment(dataset: ndarray, oversampling, reverse_pca: bool, logger: Logger) -> None:
    logger.write("\n----------------------------------------\n")
    logger.write(f"Settings[ reverse_pca: {reverse_pca}, oversampling: {oversampling.__class__.__name__}]")

    X = dataset[:, :-1]
    Y = dataset[:, -1]
    class_mapping = {'negative': 0, 'positive': 1}
    converter = lambda t: class_mapping[t.strip()]
    vfunc = np.vectorize(converter)
    Y = vfunc(Y)

    scores = np.zeros((len(CLASSIFIRES), N_SPLITS*N_REPEATS))
    for fold_id, (train, test) in enumerate(rskf.split(X, Y)):
        for clf_id, clf_name in enumerate(CLASSIFIRES):
            clf = clone(CLASSIFIRES[clf_name])

            x_train = pca.fit_transform(X[train])
            if not reverse_pca:
                x_test = pca.transform(X[test])
            else:
                x_test= X[test]

            # * oversampling - SMOTE/ROS
            x_train, y_train = oversampling.fit_resample(x_train, Y[train])
            if reverse_pca:
                x_train = pca.inverse_transform(x_train)

            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            scores[clf_id, fold_id] = balanced_accuracy_score(Y[test], y_pred)

    mean = np.mean(scores, axis=1)
    std = np.std(scores, axis=1)

    for clf_id, clf_name in enumerate(CLASSIFIRES):
        logger.write(f"{clf_name}, {mean[clf_id]}, {std[clf_id]}")

    balanced_acc_score = np.copy(scores)
    logger.write("\n Balanced acc score valuesValues:")
    logger.write(str(balanced_acc_score))
    statistical_analysis(balanced_acc_score, logger)


def statistical_analysis(balanced_acc_score, logger):
    logger.write("\n################ statistical_analysis ####################\n")
    alfa = 0.05

    t_statistic = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
    p_value = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))

    # * t-student test
    for i in range(len(CLASSIFIRES)):
        for j in range(len(CLASSIFIRES)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(balanced_acc_score[i], balanced_acc_score[j])

    # * create tables for statistics
    headers = ["MLP_A", "CART", "KNN"]
    names_column = np.array([[header] for header in headers])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    logger.write(f"t-statistic:\n {t_statistic_table} \n\np-value:\n {p_value_table}")

    # *
    advantage = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    logger.write(f"Advantage:\n {advantage_table}")

    significance = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    logger.write(f"Statistical significance (alpha = 0.05):\n {significance_table}")

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    logger.write(f"Statistically significantly better:\n {stat_better_table}")
    logger.write("\n----------------------------------------\n")
