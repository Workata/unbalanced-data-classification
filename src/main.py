from loaders import KeelDatasetLoader

from sklearn import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn import tree

from tabulate import tabulate
from scipy.stats import ttest_rel
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


# number of features left after PCA
N_COMPONENTS = 2

CLASSIFIRES = {
    'MLP_A': MLPClassifier(hidden_layer_sizes = (30, 30, 30), max_iter=400, random_state = 75, solver = 'sgd'),
    'CART': tree.DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

# * load dataset
dataset_loader = KeelDatasetLoader()
ecoli_1_df = dataset_loader.load("ecoli1.dat")
ecoli_1_nd = ecoli_1_df.to_numpy()
print(ecoli_1_nd)

pca = PCA(n_components=N_COMPONENTS)

X = ecoli_1_nd[:, :-1]
Y = ecoli_1_nd[:, -1]
print(X[:10])
print(Y[:10])
class_mapping = {'negative': 0, 'positive': 1}
converter = lambda t: class_mapping[t.strip()]
vfunc = np.vectorize(converter)
Y = vfunc(Y)
print(Y[:10])


principal_components = pca.fit_transform(X)
print("\nPrincipal componenets:")
print(principal_components)   # dataframe

smote = SMOTE(random_state=1000)

balanced_acc_score_values = []
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42
)
scores = np.zeros((len(CLASSIFIRES), n_splits * n_repeats))
print("\nScores zeros:")
print(scores)

x = principal_components
y = Y


for fold_id, (train, test) in enumerate(rskf.split(x, y)):
    for clf_id, clf_name in enumerate(CLASSIFIRES):
        clf = clone(CLASSIFIRES[clf_name])

        # * oversampling - SMOTE
        x_train, y_train = smote.fit_resample(x[train], y[train])

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x[test])
        scores[clf_id, fold_id] = balanced_accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(CLASSIFIRES):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))


balanced_acc_score_values.append(np.copy(scores))
print("\nValues:")
print(balanced_acc_score_values)



def statistical_analysis():
    print("\n################ statistical_analysis ####################\n")
    alfa = .05

    for scores in balanced_acc_score_values:
        print(f"Liczba cech: {N_COMPONENTS}")

        t_statistic = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
        p_value = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))

        for i in range(len(CLASSIFIRES)):
            for j in range(len(CLASSIFIRES)):
                t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])



        headers = ["MLP_A", "CART", "KNN"]
        names_column = np.array([["MLP_A"], [ "CART"], ["KNN"]])
        t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
        t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
        p_value_table = np.concatenate((names_column, p_value), axis=1)
        p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

        advantage = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
        advantage[t_statistic > 0] = 1
        advantage_table = tabulate(np.concatenate(
            (names_column, advantage), axis=1), headers)
        print("Advantage:\n", advantage_table)

        significance = np.zeros((len(CLASSIFIRES), len(CLASSIFIRES)))
        significance[p_value <= alfa] = 1
        significance_table = tabulate(np.concatenate(
            (names_column, significance), axis=1), headers)
        print("Statistical significance (alpha = 0.05):\n", significance_table)

        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        print("Statistically significantly better:\n", stat_better_table)

statistical_analysis()
