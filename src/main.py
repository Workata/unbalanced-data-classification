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

N_COMPONENTS = 2

dataset_loader = KeelDatasetLoader()

ecoli_1_df = dataset_loader.load("ecoli1.dat")
print(ecoli_1_df)
pca = PCA(n_components=N_COMPONENTS)

X = ecoli_1_df[ecoli_1_df.columns[~ecoli_1_df.columns.isin(['class'])]]
Y = ecoli_1_df[ecoli_1_df.columns[ecoli_1_df.columns.isin(['class'])]]

principal_components = pca.fit_transform(X)

principal_df_X = pd.DataFrame(
    data=principal_components,
    columns = ['principal component 1', 'principal component 2']
)
print("Principal df X:")
print(principal_df_X)   # dataframe
print(f"Principal df X type: {type(principal_df_X)}")


classifiers = {
    'MLP_A': MLPClassifier(hidden_layer_sizes = (30, 30, 30), max_iter=400, random_state = 75, solver = 'sgd'),
    'CART': tree.DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

smote = SMOTE(random_state=1000)

values = []
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42
)
scores = np.zeros((len(classifiers), n_splits * n_repeats))

x = principal_df_X
y = Y

for fold_id, (train, test) in enumerate(rskf.split(x, y)):
    for clf_id, clf_name in enumerate(classifiers):
        clf = clone(classifiers[clf_name])

        # * oversampling - SMOTE
        x_train, y_train = smote.fit_resample(x[train], y[train])

        clf.fit(x_train, y_train)
        y_pred = clf.predict(x[test])
        scores[clf_id, fold_id] = balanced_accuracy_score(y[test], y_pred)

mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(classifiers):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

values.append(np.copy(scores))

print("################ Analiza statystyczna ####################")

# ###########################################################################
# #                        analiza statystyczna                             #
# ###########################################################################
# values = np.load('results.npy')
print(values)

alfa = .05

for index, scores in enumerate(values):
    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])

    print("Liczba cech: ", index + 1)

    headers = ["10 neuronów bez momentum", "80 neuronów bez momentum", "400 neuronów bez momentum", "10 neuronów z momentum", "80 neuronów z momentum", "400 neuronów z momentum"]
    names_column = np.array([["10 neuronów bez momentum"], ["80 neuronów bez momentum"], ["400 neuronów bez momentum"], ["10 neuronów z momentum"], ["80 neuronów z momentum"], ["400 neuronów z momentum"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(classifiers), len(classifiers)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(classifiers), len(classifiers)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
