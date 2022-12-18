from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from classifires_comparator import ClassifiersComparator
from loaders import ConfigLoader, KeelDatasetLoader
from utils import Logger
import pandas as pd

CONFIG_FILE_PATH = './config.yaml'
CLASSIFIRES = {
    'MLP': MLPClassifier(
        hidden_layer_sizes = (30, 30, 30), max_iter=400,
        random_state = 75, solver = 'sgd'
    )#,
    # 'CART': tree.DecisionTreeClassifier(),
    # 'KNN': KNeighborsClassifier(n_neighbors=3)
}

def main() -> None:
    dataset_loader = KeelDatasetLoader()
    config = ConfigLoader.load(CONFIG_FILE_PATH)

    smote = SMOTE(random_state=1000)
    ros = RandomOverSampler(sampling_strategy='auto')

    dataset_names = config.get('datasets', [])
    logger = Logger(log_file_name="output.txt")

    df_acc = pd.DataFrame()
    df_better = pd.DataFrame()

    for dateset_name in dataset_names:
        # for classifire
        logger.write(dateset_name)
        dataset = dataset_loader.load(dateset_name, convert_to_ndarray=True)

        comparator = ClassifiersComparator(classifiers=CLASSIFIRES, dataset=dataset, logger=logger, dataset_name=dateset_name)
        all_acc_score = []
        acc_score, df_acc = comparator.compare(oversampling=smote, reverse_pca=True)
        all_acc_score.append(acc_score)
        acc_score, df_acc = comparator.compare(oversampling=smote, reverse_pca=False)
        all_acc_score.append(acc_score)
        acc_score, df_acc = comparator.compare(oversampling=ros, reverse_pca=True)
        all_acc_score.append(acc_score)
        acc_score, df_acc = comparator.compare(oversampling=ros, reverse_pca=False)
        all_acc_score.append(acc_score)

        comparator.do_statystical_analysis(all_acc_score)

        # df_acc = pd.concat([output_df_acc, df_acc], ignore_index=True)
        # df_better = pd.concat([output_df_better, df_better], ignore_index=True)
        # comparator.compare(oversampling=smote, reverse_pca=False)
        # comparator.compare(oversampling=ros, reverse_pca=True)
        # comparator.compare(oversampling=ros, reverse_pca=False)

    df_acc.to_csv('./output/results_acc.csv')
    df_better.to_csv('./output/results_better.csv')

if __name__ == "__main__":
    main()
