from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from comparator import Comparator
from loaders import ConfigLoader, KeelDatasetLoader
from utils import Logger
import pandas as pd
from pandas import DataFrame

CONFIG_FILE_PATH = './config.yaml'
CLASSIFIRES = {
    'MLP': MLPClassifier(
        hidden_layer_sizes = (30, 30, 30), max_iter=400,
        random_state = 75, solver = 'sgd'
    ),
    'CART': tree.DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

OS_SMOTE = SMOTE(random_state=1000)
OS_ROS = RandomOverSampler(sampling_strategy='auto')

def merge_df(main_df: DataFrame, row_df: DataFrame) -> DataFrame:
    return pd.concat([row_df, main_df], ignore_index=True)

def main() -> None:
    dataset_loader = KeelDatasetLoader()
    config = ConfigLoader.load(CONFIG_FILE_PATH)

    dataset_names = config.get('datasets', [])
    logger = Logger(log_file_name='output.txt')

    all_datasets_df_acc = pd.DataFrame()
    all_datasets_df_stat_signi_better = pd.DataFrame()

    for classifire_name in CLASSIFIRES:
        for dateset_name in dataset_names:
            logger.write(dateset_name)
            dataset = dataset_loader.load(dateset_name, convert_to_ndarray=True)

            comparator = Comparator(classifier=CLASSIFIRES[classifire_name], dataset=dataset, logger=logger, dataset_name=dateset_name)

            acc_score_smote_with_reverse = comparator.calculate_accuracy(oversampling=OS_SMOTE, reverse_pca=True)
            acc_score_smote_without_reverse = comparator.calculate_accuracy(oversampling=OS_SMOTE, reverse_pca=False)
            acc_score_ros_with_reverse = comparator.calculate_accuracy(oversampling=OS_ROS, reverse_pca=True)
            acc_score_ros_without_reverse = comparator.calculate_accuracy(oversampling=OS_ROS, reverse_pca=False)

            dataset_df_acc = pd.DataFrame(data = [{
                'dataset': dateset_name,
                'SMOTE-REVERSE': comparator.get_acc_mean(acc_score_smote_with_reverse),
                'SMOTE-NO-REVERSE': comparator.get_acc_mean(acc_score_smote_without_reverse),
                'ROS-REVERSE': comparator.get_acc_mean(acc_score_ros_with_reverse),
                'ROS-NO-REVERSE': comparator.get_acc_mean(acc_score_ros_without_reverse)
            }])
            all_datasets_df_acc = merge_df(dataset_df_acc, all_datasets_df_acc)

            dataset_df_stat_signi_better = comparator.do_statystical_analysis([
                acc_score_smote_with_reverse, acc_score_smote_without_reverse,
                acc_score_ros_with_reverse, acc_score_ros_without_reverse
            ])
            all_datasets_df_stat_signi_better = merge_df(dataset_df_stat_signi_better, all_datasets_df_stat_signi_better)

        all_datasets_df_acc.to_csv(f'./output/results_acc_{classifire_name}.csv')
        all_datasets_df_stat_signi_better.to_csv(f'./output/results_better_{classifire_name}.csv')
        all_datasets_df_acc = pd.DataFrame()
        all_datasets_df_stat_signi_better = pd.DataFrame()


if __name__ == "__main__":
    main()
