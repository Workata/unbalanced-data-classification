from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from comparator import Comparator, PCAMode
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
# * w/o -> without, w/ -> with, r -> reverse
HEADERS = [
    "SMOTE w/o PCA", "SMOTE w/ PCA", "SMOTE w/ rPCA",
    "ROS w/o PCA", "ROS w/ PCA", "ROS w/ rPCA"
]

def merge_df(main_df: DataFrame, row_df: DataFrame) -> DataFrame:
    return pd.concat([row_df, main_df], ignore_index=True)

def df_to_latex(df: DataFrame) -> str:
    return df.style.format(precision=2).hide(axis="index").to_latex(column_format="|c|c|c|c|c|c|c|", hrules=True, siunitx=True)

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
            comparator = Comparator(classifier=CLASSIFIRES[classifire_name], dataset=dataset, logger=logger)

            # metric used: balanced_accuracy_score
            metric_scores_for_smote_without_pca = comparator.calculate_accuracy(oversampling=OS_SMOTE, pca_mode=PCAMode.WITHOUT_PCA)
            metric_scores_for_smote_with_pca = comparator.calculate_accuracy(oversampling=OS_SMOTE, pca_mode=PCAMode.WITH_PCA)
            metric_scores_for_smote_with_revese_pca = comparator.calculate_accuracy(oversampling=OS_SMOTE, pca_mode=PCAMode.WITH_REVERSE_PCA)

            metric_scores_for_ros_without_pca = comparator.calculate_accuracy(oversampling=OS_ROS, pca_mode=PCAMode.WITHOUT_PCA)
            metric_scores_for_ros_with_pca = comparator.calculate_accuracy(oversampling=OS_ROS, pca_mode=PCAMode.WITH_PCA)
            metric_scores_for_ros_with_revese_pca = comparator.calculate_accuracy(oversampling=OS_ROS, pca_mode=PCAMode.WITH_REVERSE_PCA)

            dataset_df_acc = pd.DataFrame(data = [{
                'dataset': dataset.name,
                HEADERS[0]: comparator.get_acc_mean(metric_scores_for_smote_without_pca),
                HEADERS[1]: comparator.get_acc_mean(metric_scores_for_smote_with_pca),
                HEADERS[2]: comparator.get_acc_mean(metric_scores_for_smote_with_revese_pca),
                HEADERS[3]: comparator.get_acc_mean(metric_scores_for_ros_without_pca),
                HEADERS[4]: comparator.get_acc_mean(metric_scores_for_ros_with_pca),
                HEADERS[5]: comparator.get_acc_mean(metric_scores_for_ros_with_revese_pca),
            }])
            all_datasets_df_acc = merge_df(dataset_df_acc, all_datasets_df_acc)

            dataset_df_stat_signi_better = comparator.do_statystical_analysis([
                metric_scores_for_smote_without_pca, metric_scores_for_smote_with_pca,
                metric_scores_for_smote_with_revese_pca, metric_scores_for_ros_without_pca,
                metric_scores_for_ros_with_pca,  metric_scores_for_ros_with_revese_pca
            ], headers=HEADERS)
            all_datasets_df_stat_signi_better = merge_df(dataset_df_stat_signi_better, all_datasets_df_stat_signi_better)

        output = pd.concat([all_datasets_df_acc, all_datasets_df_stat_signi_better]).sort_index(kind='merge')
        output.to_csv(f'./output/results_all_{classifire_name}.csv')
        logger.write(df_to_latex(output))
        all_datasets_df_acc = pd.DataFrame()
        all_datasets_df_stat_signi_better = pd.DataFrame()


if __name__ == "__main__":
    main()
