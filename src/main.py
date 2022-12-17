from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from classifires_comparator import ClassifiersComparator
from loaders import ConfigLoader, KeelDatasetLoader
from utils import Logger

CONFIG_FILE_PATH = './config.yaml'
CLASSIFIRES = {
    'MLP_A': MLPClassifier(
        hidden_layer_sizes = (30, 30, 30), max_iter=400,
        random_state = 75, solver = 'sgd'
    ),
    'CART': tree.DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=3)
}

def main() -> None:
    dataset_loader = KeelDatasetLoader()
    config = ConfigLoader.load(CONFIG_FILE_PATH)

    smote = SMOTE(random_state=1000)
    ros = RandomOverSampler(sampling_strategy='auto')

    dataset_names = config.get('datasets', [])
    for dateset_name in dataset_names:
        logger = Logger(log_file_name=dateset_name.replace(".dat", ".log"))
        logger.write(dateset_name)
        dataset = dataset_loader.load(dateset_name, convert_to_ndarray=True)

        comparator = ClassifiersComparator(classifiers=CLASSIFIRES, dataset=dataset, logger=logger)
        comparator.compare(oversampling=smote, reverse_pca=True)
        comparator.compare(oversampling=smote, reverse_pca=False)
        comparator.compare(oversampling=ros, reverse_pca=True)
        comparator.compare(oversampling=ros, reverse_pca=False)


if __name__ == "__main__":
    main()
