from loaders import KeelDatasetLoader, ConfigLoader
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sandbox import experiment

from utils import Logger

CONFIG_FILE_PATH = './config.yaml'

def main() -> None:
    dataset_loader = KeelDatasetLoader()
    config = ConfigLoader.load(CONFIG_FILE_PATH)

    smote = SMOTE(random_state=1000)
    ros = RandomOverSampler(sampling_strategy='auto')

    dataset_names = config.get('datasets', [])
    for dateset_name in dataset_names:
        dataset = dataset_loader.load(dateset_name, convert_to_ndarray=True)
        logger = Logger(log_file_name=dateset_name.replace(".dat", ".log"))
        logger.write(dateset_name)

        experiment(dataset, oversampling=smote, reverse_pca=True, logger=logger)
        experiment(dataset, oversampling=smote, reverse_pca=False, logger=logger)
        # experiment(dataset, ros, reverse_pca=True, logger=logger)
        # experiment(dataset, ros, reverse_pca=False, logger=logger)

if __name__ == "__main__":
    main()
