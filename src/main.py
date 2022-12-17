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
        dataset = dataset_loader.load(dateset_name).to_numpy()
        logger = Logger(dateset_name.replace(".dat", ".txt"))
        logger.write(dateset_name)

        experiment(dataset, smote, reverse_pca=True, logger=logger)
        logger.write("\n----------------------------------------\n")

        experiment(dataset, smote, reverse_pca=False, logger=logger)
        logger.write("\n----------------------------------------\n")

        # experiment(dataset, ros, reverse_pca=True, logger=logger)
        # logger.write("\n----------------------------------------\n")

        # experiment(dataset, ros, reverse_pca=False, logger=logger)
        logger.write("\n----------------------------------------\n")

if __name__ == "__main__":
    main()
