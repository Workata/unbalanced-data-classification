from loaders import KeelDatasetLoader
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sandbox import experiment

from utils import Logger

DATASETES_NAMES = [
    "ecoli1.dat",
    "ecoli2.dat"
]


dataset_loader = KeelDatasetLoader()

smote = SMOTE(random_state=1000)
ros = RandomOverSampler(sampling_strategy='auto')


for dateset_name in DATASETES_NAMES:
    dataset = dataset_loader.load(dateset_name).to_numpy()
    logger = Logger(dateset_name.replace(".dat", ".txt"))
    logger.write(dateset_name)

    experiment(dataset, smote, reverse_pca=True, logger=logger)
    logger.write("\n########################################\n")
    experiment(dataset, smote, reverse_pca=False, logger=logger)
    logger.write("\n########################################\n")
    experiment(dataset, ros, reverse_pca=True, logger=logger)
    logger.write("\n########################################\n")
    experiment(dataset, ros, reverse_pca=False, logger=logger)
    logger.write("\n########################################\n")
