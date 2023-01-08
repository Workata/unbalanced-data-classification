import pandas as pd
from pandas import DataFrame
import re
from typing import List
from models import Dataset


class KeelDatasetLoader:
    """
    FORMAT:
    @...
    @...
    @inputs (features)
    @outputs (classes)
    @data
    (csv data)
    """

    DATASET_FOLDER = "./datasets"

    def load(self, dataset_name: str, convert_to_ndarray=False) -> Dataset:
        """
        Returns dataset by given dataset name.
        By default it's a dataframe, but can be converted to numpy's ndarray.
        """
        dataset_path = self._get_dataset_path(dataset_name)
        dataset_file = self._read_file(file_path=dataset_path)
        headers = self._get_dataset_headers(dataset_file)
        dataset_df = pd.read_csv(
            dataset_path,
            skiprows=lambda idx : dataset_file.split('\n')[idx].startswith("@"),
            names=headers,
            index_col=None
        )
        dataset_df = self._map_classes(dataset_df)
        data = dataset_df.to_numpy() if convert_to_ndarray else dataset_df
        return Dataset(name=dataset_name.replace(".dat", ""), data=data)

    def _map_classes(self, df: DataFrame) -> DataFrame:
        class_mapping = {'negative': 0, 'positive': 1}
        df['Class'] = df['Class'].apply(lambda x: class_mapping[x.strip()])
        return df

    def _get_dataset_headers(self, dataset_file: str) -> List[str]:
        features_str = self._extract_attribute(pattern='@inputs (.*)', content=dataset_file)
        classes_str = self._extract_attribute(pattern='@outputs (.*)', content=dataset_file)
        return [*features_str.split(', '), *classes_str.split(', ')]

    def _extract_attribute(self, pattern: str, content: str) -> str:
        return re.search(pattern , content, re.IGNORECASE).group(1)

    def _get_dataset_path(self, dataset_name: str) -> str:
        return f"{self.DATASET_FOLDER}/{dataset_name}"

    def _read_file(self, file_path: str):
        with open(file_path) as f:
            return f.read()
