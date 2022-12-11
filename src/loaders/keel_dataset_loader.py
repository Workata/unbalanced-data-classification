import pandas as pd
import re
from typing import List


class KeelDatasetLoader:
    """
    FORMAT:
    @...
    @...
    @inputs (headers)
    @outputs ()
    @data
    (csv data)
    """

    DATASET_FOLDER = "./datasets"

    def load(self, dataset_name: str):
        dataset_path = self._get_dataset_path(dataset_name)
        dataset_file = self._read_file(file_path=dataset_path)
        headers = self._get_dataset_headers(dataset_file)
        dataset_df = pd.read_csv(
            dataset_path,
            skiprows=lambda idx : dataset_file.split('\n')[idx].startswith("@"),
            names=headers,
            index_col=None
        )
        return dataset_df


    def _get_dataset_headers(self, dataset_file) -> List[str]:
        pattern = f'@inputs (.*)'
        headers_str = re.search(pattern, dataset_file, re.IGNORECASE).group(1)
        return [*headers_str.split(', '), 'class']

    def _get_dataset_path(self, dataset_name: str) -> str:
        return f"{self.DATASET_FOLDER}/{dataset_name}"

    def _read_file(self, file_path: str):
        with open(file_path) as f:
            return f.read()