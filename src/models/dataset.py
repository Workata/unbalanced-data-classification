from dataclasses import dataclass
from pandas import DataFrame
from typing import Union
from numpy import ndarray


@dataclass
class Dataset:
    name: str
    data: Union[DataFrame, ndarray]
