import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


class MinMaxScaler():
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.min = None
        self.max = None

    def fit(self) -> pd.DataFrame:
        self.min = np.min(self.df, axis = 0)
        self.max = np.max(self.df, axis = 0)
        return (self.df - self.min)/(self.max - self.min)

    def inverse(self) -> pd.DataFrame:
        if self.min is None or self.max is None:
            Exception('Fit the data first')
        else: return self.df*(self.max - self.min) + self.min


class CustomDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            x: list,
            y: list,
    ) -> None:
        self.df = df
        self.x = self.df[x]
        self.y = self.df[y]

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple:
        x = torch.tensor(self.x.iloc[idx].values, dtype=torch.float32)
        y = torch.tensor(self.y.iloc[idx].values, dtype=torch.float32)
        return x, y

