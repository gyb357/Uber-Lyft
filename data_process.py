import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def data_info(
        df: pd.DataFrame,
        head: bool = False,
        isnull: bool = False,
        shape: bool = False,
        columns: bool = False
) -> None:
    if head: print(f'head: {df.head()}')
    if isnull: print(f'isnull: {df.isnull().sum()}')
    if shape: print(f'shape: {df.shape}')
    if columns: print(f'columns: {df.columns}')


def data_visualize(
        df: pd.DataFrame,
        x: str,
        y: str,
        font_size: int = 10
) -> None:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, palette='Spectral')
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()


def remove_outlier(
        df: pd.DataFrame,
        column: str,
        q1: float = 0.2,
        q3: float = 0.8,
        iqr: float = 1.5
) -> pd.DataFrame:
    Q1 = df[column].quantile(q1)
    Q3 = df[column].quantile(q3)
    IQR = Q3 - Q1
    return df[~((df[column] < (Q1 - iqr*IQR)) | (df[column] > (Q3 + iqr*IQR)))]


class OneHotEncoder():
    def __init__(
        self,
        df: pd.DataFrame,
        columns: list
    ) -> None:
        self.df = df
        self.columns = columns

    def fit(self) -> pd.DataFrame:
        for column in self.columns:
            ont_hot = pd.get_dummies(self.df[column], prefix=column)
            self.df = pd.concat([self.df, ont_hot], axis=1)
            self.df.drop(column, axis=1, inplace=True)
        return self.df


def correlation_matrix(
        df: pd.DataFrame,
        column: list,
        font_size: int = 10
) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[column].corr(),
        annot=True,
        fmt='.1f'
    )
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()

