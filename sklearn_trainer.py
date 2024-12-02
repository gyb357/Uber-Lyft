import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SklearnTrainer():
    def __init__(
            self,
            df: pd.DataFrame,
            model: object
    ) -> None:
        self.df = df
        self.model = model
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def dataset_split(
            self,
            in_features: str,
            out_features: str,
            test_size: float = 0.2,
            random_state: int = 42
    ) -> None:
        x = self.df[in_features]
        y = self.df[out_features]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    def train(self) -> tuple:
        self.model.fit(self.x_train, self.y_train)
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)

        print('train score: ', self.model.score(self.x_train, self.y_train))
        print('test score: ', self.model.score(self.x_test, self.y_test))
        return y_train_pred, y_test_pred
    
    def visualize(self, y_train_pred, y_test_pred) -> None:
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_train, y_train_pred, c='blue', alpha=0.1, label='train')
        plt.scatter(self.y_test, y_test_pred, c='red', alpha=0.1, label='test')
        plt.plot([0, 60], [0, 60], c='black')
        plt.show()

