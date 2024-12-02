# %%
# import library
# data preprocessing
import pandas as pd
from data_process import data_info, data_visualize, remove_outlier, OneHotEncoder, correlation_matrix

# machine learning
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn_trainer import SklearnTrainer

# deep learning
import torch
from dataset import MinMaxScaler, CustomDataset
from torch.utils.data import DataLoader
from model import Model
from train import Trainer

# file system
import os


# %%
# data load (uber & lyft dataset)
# https://www.kaggle.com/brllrb/uber-and-lyft-dataset-boston-ma
df = pd.read_csv('dataset/rideshare_kaggle.csv')
data_info(df, isnull=True, shape=True)


# %%
# remove null value
df.dropna(axis=0, inplace=True)
df.reset_index()
data_info(df, isnull=True, shape=True)


# %%
# data visualization
data_visualize(df, 'name', 'price')
data_visualize(df, 'icon', 'price')


# %%
# remove outlier
df = remove_outlier(df, 'price')
df.reset_index()
data_visualize(df, 'name', 'price')
data_visualize(df, 'icon', 'price')


# %%
# checking unique value
str_columns = [
    'id',
    'datetime',
    'timezone',
    'source',
    'destination',
    'cab_type',
    'product_id',
    'name',
    'short_summary',
    'long_summary',
    'icon',
]

for column in str_columns:
    unique = df[column].nunique()
    print(column, ': ', unique)


# %%
# drop unnecessary columns
df.drop(['id', 'product_id', 'datetime', 'timezone'], axis=1, inplace=True)
df.reset_index()
data_info(df, shape=True, columns=True)


# %%
# one-hot encoding
onehot_encoder = OneHotEncoder(
    df=df,
    columns=[
        'source',
        'destination',
        'cab_type',
        'name',
        'short_summary',
        'long_summary',
        'icon',
    ]
)
df = onehot_encoder.fit()
data_info(df, shape=True, columns=True)


# %%
# correlation matrix
# 14
name_columns = [
    'name_Black',
    'name_Black SUV',
    'name_Lux',
    'name_Lux Black',
    'name_Lux Black XL',
    'name_Lyft',
    'name_Lyft XL',
    'name_Shared',
    'name_UberPool',
    'name_UberX',
    'name_UberXL',
    'name_WAV',
    'distance',
    'surge_multiplier',
    'price',
]
# 9
icon_columns = [
    'icon_ clear-day ',
    'icon_ clear-night ',
    'icon_ cloudy ',
    'icon_ fog ',
    'icon_ partly-cloudy-day ',
    'icon_ partly-cloudy-night ',
    'icon_ rain ',
    'distance',
    'surge_multiplier',
    'price',
]

correlation_matrix(df, name_columns)
correlation_matrix(df, icon_columns)


# %%
# machine learning models
linear_regression = SklearnTrainer(df, LinearRegression())
decision_tree = SklearnTrainer(df, DecisionTreeRegressor())
random_forest = SklearnTrainer(df, RandomForestRegressor())
gradient_boosting = SklearnTrainer(df, GradientBoostingRegressor())


# %%
# machine learning datasets
datasets = [name_columns, icon_columns]
models = [linear_regression, decision_tree, random_forest, gradient_boosting]
model_names = ['linear_regression', 'decision_tree', 'random_forest', 'gradient_boosting']


# %%
# machine learning training
# for dataset in datasets:
#     for model in models:
#         model.dataset_split(
#             in_features=dataset[:-1],
#             out_features=dataset[-1]
#         )

#         print(model_names[models.index(model)])
#         y_train_pred, y_test_pred = model.train()

#         model.visualize(y_train_pred, y_test_pred)


# %%
# bool 2 float
df = df.astype(float)
df.reset_index()
data_info(df, shape=True, columns=True)


# %%
# Min-Max scaler
# scaler = MinMaxScaler(df)
# df = scaler.fit()


# %%
# deep learning hyperparameters

# dataset length parameters
df_len = len(df)
train = 0.8
valid = 0.1

# label columns
# x = name_columns[:-1]
x = icon_columns[:-1]
y = ['price']
print(f'x: {len(x)}, y: {len(y)}')

# data loader parameters
batch_size = int(df_len*0.01)
shuffle = True
pin_memory = True
num_workers = 0

# model parameters
in_features = len(x)
hidden_features = [512, 512, 256, 256]
out_features = len(y)
batch_norm = None
dropout = 0.0
init_weights = True

# trainer parameters
criterion = torch.nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 100
lr = 0.001

# %%
# deep learning dataset length
train_len = int(df_len*train)
valid_len = int(df_len*valid)
test_len = df_len - train_len - valid_len
print('train_len:', train_len)
print('valid_len:', valid_len)
print('test_len:', test_len)


# %%
# deep learning dataset
train_dataset = CustomDataset(df[:train_len], x, y)
valid_dataset = CustomDataset(df[train_len:train_len+valid_len], x, y)
test_dataset = CustomDataset(df[train_len+valid_len:], x, y)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    pin_memory=pin_memory,
    num_workers=num_workers
)
valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    pin_memory=pin_memory,
    num_workers=num_workers
)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    pin_memory=pin_memory,
    num_workers=num_workers
)


# %%
# deep learning model and trainer
model = Model(
    in_features=in_features,
    hidden_features=hidden_features,
    out_features=out_features,
    batch_norm=batch_norm,
    dropout=dropout,
    init_weights=init_weights
).to(device)

trainer = Trainer(
    model=model,
    criterion=criterion,
    device=device,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    epochs=epoch,
    lr=lr,
    grad_steps=1
)


# %%
# deep learning training
trainer.train()


# %%
# model evaluation
trainer.eval(test_dataloader)


# %%
# model save
if not os.path.exists('model'):
    os.makedirs('model')

torch.save(model.state_dict(), 'model/model.pth')


# %%

