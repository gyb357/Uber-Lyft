# %%
# import library
from model import Model
import torch


# %%
# deep learning model hyperparameters
in_features = 14
hidden_features = [512, 512, 256, 256]
out_features = 1
batch_norm = None
dropout = 0.0
init_weights = True


# %%
# model instance
model = Model(
    in_features=in_features,
    hidden_features=hidden_features,
    out_features=out_features,
    batch_norm=batch_norm,
    dropout=dropout,
    init_weights=init_weights
)


# %%
# model load
model.load_state_dict(torch.load('model/model_14.pth'))


# %%
# model evaluation
model.eval()

input = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.11, 1], dtype=torch.float32).unsqueeze(0)
output = model(input)
print(output, output.shape)

# %%

