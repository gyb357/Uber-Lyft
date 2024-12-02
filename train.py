from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler


def mae(output: Tensor, target: Tensor) -> Tensor:
    return (output - target).abs().mean()


class Trainer():
    columns = ['epoch', 'train_loss', 'train_metric', 'valid_loss', 'valid_metric']

    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            device: str,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            epochs: int = 100,
            lr: float = 1e-3,
            grad_steps: int = 1,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.epochs = epochs
        self.lr = lr
        self.grad_steps = grad_steps

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=5)
        self.tensorboard = SummaryWriter()

    def eval(self, data_loader: DataLoader) -> tuple:
        self.model.eval()
        loss, metric = 0.0, 0.0

        with torch.no_grad():
            for inputs, outputs in tqdm(data_loader):
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)

                self.optimizer.zero_grad()

                if self.device == 'cuda':
                    with autocast():
                        preds = self.model(inputs)
                        loss += self.criterion(preds, outputs).item()
                        metric += mae(preds, outputs).item()
                else:
                    preds = self.model(inputs)
                    loss += self.criterion(preds, outputs).item()
                    metric += mae(preds, outputs).item()

            loss /= len(data_loader)
            metric /= len(data_loader)
        return loss, metric
    
    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss, train_metric, val_loss, val_metric = 0.0, 0.0, 0.0, 0.0

            for i, (inputs, outputs) in enumerate(tqdm(self.train_dataloader), start=1):
                inputs, outputs = inputs.to(self.device), outputs.to(self.device)

                if self.device == 'cuda':
                    with autocast():
                        preds = self.model(inputs)

                        loss = self.criterion(preds, outputs)
                        loss_item = loss.item()
                        train_loss += loss_item
                        train_metric += mae(preds, outputs).item()
                else:
                    preds = self.model(inputs)

                    loss = self.criterion(preds, outputs)
                    loss_item = loss.item()
                    train_loss += loss_item
                    train_metric += mae(preds, outputs).item()

                self.scaler.scale(loss).backward()

                if i % self.grad_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

            train_loss /= len(self.train_dataloader)
            train_metric /= len(self.train_dataloader)
            print(f'epoch: {epoch}/{self.epochs}, train_loss: {train_loss}, train_metric: {train_metric}')

            val_loss, val_metric = self.eval(self.valid_dataloader)
            self.scheduler.step(val_loss)
            print(f'epoch: {epoch}/{self.epochs}, valid_loss: {val_loss}, valid_metric: {val_metric}')

            values = [epoch, train_loss, train_metric, val_loss, val_metric]
            for i in range(1, len(self.columns)):
                    self.tensorboard.add_scalar(self.columns[i], values[i], epoch)

        self.tensorboard.close()

