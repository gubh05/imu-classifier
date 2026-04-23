import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _CNNBackbone(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(9, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels=9, timesteps=128)
        x = self.conv_block(x)
        return self.classifier(x)


class CNNClassifier:
    """1D-CNN classifier for IMU windows.

    Input shape : (batch, channels=9, timesteps=128)  — channels-first for Conv1d
    Output      : logits of shape (batch, 6)
    """

    def __init__(
        self,
        num_classes: int = 6,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = DEVICE
        self.model = _CNNBackbone(num_classes).to(self.device)
        self.train_losses: list[float] = []
        self.train_accs: list[float] = []
        logger.info("CNNClassifier using device: %s", self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Raw forward pass (for testing / inference on a single batch)."""
        return self.model(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "CNNClassifier":
        """Train the CNN.

        Parameters
        ----------
        X : np.ndarray, shape (N, 128, 9) — raw preprocessed windows (time-first from loader)
        y : np.ndarray, shape (N,)
        X_val, y_val : optional validation split for checkpoint saving
        """
        # CNN expects (batch, channels, timesteps) — transpose from (N, T, C)
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)  # (N, 9, 128)
        y_t = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state: dict | None = None

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(yb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(yb)

            avg_loss = epoch_loss / total
            acc = correct / total
            self.train_losses.append(avg_loss)
            self.train_accs.append(acc)

            # Validation checkpoint
            if X_val is not None and y_val is not None:
                val_loss = self._val_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(
                    "CNN epoch %d/%d — loss=%.4f acc=%.4f val_loss=%.4f",
                    epoch + 1, self.epochs, avg_loss, acc, val_loss,
                )
            else:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(
                    "CNN epoch %d/%d — loss=%.4f acc=%.4f",
                    epoch + 1, self.epochs, avg_loss, acc,
                )

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def _val_loss(self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()
        X_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        y_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        with torch.no_grad():
            loss = criterion(self.model(X_t), y_t).item()
        self.model.train()
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions, shape (N,)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (N, num_classes)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1)
        return probs.cpu().numpy()
