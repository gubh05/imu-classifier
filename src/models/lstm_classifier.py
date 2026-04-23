import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LSTMBackbone(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=9,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, timesteps=128, channels=9) — sequence-first for LSTM
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # take last layer's hidden state: (batch, 128)
        return self.classifier(last_hidden)


class LSTMClassifier:
    """2-layer LSTM classifier for IMU windows.

    Input shape : (batch, timesteps=128, channels=9)  — sequence-first for LSTM
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
        self.model = _LSTMBackbone(num_classes).to(self.device)
        self.train_losses: list[float] = []
        self.train_accs: list[float] = []
        logger.info("LSTMClassifier using device: %s", self.device)

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
    ) -> "LSTMClassifier":
        """Train the LSTM.

        Parameters
        ----------
        X : np.ndarray, shape (N, 128, 9) — already in (N, timesteps, channels) order
        y : np.ndarray, shape (N,)
        X_val, y_val : optional validation split for checkpoint saving
        """
        X_t = torch.tensor(X, dtype=torch.float32)  # (N, 128, 9) — correct for LSTM
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

            if X_val is not None and y_val is not None:
                val_loss = self._val_loss(X_val, y_val, criterion)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(
                    "LSTM epoch %d/%d — loss=%.4f acc=%.4f val_loss=%.4f",
                    epoch + 1, self.epochs, avg_loss, acc, val_loss,
                )
            else:
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                logger.info(
                    "LSTM epoch %d/%d — loss=%.4f acc=%.4f",
                    epoch + 1, self.epochs, avg_loss, acc,
                )

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        return self

    def _val_loss(self, X_val: np.ndarray, y_val: np.ndarray, criterion: nn.Module) -> float:
        self.model.eval()
        X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_val, dtype=torch.long).to(self.device)
        with torch.no_grad():
            loss = criterion(self.model(X_t), y_t).item()
        self.model.train()
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions, shape (N,)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.argmax(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities, shape (N, num_classes)."""
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(X_t), dim=1)
        return probs.cpu().numpy()
