"""
Logistic Regression on MNIST — single Linear layer + Softmax.

       input(784)  →  Linear(784→10)  →  Softmax  →  CrossEntropy

This is the simplest supervised classifier: no hidden layers, just a linear
decision boundary followed by softmax. The 10 weight vectors (one per digit)
can be visualized as 28×28 "templates" of what each digit looks like.

Logistic Regression is equivalent to a single-layer perceptron with softmax.
It serves as a baseline: if a deep model doesn't outperform this on MNIST,
something is wrong.

Expected accuracy: ~92%  (vs MLP's 97.9% — the ~6% gap comes from non-linearity)
"""

import numpy as np
from mlp.data import load_mnist, MNIST_CLASSES
from utils.config import load_config, save_config
from utils.seed import set_seed


class Linear:
    """
    Single Linear layer: y = x @ W + b.

    Same as the MLP version, included here so logistic_regression.py is
    self-contained and doesn't depend on mlp/model.py internals.
    """

    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return [(self.W, self.dW), (self.b, self.db)]


class SoftmaxCrossEntropy:
    """
    Softmax activation + Cross-Entropy loss (combined for numerical stability).

    dL/dlogit = (softmax - one_hot) / batch_size
    """

    def __init__(self):
        self.probs = None
        self.labels_one_hot = None

    def forward(self, logits, labels_one_hot):
        self.labels_one_hot = labels_one_hot
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        eps = 1e-8
        correct_log_probs = -np.log(self.probs[labels_one_hot.astype(bool)] + eps)
        return correct_log_probs.mean()

    def backward(self):
        batch_size = self.probs.shape[0]
        return (self.probs - self.labels_one_hot) / batch_size


def train():
    cfg = load_config("basics/logistic_regression.yaml")
    set_seed(cfg["seed"])

    train_loader, test_loader = load_mnist(batch_size=cfg["batch_size"])

    model = Linear(784, 10)
    loss_fn = SoftmaxCrossEntropy()
    lr = cfg["lr"]

    num_epochs = cfg["num_epochs"]
    print(f"Logistic Regression on MNIST ({784} -> 10, no hidden layers)")
    print(f"Parameters: {model.W.size + model.b.size:,}")
    print()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        for images, labels, labels_one_hot in train_loader():
            logits = model.forward(images)
            loss = loss_fn.forward(logits, labels_one_hot)
            epoch_loss += loss
            dout = loss_fn.backward()
            model.backward(dout)
            # Manual SGD step.
            for param, grad in model.params():
                param -= lr * grad
            num_batches += 1

        # Eval.
        correct = 0
        total = 0
        for images, labels, _ in test_loader():
            logits = model.forward(images)
            preds = np.argmax(logits, axis=1)
            correct += (preds == labels).sum()
            total += len(labels)
        accuracy = correct / total * 100
        print(f"Epoch [{epoch:2d}/{num_epochs}]  Loss: {epoch_loss/num_batches:.4f}  "
              f"Test Acc: {accuracy:.2f}%")

    # Save weights.
    save_path = "basics/logistic_regression.npz"
    np.savez(save_path, W=model.W, b=model.b)
    save_config(cfg, save_path.replace(".npz", "_config.yaml"))
    print(f"\nWeights saved to {save_path}")
    print(f"  W shape: {model.W.shape}, b shape: {model.b.shape}")

    # Per-class accuracy.
    print(f"\n{'Digit':<8} {'Accuracy':>8}")
    print("-" * 20)
    all_preds, all_labels = [], []
    for images, labels, _ in test_loader():
        logits = model.forward(images)
        all_preds.append(np.argmax(logits, axis=1))
        all_labels.append(labels)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    for i in range(10):
        mask = all_labels == i
        acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum() * 100
        print(f"{i:<8} {acc:>7.1f}%")
    print("-" * 20)
    print(f"{'All':<8} {(all_preds==all_labels).sum()/len(all_labels)*100:>7.1f}%")


if __name__ == "__main__":
    train()
