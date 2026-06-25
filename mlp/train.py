import numpy as np

from mlp.data import load_mnist, MNIST_CLASSES
from mlp.model import MLP, SGD


def train():
    np.random.seed(42)

    train_loader, test_loader = load_mnist(batch_size=64)

    # MLP with 784 -> 256 -> 128 -> 10.
    model = MLP([784, 256, 128, 10])
    optimizer = SGD(model, lr=0.1)

    num_epochs = 20
    total_params = sum(p.size for p, _ in model.params())
    print(f"Model parameters: {total_params:,}")

    for epoch in range(1, num_epochs + 1):
        # --- Train ---
        epoch_loss = 0.0
        num_batches = 0
        for images, labels, labels_one_hot in train_loader():
            # Forward pass.
            logits = model.forward(images)
            loss = model.compute_loss(logits, labels_one_hot)
            epoch_loss += loss

            # Backward pass (manual backpropagation, no autograd).
            model.backward(labels_one_hot)

            # Gradient descent update.
            optimizer.step()

            num_batches += 1

        avg_loss = epoch_loss / num_batches

        # --- Evaluate ---
        correct = 0
        total = 0
        for images, labels, _ in test_loader():
            logits = model.forward(images)
            predictions = np.argmax(logits, axis=1)
            correct += (predictions == labels).sum()
            total += len(labels)

        accuracy = correct / total * 100
        print(f"Epoch [{epoch:2d}/{num_epochs}]  "
              f"Train Loss: {avg_loss:.4f}  Test Acc: {accuracy:.2f}%  "
              f"LR: {optimizer.lr:.2e}")

        # LR schedule: decay by 10x at epochs 5, 10, 15.
        if epoch in (5, 10, 15):
            optimizer.set_lr(optimizer.lr / 10)

    print("\nTraining complete!")

    # Save model weights as numpy arrays.
    weights = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W'):
            weights[f"layer_{i}_W"] = layer.W
            weights[f"layer_{i}_b"] = layer.b
    np.savez("mlp/mlp_mnist.npz", **weights)
    print("Weights saved to mlp/mlp_mnist.npz")


if __name__ == "__main__":
    train()
