import numpy as np


class Linear:
    """
    Fully-connected (dense) layer: y = x @ W + b

    This is the fundamental building block of MLPs. Each output neuron
    computes a weighted sum of all inputs plus a bias term.

    Weight initialization uses He initialization (sqrt(2/in_dim)), which
    is designed for ReLU activations. It prevents activations from exploding
    or vanishing in deep networks by keeping the variance of layer outputs
    roughly constant.
    """

    def __init__(self, in_dim, out_dim):
        scale = np.sqrt(2.0 / in_dim)
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim, dtype=np.float32)
        # Cached for backward pass.
        self.x = None
        # Gradient buffers.
        self.dW = None
        self.db = None

    def forward(self, x):
        """Compute y = x @ W + b. Cache x for backward."""
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        """
        Backpropagate gradients through this layer.

        Chain rule:
          dL/dW = x^T @ dL/dy    (shape: in_dim x out_dim)
          dL/db = sum(dL/dy)     (shape: out_dim,)
          dL/dx = dL/dy @ W^T    (shape: batch x in_dim)

        Why sum for db? The bias is broadcast across the batch dimension,
        so its gradient is the sum of the incoming gradients over the batch.
        """
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        dx = dout @ self.W.T
        return dx

    def params(self):
        """Return (weight, weight_grad, bias, bias_grad) for the optimizer."""
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    """
    Rectified Linear Unit activation: f(x) = max(0, x)

    Why ReLU over sigmoid/tanh?
      - Non-saturating: gradients don't shrink for large positive inputs.
      - Sparse activation: only a subset of neurons fire at once.
      - Efficient: simple threshold operation, no expensive exponentials.
      - Helps mitigate vanishing gradient problem in deeper networks.
    """

    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        Forward: f(x) = max(0, x).
        Cache the binary mask for backward.
        """
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        """
        Backward: gradient passes through unchanged where x > 0, else 0.
        Chain rule: dL/dx = dL/df * df/dx, where df/dx = 1 if x > 0 else 0.
        """
        return dout * self.mask

    def params(self):
        return []


class SoftmaxCrossEntropy:
    """
    Softmax activation + Cross-Entropy loss in one object.

    Softmax converts logits to probabilities:
      p_i = exp(logit_i) / sum_j exp(logit_j)

    Cross-entropy loss measures the dissimilarity between predicted
    probabilities and true labels:
      L = -sum_i y_i * log(p_i)

    We combine them for numerical stability: computing softmax then log
    separately can cause log(0) = -inf. Instead, we use the identity:
      log(softmax) = logits - log(sum(exp(logits)))
    with the max-subtraction trick to prevent exp overflow.

    The combined gradient is elegantly simple:
      dL/dlogits = (softmax - one_hot) / batch_size
    """

    def __init__(self):
        self.probs = None
        self.labels_one_hot = None
        self.loss_value = None

    def forward(self, logits, labels_one_hot):
        """
        Forward pass: compute softmax probabilities and cross-entropy loss.

        logits: (batch, num_classes) raw scores from the last linear layer.
        labels_one_hot: (batch, num_classes) one-hot encoded ground truth.

        Numerical stability trick: subtract max logit from each row.
        This doesn't change softmax output (since exp(x-c)/sum(exp(x-c))
        = exp(x)/sum(exp(x))) but prevents exp(large_value) = inf.
        """
        self.labels_one_hot = labels_one_hot

        # Subtract max for numerical stability.
        logits_shifted = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        self.probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy loss = -mean(log(probs[label])).
        # Add epsilon inside log to avoid log(0) = -inf.
        eps = 1e-8
        batch_size = logits.shape[0]
        correct_log_probs = -np.log(self.probs[labels_one_hot.astype(bool)] + eps)
        self.loss_value = correct_log_probs.mean()

        return self.loss_value

    def backward(self):
        """
        Backward pass: dL/dlogits = (probs - one_hot) / batch_size.

        This comes from:
          dL/dp_i = -y_i / p_i           (derivative of cross-entropy)
          dp_i/dlogit_j = p_i * (delta_ij - p_j)  (derivative of softmax)
        Combined: dL/dlogit = p - y       (after summing over classes)

        Dividing by batch_size normalizes the gradient magnitude across
        batches of different sizes.
        """
        batch_size = self.probs.shape[0]
        return (self.probs - self.labels_one_hot) / batch_size

    def params(self):
        return []


class MLP:
    """
    Multi-Layer Perceptron with configurable hidden layer sizes.

    Architecture:
      Linear(784 -> 256) -> ReLU -> Linear(256 -> 128) -> ReLU -> Linear(128 -> 10)

    This is the simplest deep learning model: a stack of fully-connected layers
    with non-linear activations between them. The Universal Approximation Theorem
    states that a 2-layer MLP (one hidden layer) with enough neurons can
    approximate any continuous function arbitrarily well.

    We use 3 layers here (784→256→128→10) for better representational capacity.
    """

    def __init__(self, layer_sizes=None):
        if layer_sizes is None:
            layer_sizes = [784, 256, 128, 10]
        self.layers = []
        for i in range(len(layer_sizes) - 2):
            self.layers.append(Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(ReLU())
        # Final layer: no activation (SoftmaxCrossEntropy handles it).
        self.layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        """Sequential forward pass through all layers."""
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, logits, labels_one_hot):
        """Compute SoftmaxCrossEntropy loss."""
        return self.loss_fn.forward(logits, labels_one_hot)

    def backward(self, labels_one_hot):
        """
        Full backward pass through loss + all layers.

        This is the core of backpropagation: we compute gradients from
        the loss backwards through the network, applying the chain rule
        at each step.
        """
        # Start with loss gradient.
        dout = self.loss_fn.backward()
        # Propagate backward through all layers in reverse order.
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self):
        """Collect all trainable parameters and their gradients."""
        result = []
        for layer in self.layers:
            result.extend(layer.params())
        return result


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    The simplest optimization algorithm: move parameters in the opposite
    direction of the gradient, scaled by the learning rate.

      W = W - lr * dW
      b = b - lr * db

    No momentum, no adaptive learning rates — just the raw gradient step.
    This is the baseline against which all advanced optimizers (Adam, RMSprop)
    are compared.
    """

    def __init__(self, model, lr=0.1):
        self.model = model
        self.lr = lr

    def step(self):
        """Apply gradient descent update to all parameters."""
        for param, grad in self.model.params():
            param -= self.lr * grad

    def set_lr(self, lr):
        self.lr = lr
