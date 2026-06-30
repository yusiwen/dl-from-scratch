"""
LSTM from scratch for sentiment classification.

Core component: LSTMCell — each gate is written explicitly:

    i_t = sigmoid(x·W_xi + h·W_hi + b_i)    ← input gate
    f_t = sigmoid(x·W_xf + h·W_hf + b_f)    ← forget gate
    g_t = tanh(  x·W_xg + h·W_hg + b_g)     ← cell candidate
    o_t = sigmoid(x·W_xo + h·W_ho + b_o)    ← output gate

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t         ← cell state (memory)
    h_t = o_t ⊙ tanh(c_t)                   ← hidden state

The gating mechanism solves the vanishing gradient problem of vanilla RNNs
by providing a direct gradient path through c_t (the "cell state highway").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMCell(nn.Module):
    """
    A single LSTM cell — implemented gate-by-gate.

    Each gate has its own weight matrix and bias:
      - input gate: i  (controls new information flow)
      - forget gate: f (controls memory retention)
      - cell gate:   g (candidate memory)
      - output gate: o (controls hidden state output)

    The cell state c_t is the key innovation: it carries information across
    time steps with only linear operations, allowing gradients to flow
    backward without vanishing (unlike vanilla RNNs).
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Combined weight matrices for all 4 gates.
        # input-to-hidden: (input_size, 4 * hidden_size)
        # hidden-to-hidden: (hidden_size, 4 * hidden_size)
        self.W_ih = nn.Parameter(torch.randn(input_size, 4 * hidden_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, 4 * hidden_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, state):
        """
        x: (batch, input_size)
        state: (h_prev, c_prev) — each (batch, hidden_size)

        Returns: (h_next, c_next)
        """
        h_prev, c_prev = state

        # Linear projections (same operation for all 4 gates).
        gates = x @ self.W_ih + h_prev @ self.W_hh + self.bias  # (batch, 4*hidden)

        # Split into 4 gates.
        i, f, g, o = gates.chunk(4, dim=1)

        # Gate activations.
        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        g = torch.tanh(g)     # cell candidate
        o = torch.sigmoid(o)  # output gate

        # Cell state update: forget + input.
        c_next = f * c_prev + i * g
        # Hidden state: output gate controls what to expose.
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    """
    Multi-layer LSTM using our hand-written LSTMCell.

    Processes a sequence one time step at a time, collecting all hidden
    states. The last layer's final hidden state is used for classification.
    """

    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.cells.append(LSTMCell(in_size, hidden_size))

    def forward(self, x, state=None):
        """
        x: (batch, seq_len, input_size)
        state: optional (h, c) tuple for each layer

        Returns:
          output: (batch, seq_len, hidden_size) — all hidden states of last layer
          (h_n, c_n): final states for each layer — each (num_layers, batch, hidden_size)
        """
        batch, seq_len, _ = x.shape

        if state is None:
            h = [torch.zeros(batch, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
            c = [torch.zeros(batch, self.hidden_size, device=x.device)
                 for _ in range(self.num_layers)]
        else:
            h, c = state

        # Store outputs for each layer.
        layer_outputs = [None] * self.num_layers

        for t in range(seq_len):
            inp = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](inp, (h[layer], c[layer]))
                inp = h[layer]
            layer_outputs[-1] = inp  # last layer's hidden state at time t

        # Stack all time steps of the last layer.
        # (Simplification: we store only the last time step's output to avoid
        #  paying O(seq_len * batch * hidden) memory for all states.)
        # For classification we only need the last time step anyway.
        output = h[-1].unsqueeze(1)  # (batch, 1, hidden)

        h_n = torch.stack(h, dim=0)  # (num_layers, batch, hidden)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)


class LSTMSentiment(nn.Module):
    """
    LSTM for sentiment classification.

    Architecture:
      Embedding(vocab→128) → LSTM(128→64) → FC(64→2)
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_size=64, num_layers=1,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = LSTM(embed_dim, hidden_size, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len) — 1=real, 0=padding

        Returns: (batch, num_classes) logits
        """
        emb = self.embedding(input_ids)  # (batch, seq_len, embed_dim)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        # Take the last layer's final hidden state.
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logits = self.classifier(out)
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
