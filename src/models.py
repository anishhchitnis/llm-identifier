import torch
import torch.nn as nn
import torch.nn.functional as F

from common import PAD


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=200, n_filters=128, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, n_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), n_classes)

    def forward(self, x, lengths):
        e = self.embedding(x).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))
            positions = torch.arange(h.size(2), device=x.device).unsqueeze(0)
            valid = (lengths - conv.kernel_size[0] + 1).clamp(min=1).unsqueeze(1)
            h = h.masked_fill((positions >= valid).unsqueeze(1), 0.0)
            pooled.append(h.max(dim=2).values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb_dim=200, hidden=128, layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 4, n_classes)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)
        mean = (out * mask).sum(dim=1) / lengths.unsqueeze(1).to(out.dtype)
        maxed = out.masked_fill(~mask, -1e9).max(dim=1).values
        return self.fc(self.dropout(torch.cat([mean, maxed], dim=1)))
