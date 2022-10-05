import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.embed_dim = d_model
        self.nhead = nhead
        self.proj_qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, ious):
        assert query.size(-1) == self.embed_dim
        assert torch.equal(query, key) and torch.equal(key, value)

        tgt_len, bsz, embed_dim = query.size()

        head_dim = embed_dim // self.nhead
        assert head_dim * self.nhead == self.embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim) ** -0.5

        q, k, v = self.proj_qkv(query).chunk(3, dim=-1)
        q = q * scaling
        q = q.contiguous().view(tgt_len, bsz * self.nhead, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.nhead, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.nhead, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.nhead, tgt_len, src_len]

        attn_output_weights = torch.exp(attn_output_weights)
        attn_output_weights = attn_output_weights.contiguous().view(bsz, self.nhead, tgt_len, src_len)
        ious = ious.unsqueeze(1).repeat(1, self.nhead, 1, 1)
        attn_output_weights = attn_output_weights * ious
        attn_output_weights = attn_output_weights.contiguous().view(bsz * self.nhead, tgt_len, src_len)
        attn_output_weights = attn_output_weights / torch.sum(attn_output_weights, -1, keepdim=True)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.nhead, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output