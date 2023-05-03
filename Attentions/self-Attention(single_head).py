class SelfAttention(nn.Module):
    def __init__(self, d_model, proj_values=True):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model) if proj_values else None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x) if self.value_proj is not None else x

        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores /= self.d_model ** 0.5

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        alphas = self.softmax(scores)
        attended = torch.matmul(alphas, V)

        return attended
