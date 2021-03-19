import torch
from torch import Tensor

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
  temp = query.bmm(key.transpose(1, 2))
  scale = query.size(-1) ** 0.5
  softmax = torch.nn.functional.softmax(temp / scale, dim=-1)
  return softmax.bmm(value)

class AttentionHead(torch.nn.Module):
  def __init__(self, dim_in: int, dim_k: int, dim_v: int):
    super().__init__()
    self.q = torch.nn.Linear(dim_in, dim_k)
    self.k = torch.nn.Linear(dim_in, dim_k)
    self.v = torch.nn.Linear(dim_in, dim_v)

  def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

class MultiHeadAttention(torch.nn.Module):
  def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
    super().__init__()
    self.heads = torch.nn.ModuleList(
      [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
    )
    self.linear = torch.nn.Linear(num_heads * dim_v, dim_in)

  def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    return self.linear(torch.cat([h(query, key, value) for h in self.heads], dim=-1))

def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cuda")) -> Tensor:
  pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
  dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
  phase = pos / 1e4 ** (dim / dim_model)

  return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> torch.nn.Module:
  return torch.nn.Sequential(
    torch.nn.Linear(dim_input, dim_feedforward),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_feedforward, dim_input)
  )

class Residual(torch.nn.Module):
  def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
    super().__init__()
    self.sublayer = sublayer
    self.norm = torch.nn.LayerNorm(dimension)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, *tensors: Tensor):
    return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

class TransformerEncoderLayer(torch.nn.Module):
  def __init__(self, dim_model: int = 512, num_heads: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
    super().__init__()
    dim_k = dim_v = dim_model // num_heads
    self.attention = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.feedforward = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )

  def forward(self, src: Tensor) -> Tensor:
    src = self.attention(src, src, src)
    return self.feedforward(src)

class TransformerEncoder(torch.nn.Module):
  def __init__(self, num_layers: int = 6, dim_model: int = 512, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
    super().__init__()
    self.layers = torch.nn.ModuleList([
      TransformerEncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
      for _ in range(num_layers)
    ])

  def forward(self, src: Tensor) -> Tensor:
    seq_len, dimension = src.size(1), src.size(2)
    src += position_encoding(seq_len, dimension)
    for layer in self.layers:
      src = layer(src)

    return src

class TransformerDecoderLayer(torch.nn.Module):
  def __init__(self, dim_model: int = 512, num_heads: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1):
    super().__init__()
    dim_k = dim_v = dim_model // num_heads
    self.attention_1 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.attention_2 = Residual(
      MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
      dimension = dim_model,
      dropout = dropout
    )
    self.feed_forward = Residual(
      feed_forward(dim_model, dim_feedforward),
      dimension = dim_model,
      dropout = dropout
    )

  def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    tgt = self.attention_1(tgt, tgt, tgt)
    tgt = self.attention_2(memory, memory, tgt)
    return self.feed_forward(tgt)

class TransformerDecoder(torch.nn.Module):
  def __init__(self, num_layers: int = 6, dim_model: int = 512, num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
    super().__init__()
    self.layers = torch.nn.ModuleList([
      TransformerDecoderLayer(dim_model, num_heads, dim_feedforward, dropout)
      for _ in range(num_layers)
    ])
    self.linear = torch.nn.Linear(dim_model, dim_model)

  def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    seq_len, dimension = tgt.size(1), tgt.size(2)
    tgt += position_encoding(seq_len, dimension)
    for layer in self.layers:
      tgt = layer(tgt, memory)

    return torch.softmax(self.linear(tgt), dim=-1)

class Transformer(torch.nn.Module):
  def __init__(self, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_model: int = 512, num_heads: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: torch.nn.Module = torch.nn.ReLU()):
    super().__init__()
    self.encoder = TransformerEncoder(
      num_layers = num_encoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout
    )
    self.decoder = TransformerDecoder(
      num_layers = num_decoder_layers,
      dim_model = dim_model,
      num_heads = num_heads,
      dim_feedforward = dim_feedforward,
      dropout = dropout
    )

  def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
    return self.decoder(tgt, self.encoder(src))