"""Core neural modules for TPA-VLA.

The public implementation intentionally focuses on the method-specific modules:
QueryModule, the continuous ActionExpert, and the proprioceptive projector.
They operate on VLM hidden states with shape [batch, layers, tokens, hidden].
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .constants import ACTION_CHUNK_SIZE, ACTION_DIM, DEFAULT_TASK_TOKENS, PROPRIO_DIM


class QueryModule(nn.Module):
    """Lightweight task-specific module used in Phase II and inference.

    TPA-VLA keeps the VLM and ActionExpert shared at deployment time. Each task
    owns only a QueryModule, which maps frozen VLM hidden states into the feature
    distribution expected by the shared expert.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.self_attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=input_dim,
                    nhead=num_heads,
                    dim_feedforward=input_dim * 4,
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_transformer_layers)
            ]
        )
        self.output_projection = nn.Linear(input_dim, self.output_dim) if self.output_dim != input_dim else nn.Identity()
        self.final_norm = nn.LayerNorm(self.output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise ValueError(f"Expected [B, L, T, D] hidden states, got shape {tuple(hidden_states.shape)}")
        batch, num_layers, seq_len, hidden_dim = hidden_states.shape
        x = hidden_states.reshape(batch, num_layers * seq_len, hidden_dim)
        for layer in self.self_attention_layers:
            x = layer(x)
        x = x.reshape(batch, num_layers, seq_len, hidden_dim)
        return self.final_norm(self.output_projection(x))


class ProprioProjector(nn.Module):
    """Project proprioceptive state into the VLM hidden dimension."""

    def __init__(self, llm_dim: int, proprio_dim: int = PROPRIO_DIM) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(proprio_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.model(proprio)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_even, q_odd = q[..., ::2], q[..., 1::2]
    k_even, k_odd = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack((-q_odd, q_even), dim=-1).reshape_as(q)
    k_rot = torch.stack((-k_odd, k_even), dim=-1).reshape_as(k)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


class _RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class _ExpertBlock(nn.Module):
    """Conditioned MLP block used by the ActionExpert."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())
        self.q_proj = nn.Linear(dim, dim)
        self.k_self = nn.Linear(dim, dim)
        self.v_self = nn.Linear(dim, dim)
        self.k_action = nn.Linear(dim, dim)
        self.v_action = nn.Linear(dim, dim)
        self.k_task = nn.Linear(dim, dim)
        self.v_task = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.gating_factor = nn.Parameter(torch.zeros(1))
        self.rope = _RotaryPositionEmbedding(self.head_dim)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, action_context: torch.Tensor, task_context: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        action_len = action_context.shape[1]
        task_len = task_context.shape[1]

        q = self._reshape_heads(self.q_proj(x))
        k_self = self._reshape_heads(self.k_self(x))
        v_self = self._reshape_heads(self.v_self(x))
        k_action = self._reshape_heads(self.k_action(action_context))
        v_action = self._reshape_heads(self.v_action(action_context))
        k_task = self._reshape_heads(self.k_task(task_context))
        v_task = self._reshape_heads(self.v_task(task_context))

        cos, sin = self.rope(seq_len, x.device, x.dtype)
        q, k_self = _apply_rope(q, k_self, cos, sin)
        _, k_action = _apply_rope(k_action, k_action, *self.rope(action_len, x.device, x.dtype))
        _, k_task = _apply_rope(k_task, k_task, *self.rope(task_len, x.device, x.dtype))

        scores = [
            torch.matmul(q, k_self.transpose(-2, -1)),
            torch.matmul(q, k_action.transpose(-2, -1)),
            torch.matmul(q, k_task.transpose(-2, -1)) * torch.tanh(self.gating_factor),
        ]
        attn = torch.softmax(torch.cat(scores, dim=-1) / math.sqrt(self.head_dim), dim=-1)
        values = torch.cat([v_self, v_action, v_task], dim=2)
        out = torch.matmul(attn, values).transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.ffn(self.o_proj(out) + x)


class ActionExpert(nn.Module):
    """Continuous action-chunk decoder used as the shared Expert G."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        action_dim: int = ACTION_DIM,
        action_chunk_size: int = ACTION_CHUNK_SIZE,
        num_task_tokens: int = DEFAULT_TASK_TOKENS,
        num_blocks: int = 24,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_task_tokens = num_task_tokens
        self.layer_norm_in = nn.LayerNorm(input_dim * action_dim)
        self.fc_in = nn.Linear(input_dim * action_dim, hidden_dim)
        self.blocks = nn.ModuleList([_ExpertBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm_out = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

    def predict_action(
        self,
        hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: ProprioProjector,
    ) -> torch.Tensor:
        batch = hidden_states.shape[0]
        task_hidden = hidden_states[:, :, : self.num_task_tokens, :]
        action_hidden = hidden_states[:, :, self.num_task_tokens :, :]
        proprio_hidden = proprio_projector(proprio.reshape(batch, -1).to(hidden_states.dtype)).unsqueeze(1)

        x = torch.zeros(
            batch,
            self.action_chunk_size,
            self.action_dim * self.input_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        x = torch.relu(self.fc_in(self.layer_norm_in(x)))
        for idx, block in enumerate(self.blocks):
            action_context = torch.cat([action_hidden[:, idx + 1, :, :], proprio_hidden], dim=1)
            task_context = task_hidden[:, idx + 1, :, :]
            x = block(x, action_context=action_context, task_context=task_context)
        return self.fc_out(self.layer_norm_out(x))

    forward = predict_action


class QueryWrappedExpert(nn.Module):
    """Inference wrapper for F -> Q_i -> G."""

    def __init__(self, query_module: QueryModule, expert: ActionExpert) -> None:
        super().__init__()
        self.query_module = query_module
        self.expert = expert

    def predict_action(
        self,
        hidden_states: torch.Tensor,
        proprio: torch.Tensor,
        proprio_projector: ProprioProjector,
    ) -> torch.Tensor:
        return self.expert.predict_action(self.query_module(hidden_states), proprio, proprio_projector)

    forward = predict_action
