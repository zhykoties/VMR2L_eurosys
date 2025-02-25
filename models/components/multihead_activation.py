from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from models.components.multihead import multi_head_attention_forward


class MultiheadAttention_Split(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> multihead_attn = nn.MultiheadAttentionSplit(embed_dim, num_heads, split_point)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, split_point, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention_Split, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.split_point = split_point
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight1 = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight1 = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight1 = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.q_proj_weight2 = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight2 = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight2 = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight1', None)
            self.register_parameter('in_proj_weight2', None)
        else:
            self.in_proj_weight1 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.in_proj_weight2 = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight1', None)
            self.register_parameter('k_proj_weight1', None)
            self.register_parameter('v_proj_weight1', None)
            self.register_parameter('q_proj_weight2', None)
            self.register_parameter('k_proj_weight2', None)
            self.register_parameter('v_proj_weight2', None)

        if bias:
            self.in_proj_bias1 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
            self.in_proj_bias2 = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias1', None)
            self.register_parameter('in_proj_bias2', None)
        self.out_proj1 = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj2 = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k1 = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v1 = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_k2 = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v2 = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k1 = self.bias_v1 = self.bias_k2 = self.bias_v2 = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight1)
            xavier_uniform_(self.in_proj_weight2)
        else:
            xavier_uniform_(self.q_proj_weight1)
            xavier_uniform_(self.k_proj_weight1)
            xavier_uniform_(self.v_proj_weight1)
            xavier_uniform_(self.q_proj_weight2)
            xavier_uniform_(self.k_proj_weight2)
            xavier_uniform_(self.v_proj_weight2)

        if self.in_proj_bias1 is not None:
            constant_(self.in_proj_bias1, 0.)
            constant_(self.out_proj1.bias, 0.)
            constant_(self.in_proj_bias2, 0.)
            constant_(self.out_proj2.bias, 0.)
        if self.bias_k1 is not None:
            xavier_normal_(self.bias_k1)
            xavier_normal_(self.bias_k2)
        if self.bias_v1 is not None:
            xavier_normal_(self.bias_v1)
            xavier_normal_(self.bias_v2)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention_Split, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, N, E_q)` when ``batch_first=False`` or :math:`(N, L, E_q)`
            when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is the batch size,
            and :math:`E_q` is the query embedding dimension ``embed_dim``. Queries are compared against
            key-value pairs to produce the output. See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, N, E_k)` when ``batch_first=False`` or :math:`(N, S, E_k)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_k` is the key embedding dimension ``kdim``. See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, N, E_v)` when ``batch_first=False`` or :math:`(N, S, E_v)` when
            ``batch_first=True``, where :math:`S` is the source sequence length, :math:`N` is the batch size, and
            :math:`E_v` is the value embedding dimension ``vdim``. See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). Binary and byte masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a byte mask, a non-zero value indicates that the corresponding ``key``
            value will be ignored.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary, byte, and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a byte mask, a non-zero value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, N, E)` when ``batch_first=False`` or
          :math:`(N, L, E)` when ``batch_first=True``, where :math:`L` is the target sequence length, :math:`N` is
          the batch size, and :math:`E` is the embedding dimension ``embed_dim``.
        - **attn_output_weights** - Attention output weights of shape :math:`(N, L, S)`, where :math:`N` is the batch
          size, :math:`L` is the target sequence length, and :math:`S` is the source sequence length. Only returned
          when ``need_weights=True``.
        """
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.split_point, self.embed_dim, self.num_heads,
                self.in_proj_weight1, self.in_proj_bias1,
                self.bias_k1, self.bias_v1, self.in_proj_weight2, self.in_proj_bias2,
                self.bias_k2, self.bias_v2, self.add_zero_attn,
                self.dropout, self.out_proj1.weight, self.out_proj1.bias,
                self.out_proj2.weight, self.out_proj2.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight1=self.q_proj_weight1, k_proj_weight1=self.k_proj_weight1,
                v_proj_weight1=self.v_proj_weight1, q_proj_weight2=self.q_proj_weight2,
                k_proj_weight2=self.k_proj_weight2, v_proj_weight2=self.v_proj_weight2)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.split_point, self.embed_dim, self.num_heads,
                self.in_proj_weight1, self.in_proj_bias1,
                self.bias_k1, self.bias_v1, self.in_proj_weight2, self.in_proj_bias2,
                self.bias_k2, self.bias_v2, self.add_zero_attn,
                self.dropout, self.out_proj1.weight, self.out_proj1.bias,
                self.out_proj2.weight, self.out_proj2.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
