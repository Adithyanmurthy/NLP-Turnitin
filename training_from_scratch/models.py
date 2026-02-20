"""
Training From Scratch — Model Architectures
All three transformer models built from raw PyTorch — no pretrained weights.

Model 1: AIDetectorFromScratch       — Transformer Encoder (classification)
Model 2: PlagiarismDetectorFromScratch — Siamese Transformer (similarity)
Model 3: HumanizerFromScratch         — Encoder-Decoder Transformer (seq2seq)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
#  SHARED BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention from scratch."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv=None):
        B, T, _ = x.shape
        kv_source = kv if kv is not None else x
        S = kv_source.size(1)

        Q = self.W_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(kv_source).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(kv_source).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(context)


class FeedForward(nn.Module):
    """Position-wise FFN with GELU."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))



class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attn(x, mask=mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.ffn(x))
        return x


class TransformerDecoderLayer(nn.Module):
    """Pre-norm transformer decoder layer: self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attn(x, mask=tgt_mask))
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout2(self.cross_attn(x, mask=memory_mask, kv=encoder_output))
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout3(self.ffn(x))
        return x


def _init_weights(module, init_range=0.02):
    """Shared weight initialization."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


# ═══════════════════════════════════════════════════════════
#  MODEL 1: AI DETECTOR — Transformer Encoder + Classification Head
# ═══════════════════════════════════════════════════════════

class AIDetectorFromScratch(nn.Module):
    """
    Transformer encoder for AI text detection.
    Phase 1: Pre-train with MLM    Phase 2: Fine-tune with classification head
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        vs, hs = config["vocab_size"], config["hidden_size"]
        nl, nh = config["num_hidden_layers"], config["num_attention_heads"]
        ff, dp = config["intermediate_size"], config["hidden_dropout_prob"]
        mp, eps = config["max_position_embeddings"], config["layer_norm_eps"]

        self.token_embedding = nn.Embedding(vs, hs, padding_idx=0)
        self.position_encoding = PositionalEncoding(hs, mp, dp)
        self.embedding_norm = nn.LayerNorm(hs, eps=eps)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hs, nh, ff, dp, eps) for _ in range(nl)
        ])
        self.final_norm = nn.LayerNorm(hs, eps=eps)

        # MLM head (pre-training)
        self.mlm_head = nn.Sequential(
            nn.Linear(hs, hs), nn.GELU(), nn.LayerNorm(hs, eps=eps), nn.Linear(hs, vs),
        )
        # Classification head (fine-tuning)
        self.classifier = nn.Sequential(
            nn.Linear(hs, hs), nn.Tanh(), nn.Dropout(dp),
            nn.Linear(hs, config["num_labels"]),
        )
        self.apply(lambda m: _init_weights(m, config["initializer_range"]))

    def encode(self, input_ids, attention_mask=None):
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        x = self.embedding_norm(x)
        mask = attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)

    def forward_mlm(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        logits = self.mlm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]),
                                   labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.encode(input_ids, attention_mask)
        logits = self.classifier(hidden[:, 0, :])  # [CLS] token
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


# ═══════════════════════════════════════════════════════════
#  MODEL 2: PLAGIARISM DETECTOR — Siamese Transformer Encoder
# ═══════════════════════════════════════════════════════════

class PlagiarismDetectorFromScratch(nn.Module):
    """
    Siamese (shared-weight) transformer encoder for plagiarism detection.
    Phase 1: Pre-train with MLM    Phase 2: Fine-tune with contrastive loss
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        vs, hs = config["vocab_size"], config["hidden_size"]
        nl, nh = config["num_hidden_layers"], config["num_attention_heads"]
        ff, dp = config["intermediate_size"], config["hidden_dropout_prob"]
        mp, eps = config["max_position_embeddings"], config["layer_norm_eps"]

        self.token_embedding = nn.Embedding(vs, hs, padding_idx=0)
        self.position_encoding = PositionalEncoding(hs, mp, dp)
        self.embedding_norm = nn.LayerNorm(hs, eps=eps)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hs, nh, ff, dp, eps) for _ in range(nl)
        ])
        self.final_norm = nn.LayerNorm(hs, eps=eps)

        # MLM head (pre-training)
        self.mlm_head = nn.Sequential(
            nn.Linear(hs, hs), nn.GELU(), nn.LayerNorm(hs, eps=eps), nn.Linear(hs, vs),
        )
        # Projection for similarity (fine-tuning)
        self.projection = nn.Sequential(nn.Linear(hs, hs), nn.Tanh())
        self.apply(lambda m: _init_weights(m, config["initializer_range"]))

    def _encode_hidden(self, input_ids, attention_mask=None):
        """Run encoder, return full hidden states."""
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        x = self.embedding_norm(x)
        mask = attention_mask.unsqueeze(1).unsqueeze(2) if attention_mask is not None else None
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)

    def encode(self, input_ids, attention_mask=None):
        """Encode text into a fixed-size embedding (mean pooling + projection)."""
        x = self._encode_hidden(input_ids, attention_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        return self.projection(pooled)

    def forward_mlm(self, input_ids, attention_mask=None, labels=None):
        hidden = self._encode_hidden(input_ids, attention_mask)
        logits = self.mlm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]),
                                   labels.view(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits}

    def forward(self, input_ids_a, attention_mask_a,
                input_ids_b, attention_mask_b, labels=None):
        emb_a = self.encode(input_ids_a, attention_mask_a)
        emb_b = self.encode(input_ids_b, attention_mask_b)
        similarity = F.cosine_similarity(emb_a, emb_b, dim=-1)
        loss = None
        if labels is not None:
            if labels.dtype == torch.float:
                loss = F.mse_loss(similarity, labels)
            else:
                temp = self.config.get("contrastive_temperature", 0.05)
                logits_matrix = torch.matmul(emb_a, emb_b.T) / temp
                targets = torch.arange(len(emb_a), device=emb_a.device)
                loss = F.cross_entropy(logits_matrix, targets)
        return {"loss": loss, "similarity": similarity}


# ═══════════════════════════════════════════════════════════
#  MODEL 3: HUMANIZER — Full Encoder-Decoder Transformer
# ═══════════════════════════════════════════════════════════

class HumanizerFromScratch(nn.Module):
    """
    Encoder-decoder transformer for text humanization (seq2seq).
    Phase 1: Pre-train as denoising autoencoder
    Phase 2: Fine-tune on paraphrase pairs (AI text → human-like text)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        vs, dm = config["vocab_size"], config["d_model"]
        el, dl = config["encoder_layers"], config["decoder_layers"]
        nh, ff = config["num_attention_heads"], config["d_ff"]
        dp, mp = config["dropout"], config["max_position_embeddings"]
        eps = config["layer_norm_eps"]

        self.token_embedding = nn.Embedding(vs, dm, padding_idx=0)
        self.position_encoding = PositionalEncoding(dm, mp, dp)
        self.embedding_scale = math.sqrt(dm)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(dm, nh, ff, dp, eps) for _ in range(el)
        ])
        self.encoder_norm = nn.LayerNorm(dm, eps=eps)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(dm, nh, ff, dp, eps) for _ in range(dl)
        ])
        self.decoder_norm = nn.LayerNorm(dm, eps=eps)

        # Output projection (weight-tied with token embedding)
        self.output_projection = nn.Linear(dm, vs, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        self.apply(lambda m: _init_weights(m, config["initializer_range"]))

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return (mask == 0).unsqueeze(0).unsqueeze(0)

    def encode(self, src_ids, src_mask=None):
        x = self.token_embedding(src_ids) * self.embedding_scale
        x = self.position_encoding(x)
        mask = src_mask.unsqueeze(1).unsqueeze(2) if src_mask is not None else None
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        return self.encoder_norm(x)

    def decode(self, tgt_ids, encoder_output, tgt_mask=None, memory_mask=None):
        x = self.token_embedding(tgt_ids) * self.embedding_scale
        x = self.position_encoding(x)
        seq_len = tgt_ids.size(1)
        causal_mask = self._make_causal_mask(seq_len, tgt_ids.device)
        if tgt_mask is not None:
            combined_mask = causal_mask & tgt_mask.unsqueeze(1).unsqueeze(2)
        else:
            combined_mask = causal_mask
        mem_mask = memory_mask.unsqueeze(1).unsqueeze(2) if memory_mask is not None else None
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask=combined_mask, memory_mask=mem_mask)
        return self.decoder_norm(x)

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask=None, labels=None):
        encoder_output = self.encode(src_ids, src_mask)
        decoder_output = self.decode(tgt_ids, encoder_output, tgt_mask, src_mask)
        logits = self.output_projection(decoder_output)
        loss = None
        if labels is not None:
            ls = self.config.get("label_smoothing", 0.0)
            loss = F.cross_entropy(logits.view(-1, self.config["vocab_size"]),
                                   labels.view(-1), ignore_index=0,
                                   label_smoothing=ls)
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, src_ids, src_mask, max_length=256,
                 bos_token_id=5, eos_token_id=6):
        """Greedy autoregressive generation."""
        self.eval()
        B = src_ids.size(0)
        encoder_output = self.encode(src_ids, src_mask)
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long,
                               device=src_ids.device)
        for _ in range(max_length - 1):
            decoder_output = self.decode(generated, encoder_output, memory_mask=src_mask)
            next_logits = self.output_projection(decoder_output[:, -1, :])
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_token_id).all():
                break
        return generated
