
from dataclasses import dataclass

from .model_keras import GPT, GPTConfig, MLP
from keras import ops
from keras.layers import Layer, Add, LayerNormalization, Dense, Dropout, Softmax, Embedding

import math

class NonCausalSelfAttention(Layer):
    def __init__(self, config):
        super(NonCausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.key_dim = config.n_embd // config.n_head
        self.sqrt_key_dim = math.sqrt(self.key_dim)

        # key, query, value projections for all heads
        self.c_attn = Dense(3 * config.n_embd, use_bias=config.bias)

        # output projection
        self.c_proj = Dense(config.n_embd, use_bias=config.bias)

        # Regularization
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

    def call(self, x):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        combined = self.c_attn(x)  # shape: (B, T, 3*C)
        q, k, v = ops.split(combined.numpy(), 3, axis=2)
        q = ops.reshape(q, (B, self.n_head, T, self.key_dim))
        k = ops.reshape(k, (B, self.n_head, T, self.key_dim))
        v = ops.reshape(v, (B, self.n_head, T, self.key_dim))

        q = ops.transpose(q, (0, 2, 1, 3))  # Permute for attention computation
        k = ops.transpose(k, (0, 2, 1, 3))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Scaled dot product attention
        att = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) / self.sqrt_key_dim
        att = Softmax(axis=-1)(att)
        att = self.attn_dropout(att)

        y = ops.matmul(att, v)  # shape: (B, T, nh, hs)
        y = ops.transpose(y, (0, 2, 1, 3))
        y = ops.reshape(y, (B, T, C))  # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FineBlock(Layer):
    def __init__(self, config):
        super(FineBlock, self).__init__()
        self.ln_1 = LayerNormalization(epsilon=1e-5)  # Keras LayerNormalization
        self.attn = NonCausalSelfAttention(config)    # Assuming this is a Keras Layer
        self.ln_2 = LayerNormalization(epsilon=1e-5)  # Keras LayerNormalization
        self.mlp = MLP(config)                        # Assuming this is a Keras Layer

    def call(self, x, training=False):
        attn_output = self.attn(self.ln_1(x))
        x = Add()([x, attn_output])
        x = Add()([x, self.mlp(self.ln_2(x))])
        return x



@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1




class FineGPT(GPT):
    def __init__(self, config):
        super(FineGPT, self).__init__()
        self.config = config
        self.n_codes_total = config.n_codes_total

        # Multiple token embeddings
        self.wtes = [Embedding(config.input_vocab_size, config.n_embd) for _ in range(config.n_codes_total)]

        # Position Embedding
        self.wpe = Embedding(config.block_size, config.n_embd)

        # Dropout
        self.drop = Dropout(config.dropout)

        # FineBlocks
        self.h = [FineBlock(config) for _ in range(config.n_layer)]

        # Layer Normalization
        self.ln_f = LayerNormalization(epsilon=1e-5)

        # Multiple language model heads
        self.lm_heads = [Dense(config.output_vocab_size, use_bias=False) for _ in range(config.n_codes_given, self.n_codes_total)]

        # Weight sharing between token embeddings and lm heads
        for i in range(self.n_codes_total - config.n_codes_given):
            self.wtes[i + 1].set_weights(self.lm_heads[i].get_weights())

    def call(self, pred_idx, idx):
        b, t, codes = idx.shape
        assert t <= self.config.block_size, f"Sequence length {t} is longer than block size {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, f"Expected {self.n_codes_total} codes, got {codes}"

        # Position IDs
        position_ids = ops.arange(0, t)
        position_ids = ops.expand_dims(position_ids, 0)

        # Token embeddings for each codebook
        tok_embs = [wte(idx[:, :, i]) for i, wte in enumerate(self.wtes)]

        # Sum token embeddings up to pred_idx
        tok_emb = ops.sum(tok_embs[:pred_idx + 1], axis=0)

        # Position embeddings
        pos_emb = self.wpe(position_ids)

        # Combine token and position embeddings with dropout
        x = self.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.h:
            x = block(x)

        # Layer normalization
        x = self.ln_f(x)

        # Select appropriate language model head and compute logits
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self):
        # Calculate the number of trainable parameters
        num_params = ops.sum([ops.prod(w.shape.as_list()) for w in self.trainable_weights])
        return num_params
    
@dataclass
class FineGPTConfig(GPTConfig):
    n_codes_total: int = 8
    n_codes_given: int = 1