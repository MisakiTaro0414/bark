from keras.layers import Layer, Dense, Dropout, Activation, Add, Embedding
from keras.models import Model
from keras import ops
from dataclasses import dataclass

class LayerNorm(Layer):
    """ LayerNorm but with an optional bias. """

    def __init__(self, ndim, bias=True, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.ndim = ndim
        self.bias_enabled = bias

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.weight = self.add_weight(name='weight',
                                      shape=(self.ndim,),
                                      initializer='ones',
                                      trainable=True)
        if self.bias_enabled:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.ndim,),
                                        initializer='zeros',
                                        trainable=True)
        else:
            self.bias = None
        super(LayerNorm, self).build(input_shape)

    def call(self, inputs):
        mean = ops.mean(inputs, axis=-1, keepdims=True)
        std = ops.std(inputs, axis=-1, keepdims=True)
        normalized = (inputs - mean) / (std + 1e-5)
        return normalized * self.weight + (self.bias if self.bias_enabled else 0)

    def compute_output_shape(self, input_shape):
        return input_shape
    

class MLP(Layer):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.n_embd = config.n_embd
        self.bias = config.bias
        self.dropout_rate = config.dropout

        self.c_fc = Dense(4 * self.n_embd, use_bias=self.bias)
        self.c_proj = Dense(self.n_embd, use_bias=self.bias)
        self.dropout = Dropout(self.dropout_rate)

    def call(self, x):
        x = self.c_fc(x)
        x = Activation('gelu')(x)  # Gelu activation
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_embd)


class CausalSelfAttention(Layer):
    def __init__(self, config):
        super(CausalSelfAttention, self).__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.key_dim = config.n_embd // config.n_head

        self.c_attn = Dense(3 * config.n_embd, use_bias=config.bias)
        self.c_proj = Dense(config.n_embd, use_bias=config.bias)
        self.attn_dropout = Dropout(config.dropout)
        self.resid_dropout = Dropout(config.dropout)

        # Causal mask
        self.bias = ops.tril(ops.ones((config.block_size, config.block_size)), k=0)

    def call(self, x):
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = ops.split(qkv, 3, axis=2)
        q, k, v = [ops.reshape(z, (B, T, self.n_head, self.key_dim)) for z in [q, k, v]]
        q, k, v = [ops.transpose(z, (0, 2, 1, 3)) for z in [q, k, v]]

        # causal self-attention
        att = ops.matmul(q, k.transpose(0, 1, 3, 2)) / ops.sqrt(self.key_dim)
        mask = ops.expand_dims(self.bias[:T, :T], axis=0)
        att = att - (1 - mask) * 1e10
        att = ops.exp(att - ops.max(att, axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        att = self.attn_dropout(att)
        y = ops.matmul(att, v)

        y = ops.transpose(y, (0, 2, 1, 3))
        y = ops.reshape(y, (B, T, C))

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

    def compute_output_shape(self, input_shape):
        return input_shape
    

class Block(Layer):
    def __init__(self, config, layer_idx):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.layer_idx = layer_idx

    def call(self, x, past_kv=None, use_cache=False, training=False):
        attn_output, prev_kvs = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = Add()([x, attn_output])
        x = Add()([x, self.mlp(self.ln_2(x))])
        return x, prev_kvs


@dataclass
class GPTConfig:
    block_size: int = 1024
    input_vocab_size: int = 10_048
    output_vocab_size: int = 10_048
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(Model):
    def __init__(self, config):
        super(GPT, self).__init__()
        self.config = config

        # Embeddings
        self.wte = Embedding(config.input_vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)

        # Dropout
        self.drop = Dropout(config.dropout)

        # Transformer blocks
        self.h = [Block(config, idx) for idx in range(config.n_layer)]

        # Layer Normalization
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Final linear layer
        self.lm_head = Dense(config.output_vocab_size, use_bias=False)

    def call(self, idx, merge_context=False, past_kv=None, position_ids=None, use_cache=False):
        b, t = idx.shape

        # Token embeddings
        if merge_context:
            assert idx.shape[1] >= 256 + 256 + 1
            t = idx.shape[1] - 256
            tok_emb = ops.concatenate([
                self.wte(idx[:, :256]) + self.wte(idx[:, 256:256 + 256]),
                self.wte(idx[:, 256 + 256:])
            ], axis=1)
        else:
            assert t <= self.config.block_size, f"Sequence length {t} is longer than block size {self.config.block_size}"
            tok_emb = self.wte(idx)

        if past_kv is None:
            past_length = 0
            past_kv = [None] * len(self.h)
        else:
            past_length = past_kv[0][0].shape[-2]

        if position_ids is None:
            position_ids = ops.arange(past_length, t + past_length)
            position_ids = ops.expand_dims(position_ids, 0)

        # Position embeddings
        pos_emb = self.wpe(position_ids)

        # Combine token and position embeddings with dropout
        x = self.drop(tok_emb + pos_emb)

        new_kv = [] if use_cache else None

        for i, block in enumerate(self.h):
            past_layer_kv = past_kv[i] if past_kv else None
            x, kv = block(x, past_kv=past_layer_kv, use_cache=use_cache)

            if use_cache:
                new_kv.append(kv)

        x = self.ln_f(x)

        # Inference optimization: process only the last position
        logits = self.lm_head(x[:, -1, :])
        return logits, new_kv

    def get_num_params(self):
        # Calculate the number of trainable parameters
        return ops.sum([ops.prod(w.shape) for w in self.trainable_weights])



