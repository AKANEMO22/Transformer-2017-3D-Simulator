"""
Transformer 2017 (Vaswani et al.) Educational Architecture in Keras 3.
Extracts complete, mathematically grounded execution traces for 3D visual simulation.
All tensors are calculated from real Keras layer forward passes (Embedding, Positional Encoding,
Multi-Head Attention, Add & LayerNorm, FFN, Masked Causal Attention, Cross-Attention, Linear & Softmax).
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import math
import time
import numpy as np
import keras
from keras import layers, ops


class SinusoidalPositionalEncoding(layers.Layer):
    """Positional Encoding using sine and cosine functions of different frequencies."""
    def __init__(self, d_model: int, max_len: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.d_model = d_model
        self.max_len = max_len

        # Compute sinusoidal positional encoding matrix according to Vaswani et al. 2017
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe_matrix = ops.convert_to_tensor(pe[np.newaxis, ...], dtype="float32")

    def call(self, x):
        seq_len = ops.shape(x)[1]
        return x + self.pe_matrix[:, :seq_len, :]

    def get_pe(self, seq_len: int):
        return ops.convert_to_numpy(self.pe_matrix[0, :seq_len, :])


class MultiHeadAttentionWithTrace(layers.Layer):
    """
    Multi-Head Attention layer with explicit extraction of:
    - Q, K, V projections
    - Scaled dot products (Q K^T / sqrt(d_k))
    - Softmax attention weights per head
    - Head outputs and final projection W_o
    """
    def __init__(self, d_model: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = layers.Dense(d_model, use_bias=False, name=f"{self.name}_wq")
        self.w_k = layers.Dense(d_model, use_bias=False, name=f"{self.name}_wk")
        self.w_v = layers.Dense(d_model, use_bias=False, name=f"{self.name}_wv")
        self.w_o = layers.Dense(d_model, use_bias=False, name=f"{self.name}_wo")

    def split_heads(self, x, batch_size):
        seq_len = ops.shape(x)[1]
        x = ops.reshape(x, (batch_size, seq_len, self.num_heads, self.d_k))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, q_in, k_in, v_in, mask=None, return_trace=False):
        batch_size = ops.shape(q_in)[0]

        # 1. Linear projections
        q = self.w_q(q_in)  # [B, T_q, d_model]
        k = self.w_k(k_in)  # [B, T_k, d_model]
        v = self.w_v(v_in)  # [B, T_v, d_model]

        # 2. Split heads
        q_heads = self.split_heads(q, batch_size)  # [B, H, T_q, d_k]
        k_heads = self.split_heads(k, batch_size)  # [B, H, T_k, d_k]
        v_heads = self.split_heads(v, batch_size)  # [B, H, T_v, d_k]

        # 3. Scaled Dot-Product Attention: (Q * K^T) / sqrt(d_k)
        scale = math.sqrt(self.d_k)
        raw_scores = ops.matmul(q_heads, ops.transpose(k_heads, (0, 1, 3, 2))) / scale

        # 4. Optional Causal / Padding Mask
        if mask is not None:
            # Add -1e9 to masked positions so softmax zeroes them out
            score_mask = (1.0 - mask) * -1e9
            masked_scores = raw_scores + score_mask
        else:
            masked_scores = raw_scores

        # 5. Softmax along key dimension
        attn_weights = ops.softmax(masked_scores, axis=-1)  # [B, H, T_q, T_k]

        # 6. Weighted sum with V
        context_heads = ops.matmul(attn_weights, v_heads)  # [B, H, T_q, d_k]

        # 7. Concatenate heads
        context = ops.transpose(context_heads, (0, 2, 1, 3))  # [B, T_q, H, d_k]
        t_q = ops.shape(q_in)[1]
        concat_context = ops.reshape(context, (batch_size, t_q, self.d_model))

        # 8. Output projection W_o
        output = self.w_o(concat_context)

        if return_trace:
            trace = {
                "q": ops.convert_to_numpy(q_heads[0]),
                "k": ops.convert_to_numpy(k_heads[0]),
                "v": ops.convert_to_numpy(v_heads[0]),
                "raw_scores": ops.convert_to_numpy(raw_scores[0]),
                "attn_weights": ops.convert_to_numpy(attn_weights[0]),
                "output": ops.convert_to_numpy(output[0])
            }
            return output, trace
        return output


class PositionWiseFeedForward(layers.Layer):
    """Feed-Forward Network: FFN(x) = max(0, xW1 + b1)W2 + b2"""
    def __init__(self, d_model: int, d_ff: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.dense1 = layers.Dense(d_ff, activation="relu", name=f"{self.name}_dense1")
        self.dense2 = layers.Dense(d_model, name=f"{self.name}_dense2")

    def call(self, x, return_trace=False):
        hidden = self.dense1(x)
        out = self.dense2(hidden)
        if return_trace:
            return out, {
                "hidden": ops.convert_to_numpy(hidden[0]),
                "output": ops.convert_to_numpy(out[0])
            }
        return out


class EncoderLayer(layers.Layer):
    """
    Standard Transformer Encoder Layer:
    - Multi-Head Self-Attention + Residual + LayerNorm
    - FFN + Residual + LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.mha = MultiHeadAttentionWithTrace(d_model, num_heads, name=f"{self.name}_mha")
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln1")
        self.ffn = PositionWiseFeedForward(d_model, d_ff, name=f"{self.name}_ffn")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln2")

    def call(self, x, mask=None, return_trace=False):
        attn_out, attn_trace = self.mha(x, x, x, mask=mask, return_trace=True)
        x_norm1 = self.ln1(x + attn_out)

        ffn_out, ffn_trace = self.ffn(x_norm1, return_trace=True)
        out = self.ln2(x_norm1 + ffn_out)

        if return_trace:
            layer_trace = {
                "attention": attn_trace,
                "residual_1": ops.convert_to_numpy(x_norm1[0]),
                "ffn": ffn_trace,
                "residual_2": ops.convert_to_numpy(out[0]),
                "layer_output": ops.convert_to_numpy(out[0])
            }
            return out, layer_trace
        return out


class DecoderLayer(layers.Layer):
    """
    Standard Transformer Decoder Layer:
    - Masked Multi-Head Self-Attention + Residual + LayerNorm
    - Multi-Head Cross-Attention + Residual + LayerNorm
    - FFN + Residual + LayerNorm
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.self_mha = MultiHeadAttentionWithTrace(d_model, num_heads, name=f"{self.name}_self_mha")
        self.ln1 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln1")

        self.cross_mha = MultiHeadAttentionWithTrace(d_model, num_heads, name=f"{self.name}_cross_mha")
        self.ln2 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln2")

        self.ffn = PositionWiseFeedForward(d_model, d_ff, name=f"{self.name}_ffn")
        self.ln3 = layers.LayerNormalization(epsilon=1e-6, name=f"{self.name}_ln3")

    def call(self, x, enc_output, self_mask=None, cross_mask=None, return_trace=False):
        self_attn_out, self_attn_trace = self.self_mha(x, x, x, mask=self_mask, return_trace=True)
        x_norm1 = self.ln1(x + self_attn_out)

        cross_attn_out, cross_attn_trace = self.cross_mha(
            x_norm1, enc_output, enc_output, mask=cross_mask, return_trace=True
        )
        x_norm2 = self.ln2(x_norm1 + cross_attn_out)

        ffn_out, ffn_trace = self.ffn(x_norm2, return_trace=True)
        out = self.ln3(x_norm2 + ffn_out)

        if return_trace:
            layer_trace = {
                "self_attention": self_attn_trace,
                "residual_1": ops.convert_to_numpy(x_norm1[0]),
                "cross_attention": cross_attn_trace,
                "residual_2": ops.convert_to_numpy(x_norm2[0]),
                "ffn": ffn_trace,
                "residual_3": ops.convert_to_numpy(out[0]),
                "layer_output": ops.convert_to_numpy(out[0])
            }
            return out, layer_trace
        return out


class EducationalTransformer2017:
    """
    Full Transformer 2017 Encoder-Decoder Model with grounded trace extraction.
    Real Keras 3 layers with weights initialized for clean educational demonstration.
    Does not manipulate logits or claim bilingual semantic fluency without training.
    """
    VOCAB_SOURCE = ["lắng", "nghe", "ý", "kiến", "nhóm", "giải", "quyết", "mâu", "thuẫn"]
    VOCAB_CONTRAST = ["python", "sql", "code", "mô", "hình"]
    VOCAB_TARGET = ["thấu", "hiểu", "đồng", "thuận", "hòa", "hợp", "kết", "quả"]
    SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]

    def __init__(self, d_model: int = 8, num_heads: int = 2, d_ff: int = 16):
        raw_vocab = self.SPECIAL_TOKENS + self.VOCAB_SOURCE + self.VOCAB_CONTRAST + self.VOCAB_TARGET
        seen = set()
        self.vocab = []
        for v in raw_vocab:
            if v not in seen:
                self.vocab.append(v)
                seen.add(v)

        self.vocab_size = len(self.vocab)
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for i, token in enumerate(self.vocab)}

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        # Keras Layers
        self.token_embedding = layers.Embedding(self.vocab_size, d_model, name="token_embedding")
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=64)

        self.enc_layer1 = EncoderLayer(d_model, num_heads, d_ff, name="enc_l1")
        self.enc_layer2 = EncoderLayer(d_model, num_heads, d_ff, name="enc_l2")

        self.dec_layer1 = DecoderLayer(d_model, num_heads, d_ff, name="dec_l1")
        self.dec_layer2 = DecoderLayer(d_model, num_heads, d_ff, name="dec_l2")

        self.final_linear = layers.Dense(self.vocab_size, name="output_logits")

        self._build_and_tune_weights()

    def _build_and_tune_weights(self):
        """Initializes model layers and builds weights."""
        dummy_in = ops.convert_to_tensor([[1, 3, 4]], dtype="int32")
        emb = self.token_embedding(dummy_in)
        pos = self.pos_encoding(emb)
        enc_out, _ = self.enc_layer1(pos, return_trace=True)
        enc_out2, _ = self.enc_layer2(enc_out, return_trace=True)

        dummy_tgt = ops.convert_to_tensor([[1, 5]], dtype="int32")
        tgt_emb = self.pos_encoding(self.token_embedding(dummy_tgt))
        dec_out, _ = self.dec_layer1(tgt_emb, enc_out2, return_trace=True)
        dec_out2, _ = self.dec_layer2(dec_out, enc_out2, return_trace=True)
        _ = self.final_linear(dec_out2)

        # Initialize structured embedding weights with deterministic seed
        rng = np.random.RandomState(42)
        emb_weights = rng.randn(self.vocab_size, self.d_model).astype(np.float32) * 0.4
        self.token_embedding.set_weights([emb_weights])

    def tokenize(self, text: str, max_len: int = 5):
        """Tokenize input string into vocab tokens with fallback mapping for arbitrary text."""
        if not text or not text.strip():
            raise ValueError("source_text cannot be empty")

        words = text.lower().replace(",", " ").replace(".", " ").replace("?", " ").replace("!", " ").split()
        if not words:
            raise ValueError("No valid words found in input text")

        tokens = []
        for w in words:
            if w in self.token2id:
                tokens.append(w)
            else:
                found = False
                for v in self.vocab:
                    if v in w or w in v:
                        tokens.append(v)
                        found = True
                        break
                if not found:
                    # Deterministically map out-of-vocab word using character hash
                    char_sum = sum(ord(c) for c in w)
                    mapped_idx = 3 + (char_sum % (self.vocab_size - 3))
                    tokens.append(self.id2token[mapped_idx])
            if len(tokens) >= max_len:
                break
        return tokens

    def get_causal_mask(self, seq_len: int):
        """Lower triangular matrix where past tokens can be attended (1), future masked (0)."""
        mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        return ops.convert_to_tensor(mask[np.newaxis, np.newaxis, ...], dtype="float32")

    def run_trace(self, source_text: str, target_prompt: str = None, max_decode_steps: int = 4):
        src_tokens = self.tokenize(source_text, max_len=5)
        src_ids = [self.token2id[t] for t in src_tokens]
        src_tensor = ops.convert_to_tensor([src_ids], dtype="int32")
        seq_len = len(src_ids)

        # 1. Embedding lookup
        raw_emb = self.token_embedding(src_tensor)
        emb_np = ops.convert_to_numpy(raw_emb[0])

        # 2. Positional Encoding
        pe_np = self.pos_encoding.get_pe(seq_len)
        emb_pe = self.pos_encoding(raw_emb)
        emb_pe_np = ops.convert_to_numpy(emb_pe[0])

        # 3. Encoder Layer 1 & 2
        enc1_out, enc1_trace = self.enc_layer1(emb_pe, return_trace=True)
        enc2_out, enc2_trace = self.enc_layer2(enc1_out, return_trace=True)
        enc_final = enc2_out

        encoder_trace = {
            "source_tokens": src_tokens,
            "source_ids": src_ids,
            "embeddings": emb_np.tolist(),
            "positional_encodings": pe_np.tolist(),
            "embedded_positions": emb_pe_np.tolist(),
            "layer_1": {
                "q": enc1_trace["attention"]["q"].tolist(),
                "k": enc1_trace["attention"]["k"].tolist(),
                "v": enc1_trace["attention"]["v"].tolist(),
                "raw_scores": enc1_trace["attention"]["raw_scores"].tolist(),
                "attention_weights": enc1_trace["attention"]["attn_weights"].tolist(),
                "residual_1": enc1_trace["residual_1"].tolist(),
                "ffn_output": enc1_trace["ffn"]["output"].tolist(),
                "residual_2": enc1_trace["residual_2"].tolist(),
                "layer_output": enc1_trace["layer_output"].tolist()
            },
            "layer_2": {
                "q": enc2_trace["attention"]["q"].tolist(),
                "k": enc2_trace["attention"]["k"].tolist(),
                "v": enc2_trace["attention"]["v"].tolist(),
                "raw_scores": enc2_trace["attention"]["raw_scores"].tolist(),
                "attention_weights": enc2_trace["attention"]["attn_weights"].tolist(),
                "residual_1": enc2_trace["residual_1"].tolist(),
                "ffn_output": enc2_trace["ffn"]["output"].tolist(),
                "residual_2": enc2_trace["residual_2"].tolist(),
                "layer_output": enc2_trace["layer_output"].tolist()
            }
        }

        # 4. Autoregressive Decoding with genuine Keras logits and softmax
        current_tgt_ids = [self.token2id["<s>"]]
        decoder_steps = []

        for step in range(max_decode_steps):
            tgt_tensor = ops.convert_to_tensor([current_tgt_ids], dtype="int32")
            cur_len = len(current_tgt_ids)

            dec_emb = self.token_embedding(tgt_tensor)
            dec_pe = self.pos_encoding(dec_emb)
            dec_pe_np = ops.convert_to_numpy(dec_pe[0])

            causal_mask = self.get_causal_mask(cur_len)
            mask_matrix_np = ops.convert_to_numpy(causal_mask[0, 0])

            dec1_out, dec1_trace = self.dec_layer1(
                dec_pe, enc_final, self_mask=causal_mask, return_trace=True
            )
            dec2_out, dec2_trace = self.dec_layer2(
                dec1_out, enc_final, self_mask=causal_mask, return_trace=True
            )

            # Genuine Keras Linear Projection & Softmax (NO silent hand-crafted manipulation!)
            step_logits = self.final_linear(dec2_out)[:, -1, :]  # [1, vocab_size]
            step_probs = ops.softmax(step_logits, axis=-1)
            probs_np = ops.convert_to_numpy(step_probs[0])
            logits_np = ops.convert_to_numpy(step_logits[0])

            sorted_indices = np.argsort(-probs_np)
            top_candidates = []
            for rank_idx in range(min(5, self.vocab_size)):
                idx = int(sorted_indices[rank_idx])
                top_candidates.append({
                    "id": idx,
                    "token": self.id2token[idx],
                    "prob": round(float(probs_np[idx]), 4),
                    "logit": round(float(logits_np[idx]), 4)
                })

            next_token_id = int(sorted_indices[0])
            # Avoid immediately generating start token or padding
            if self.id2token[next_token_id] in ["<pad>", "<s>"]:
                next_token_id = int(sorted_indices[1])

            next_token = self.id2token[next_token_id]

            # Detailed Decoder Layer 0 & Layer 1 traces
            step_trace = {
                "step_index": step,
                "current_tokens": [self.id2token[tid] for tid in current_tgt_ids],
                "current_token_ids": current_tgt_ids,
                "embeddings": ops.convert_to_numpy(dec_emb[0]).tolist(),
                "positional_encodings": self.pos_encoding.get_pe(cur_len).tolist(),
                "causal_mask": mask_matrix_np.tolist(),
                "layer_0": {
                    "masked_self_attention": {
                        "q": dec1_trace["self_attention"]["q"].tolist(),
                        "k": dec1_trace["self_attention"]["k"].tolist(),
                        "v": dec1_trace["self_attention"]["v"].tolist(),
                        "raw_scores": dec1_trace["self_attention"]["raw_scores"].tolist(),
                        "attention_weights": dec1_trace["self_attention"]["attn_weights"].tolist()
                    },
                    "residual_1": dec1_trace["residual_1"].tolist(),
                    "cross_attention": {
                        "q": dec1_trace["cross_attention"]["q"].tolist(),
                        "k": dec1_trace["cross_attention"]["k"].tolist(),
                        "v": dec1_trace["cross_attention"]["v"].tolist(),
                        "raw_scores": dec1_trace["cross_attention"]["raw_scores"].tolist(),
                        "attention_weights": dec1_trace["cross_attention"]["attn_weights"].tolist()
                    },
                    "residual_2": dec1_trace["residual_2"].tolist(),
                    "ffn_output": dec1_trace["ffn"]["output"].tolist(),
                    "residual_3": dec1_trace["residual_3"].tolist(),
                    "layer_output": dec1_trace["layer_output"].tolist()
                },
                "layer_1": {
                    "masked_self_attention": {
                        "q": dec2_trace["self_attention"]["q"].tolist(),
                        "k": dec2_trace["self_attention"]["k"].tolist(),
                        "v": dec2_trace["self_attention"]["v"].tolist(),
                        "raw_scores": dec2_trace["self_attention"]["raw_scores"].tolist(),
                        "attention_weights": dec2_trace["self_attention"]["attn_weights"].tolist()
                    },
                    "residual_1": dec2_trace["residual_1"].tolist(),
                    "cross_attention": {
                        "q": dec2_trace["cross_attention"]["q"].tolist(),
                        "k": dec2_trace["cross_attention"]["k"].tolist(),
                        "v": dec2_trace["cross_attention"]["v"].tolist(),
                        "raw_scores": dec2_trace["cross_attention"]["raw_scores"].tolist(),
                        "attention_weights": dec2_trace["cross_attention"]["attn_weights"].tolist()
                    },
                    "residual_2": dec2_trace["residual_2"].tolist(),
                    "ffn_output": dec2_trace["ffn"]["output"].tolist(),
                    "residual_3": dec2_trace["residual_3"].tolist(),
                    "layer_output": dec2_trace["layer_output"].tolist()
                },
                # Backward compatibility
                "masked_self_attention": {
                    "q": dec1_trace["self_attention"]["q"].tolist(),
                    "k": dec1_trace["self_attention"]["k"].tolist(),
                    "v": dec1_trace["self_attention"]["v"].tolist(),
                    "attention_weights": dec1_trace["self_attention"]["attn_weights"].tolist()
                },
                "cross_attention": {
                    "q": dec1_trace["cross_attention"]["q"].tolist(),
                    "k": dec1_trace["cross_attention"]["k"].tolist(),
                    "v": dec1_trace["cross_attention"]["v"].tolist(),
                    "attention_weights": dec1_trace["cross_attention"]["attn_weights"].tolist()
                },
                "ffn_output": dec1_trace["ffn"]["output"].tolist(),
                "final_hidden": ops.convert_to_numpy(dec2_out[0]).tolist(),
                "logits": [round(float(l), 4) for l in logits_np],
                "top_candidates": top_candidates,
                "next_token": next_token,
                "next_token_id": next_token_id,
                "all_probabilities": [round(float(p), 4) for p in probs_np]
            }
            decoder_steps.append(step_trace)

            current_tgt_ids.append(next_token_id)
            if next_token == "</s>":
                break

        full_generated_tokens = [self.id2token[tid] for tid in current_tgt_ids]

        return {
            "metadata": {
                "runId": f"keras_{int(time.time()*1000)}",
                "source": "keras",
                "inputText": source_text,
                "modelName": "Keras 3 Educational Transformer (Vaswani 2017)",
                "modelConfig": {
                    "numEncoderLayers": 2,
                    "numDecoderLayers": 2,
                    "dModel": self.d_model,
                    "numHeads": self.num_heads,
                    "dFF": self.d_ff,
                    "vocabSize": self.vocab_size
                },
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "isUntrained": True,
                "disclaimer": "Mô hình giáo dục Keras 3 forward-pass nguyên bản với trọng số khởi tạo chuẩn. Không giả mạo ngữ nghĩa tiếng Việt."
            },
            "source_text": source_text,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_k": self.d_model // self.num_heads,
            "d_ff": self.d_ff,
            "vocab": self.vocab,
            "encoder": encoder_trace,
            "decoder_steps": decoder_steps,
            "final_generated_tokens": full_generated_tokens,
            "output_sentence": " ".join([t for t in full_generated_tokens if t not in ["<s>", "</s>", "<pad>"]])
        }
