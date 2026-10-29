"""
Automated Verification Test Suite for Transformer 2017 3D Simulator.
Verifies genuine Keras 3 layers, causal masking, cross-attention, and JSON trace generation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
from transformer_3d_simulator.backend.transformer_keras_2017 import (
    EducationalTransformer2017,
    SinusoidalPositionalEncoding
)


def test_positional_encoding_formula():
    pe = SinusoidalPositionalEncoding(d_model=8, max_len=10)
    matrix = pe.get_pe(seq_len=5)
    assert matrix.shape == (5, 8)
    assert np.isclose(matrix[0, 0], 0.0, atol=1e-5)
    assert np.isclose(matrix[0, 1], 1.0, atol=1e-5)


def test_causal_mask_structure():
    model = EducationalTransformer2017(d_model=8, num_heads=2)
    mask = model.get_causal_mask(seq_len=4).numpy()[0, 0]
    # Verify upper triangle (future tokens) is strictly 0.0
    upper_tri = np.triu(mask, k=1)
    assert np.all(upper_tri == 0.0)
    # Verify lower triangle (past tokens) is strictly 1.0
    for i in range(4):
        for j in range(4):
            if j <= i:
                assert mask[i, j] == 1.0
            else:
                assert mask[i, j] == 0.0


def test_encoder_attention_shapes_and_weights():
    model = EducationalTransformer2017(d_model=8, num_heads=2)
    trace = model.run_trace("Lắng nghe ý kiến nhóm")
    enc = trace["encoder"]
    seq_len = len(enc["source_tokens"])
    assert seq_len == 5  # ["lắng", "nghe", "ý", "kiến", "nhóm"]
    
    attn_weights = np.array(enc["layer_1"]["attention_weights"])
    # Shape: [num_heads=2, seq_len=5, seq_len=5]
    assert attn_weights.shape == (2, seq_len, seq_len)
    for h in range(2):
        for i in range(seq_len):
            assert np.isclose(attn_weights[h, i].sum(), 1.0, atol=1e-4)


def test_decoder_masked_and_cross_attention():
    model = EducationalTransformer2017(d_model=8, num_heads=2)
    trace = model.run_trace("Lắng nghe ý kiến nhóm", max_decode_steps=3)
    dec_steps = trace["decoder_steps"]
    assert len(dec_steps) >= 2

    # Step 1 has current_tokens: ["<s>", "thấu"]
    step1 = dec_steps[1]
    masked_attn = np.array(step1["masked_self_attention"]["attention_weights"])
    cur_len = len(step1["current_tokens"])
    assert cur_len == 2
    assert masked_attn.shape == (2, 2, 2)
    
    # Position 0 cannot attend to position 1 (future token masked to 0.0)
    assert np.isclose(masked_attn[0, 0, 1], 0.0, atol=1e-5)
    assert np.isclose(masked_attn[1, 0, 1], 0.0, atol=1e-5)

    # Cross attention: [num_heads=2, cur_len=2, enc_len=5]
    cross_attn = np.array(step1["cross_attention"]["attention_weights"])
    assert cross_attn.shape == (2, 2, 5)
    for h in range(2):
        for i in range(cur_len):
            assert np.isclose(cross_attn[h, i].sum(), 1.0, atol=1e-4)


def test_autoregressive_generation():
    model = EducationalTransformer2017(d_model=8, num_heads=2)
    trace = model.run_trace("Lắng nghe ý kiến nhóm", max_decode_steps=4)
    assert len(trace["final_generated_tokens"]) > 1
    assert isinstance(trace["output_sentence"], str)
    assert len(trace["output_sentence"]) > 0
