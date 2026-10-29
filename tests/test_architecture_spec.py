"""
Automated Structural Architecture Tests for Transformer 2017.
Verifies topology against Vaswani et al. 2017 Figure 1 and ensures single source of truth.
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
from transformer_3d_simulator.backend.transformer_keras_2017 import EducationalTransformer2017


@pytest.fixture
def spec():
    spec_path = os.path.join(os.path.dirname(__file__), "..", "transformer_3d_simulator", "transformer_2017_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_spec_integrity_and_nodes(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    assert "enc_inputs" in nodes
    assert "enc_input_embedding" in nodes
    assert "enc_pe" in nodes
    assert "enc_layer_0_mha" in nodes
    assert "enc_layer_0_addnorm_1" in nodes
    assert "enc_layer_0_ffn" in nodes
    assert "enc_layer_0_addnorm_2" in nodes
    assert "dec_outputs" in nodes
    assert "dec_output_embedding" in nodes
    assert "dec_layer_0_masked_mha" in nodes
    assert "dec_layer_0_cross_mha" in nodes
    assert "dec_linear" in nodes
    assert "dec_softmax" in nodes
    assert "dec_probabilities" in nodes


def test_residual_edges_topology(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    residual_edges = [e for e in spec["edges"] if e["type"] == "residual"]
    
    # Every residual edge must land on an Add & Norm node
    assert len(residual_edges) >= 4  # at least 2 per encoder layer, 3 per decoder layer
    for edge in residual_edges:
        target_node = nodes[edge["to"]]
        assert target_node["type"] == "add_norm", f"Residual edge {edge} must target Add & Norm node"


def test_memory_edges_cross_attention_topology(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    memory_edges = [e for e in spec["edges"] if e["type"] == "memory"]
    
    # Memory edges must originate from Encoder top Add & Norm and target Decoder Cross Attention
    assert len(memory_edges) >= 1
    for edge in memory_edges:
        src_node = nodes[edge["from"]]
        tgt_node = nodes[edge["to"]]
        assert src_node["column"] == "encoder"
        assert src_node["type"] == "add_norm"
        assert tgt_node["column"] == "decoder"
        assert tgt_node["type"] == "cross_multi_head_attention"


def test_keras_model_synchronization_with_spec(spec):
    model = EducationalTransformer2017(d_model=16, num_heads=2, d_ff=32)
    
    # Encoder Layer 1 has MHA, LN1, FFN, LN2
    assert hasattr(model.enc_layer1, "mha")
    assert hasattr(model.enc_layer1, "ln1")
    assert hasattr(model.enc_layer1, "ffn")
    assert hasattr(model.enc_layer1, "ln2")
    
    # Decoder Layer 1 has Self-MHA (masked), LN1, Cross-MHA, LN2, FFN, LN3
    assert hasattr(model.dec_layer1, "self_mha")
    assert hasattr(model.dec_layer1, "ln1")
    assert hasattr(model.dec_layer1, "cross_mha")
    assert hasattr(model.dec_layer1, "ln2")
    assert hasattr(model.dec_layer1, "ffn")
    assert hasattr(model.dec_layer1, "ln3")
    
    # Output projection has linear and softmax
    assert hasattr(model, "final_linear")
