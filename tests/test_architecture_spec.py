"""
Automated Structural Architecture Tests for Transformer 2017.
Verifies topology against Vaswani et al. 2017 Figure 1 and ensures single source of truth.
Checks all 31 nodes, 41 edges, residual bypasses, cross-attention memory edges, and synchronization.
"""

import os
import sys
import json
sys.path.insert(0, os.path.abspath("."))

import pytest
import numpy as np
from backend.transformer_keras_2017 import EducationalTransformer2017


@pytest.fixture
def spec():
    spec_path = os.path.join(os.path.dirname(__file__), "..", "transformer_2017_spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_spec_integrity_and_nodes(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    # Verify exactly 31 nodes
    assert len(nodes) == 31, f"Expected 31 nodes in Figure 1 spec, found {len(nodes)}"

    # Required Encoder nodes
    encoder_nodes = [
        "enc_inputs", "enc_input_embedding", "enc_pe", "enc_pe_add",
        "enc_layer_0_mha", "enc_layer_0_addnorm_1", "enc_layer_0_ffn", "enc_layer_0_addnorm_2",
        "enc_layer_1_mha", "enc_layer_1_addnorm_1", "enc_layer_1_ffn", "enc_layer_1_addnorm_2"
    ]
    for nid in encoder_nodes:
        assert nid in nodes, f"Missing encoder node: {nid}"

    # Required Decoder nodes
    decoder_nodes = [
        "dec_outputs", "dec_output_embedding", "dec_pe", "dec_pe_add",
        "dec_layer_0_masked_mha", "dec_layer_0_addnorm_1", "dec_layer_0_cross_mha", "dec_layer_0_addnorm_2", "dec_layer_0_ffn", "dec_layer_0_addnorm_3",
        "dec_layer_1_masked_mha", "dec_layer_1_addnorm_1", "dec_layer_1_cross_mha", "dec_layer_1_addnorm_2", "dec_layer_1_ffn", "dec_layer_1_addnorm_3"
    ]
    for nid in decoder_nodes:
        assert nid in nodes, f"Missing decoder node: {nid}"

    # Required Output nodes
    output_nodes = ["dec_linear", "dec_softmax", "dec_probabilities"]
    for nid in output_nodes:
        assert nid in nodes, f"Missing output node: {nid}"


def test_edge_count_and_topology(spec):
    edges = spec["edges"]
    assert len(edges) == 41, f"Expected 41 edges in Figure 1 spec, found {len(edges)}"


def test_residual_edges_topology(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    residual_edges = [e for e in spec["edges"] if e["type"] == "residual"]

    # Every residual edge must land on an Add & Norm node
    assert len(residual_edges) >= 4
    for edge in residual_edges:
        target_node = nodes[edge["to"]]
        assert target_node["type"] == "add_norm", f"Residual edge {edge} must target Add & Norm node"


def test_memory_edges_cross_attention_topology(spec):
    nodes = {n["id"]: n for n in spec["nodes"]}
    memory_edges = [e for e in spec["edges"] if e["type"] == "memory"]

    # Memory edges must originate from Encoder top Add & Norm and target Decoder Cross Attention
    assert len(memory_edges) >= 2, f"Expected at least 2 cross-attention memory edges, found {len(memory_edges)}"
    for edge in memory_edges:
        src_node = nodes[edge["from"]]
        tgt_node = nodes[edge["to"]]
        assert src_node["column"] == "encoder"
        assert src_node["type"] == "add_norm"
        assert tgt_node["column"] == "decoder"
        assert tgt_node["type"] == "cross_multi_head_attention"


def test_frontend_spec_synchronization(spec):
    fe_spec_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "spec", "transformer_2017_spec.json")
    with open(fe_spec_path, "r", encoding="utf-8") as f:
        fe_spec = json.load(f)

    assert len(fe_spec["nodes"]) == len(spec["nodes"])
    assert len(fe_spec["edges"]) == len(spec["edges"])


def test_keras_model_synchronization_with_spec(spec):
    model = EducationalTransformer2017(d_model=8, num_heads=2, d_ff=16)

    # Encoder Layer 1 & 2
    for enc_l in [model.enc_layer1, model.enc_layer2]:
        assert hasattr(enc_l, "mha")
        assert hasattr(enc_l, "ln1")
        assert hasattr(enc_l, "ffn")
        assert hasattr(enc_l, "ln2")

    # Decoder Layer 1 & 2
    for dec_l in [model.dec_layer1, model.dec_layer2]:
        assert hasattr(dec_l, "self_mha")
        assert hasattr(dec_l, "ln1")
        assert hasattr(dec_l, "cross_mha")
        assert hasattr(dec_l, "ln2")
        assert hasattr(dec_l, "ffn")
        assert hasattr(dec_l, "ln3")

    # Output projection
    assert hasattr(model, "final_linear")
