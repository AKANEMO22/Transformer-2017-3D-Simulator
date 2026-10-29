import type {
  TransformerTrace,
  NodeExecutionTrace,
  AttentionHeadDetail
} from '../types/trace';

export function getNodeDetail(
  nodeId: string,
  trace: TransformerTrace | null,
  activeDecoderStep = 0
): NodeExecutionTrace {
  if (!trace) {
    return {
      nodeId,
      status: 'unavailable',
      unavailableReason: 'Chưa có trace dữ liệu. Hãy nhấn [SIMULATE] để khởi chạy mô phỏng.',
      operation: 'Idle',
      inputs: [],
      outputs: []
    };
  }

  const enc = trace.encoder;
  const decSteps = trace.decoder_steps || [];
  const validStepIdx = Math.max(0, Math.min(activeDecoderStep, decSteps.length - 1));
  const currentDecStep = decSteps[validStepIdx];

  const dModel = trace.d_model || 8;
  const numHeads = trace.num_heads || 2;
  const dK = trace.d_k || dModel / numHeads;
  const encTokens = enc?.source_tokens || [];
  const encLen = encTokens.length;

  switch (nodeId) {
    // ---------------- ENCODER INPUT & EMBEDDING ----------------
    case 'enc_inputs': {
      return {
        nodeId,
        nodeName: 'Inputs (Source Sequence)',
        operation: 'Tokenization & Vocabulary Mapping',
        formula: 'Tokens \\rightarrow Token IDs',
        description: 'Tách câu đầu vào thành các token từ vựng và ánh xạ sang chỉ số nguyên (Token IDs).',
        status: 'available',
        inputs: [
          {
            name: 'raw_sentence',
            shape: [1],
            values: [trace.source_text as any],
            dtype: 'string',
            description: 'Văn bản đầu vào nguyên bản của người dùng'
          }
        ],
        outputs: [
          {
            name: 'tokens',
            shape: [encLen],
            values: encTokens as any,
            dtype: 'string[]',
            description: 'Danh sách các token từ vựng'
          },
          {
            name: 'token_ids',
            shape: [encLen],
            values: enc?.source_ids || [],
            dtype: 'int32',
            description: 'Mã số chỉ mục trong từ điển (Vocabulary IDs)'
          }
        ]
      };
    }

    case 'enc_input_embedding': {
      return {
        nodeId,
        nodeName: 'Input Embedding',
        operation: 'Embedding Matrix Lookup & Scaling',
        formula: 'E(x) \\times \\sqrt{d_{model}}',
        description: 'Tra cứu vector không gian dày đặc d_model cho từng token và nhân hệ số căn bậc hai của d_model.',
        status: 'available',
        inputs: [
          {
            name: 'source_ids',
            shape: [encLen],
            values: enc?.source_ids || [],
            dtype: 'int32',
            description: 'Token IDs đầu vào'
          }
        ],
        outputs: [
          {
            name: 'embeddings',
            shape: [encLen, dModel],
            values: enc?.embeddings || [],
            dtype: 'float32',
            description: 'Ma trận vector nhúng kích thước [T, d_model]'
          }
        ]
      };
    }

    case 'enc_pe': {
      return {
        nodeId,
        nodeName: 'Positional Encoding (Sinusoidal)',
        operation: 'Sinusoidal Positional Encoding Generation',
        formula: 'PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d}), PE_{(pos, 2i+1)} = \\cos(pos/10000^{2i/d})',
        description: 'Tạo mã vị trí điều hòa theo hàm sin và cos ở các tần số khác nhau để mô hình nhận biết thứ tự chuỗi.',
        status: 'available',
        inputs: [
          {
            name: 'positions',
            shape: [encLen],
            values: Array.from({ length: encLen }, (_, i) => i),
            dtype: 'int32',
            description: 'Vị trí thứ tự từ 0 đến seq_len - 1'
          }
        ],
        outputs: [
          {
            name: 'pe_matrix',
            shape: [encLen, dModel],
            values: enc?.positional_encodings || [],
            dtype: 'float32',
            description: 'Ma trận mã hóa vị trí [T, d_model]'
          }
        ]
      };
    }

    case 'enc_pe_add': {
      return {
        nodeId,
        nodeName: 'Add Positional Encoding',
        operation: 'Element-wise Tensor Addition',
        formula: 'x_0 = Embedding(x) + PE',
        description: 'Cộng từng phần tử giữa ma trận Embedding và Positional Encoding.',
        status: 'available',
        inputs: [
          {
            name: 'embeddings',
            shape: [encLen, dModel],
            values: enc?.embeddings || [],
            dtype: 'float32'
          },
          {
            name: 'pe',
            shape: [encLen, dModel],
            values: enc?.positional_encodings || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'embedded_positions',
            shape: [encLen, dModel],
            values: enc?.embedded_positions || [],
            dtype: 'float32',
            description: 'Biểu diễn ban đầu kết hợp ngữ nghĩa và vị trí truyền vào Encoder Stack'
          }
        ]
      };
    }

    // ---------------- ENCODER LAYER 0 ----------------
    case 'enc_layer_0_mha': {
      const l = enc?.layer_1;
      const headDetails: AttentionHeadDetail[] = [];
      if (l?.attention_weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: l.q?.[h] || [],
            k: l.k?.[h] || [],
            v: l.v?.[h] || [],
            rawScores: l.raw_scores?.[h] || [],
            weights: l.attention_weights?.[h] || [],
            queryTokens: encTokens,
            keyTokens: encTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Encoder Layer 0: Multi-Head Self-Attention',
        layerIndex: 0,
        operation: 'Scaled Dot-Product Self-Attention',
        formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
        description: 'Tính toán mối tương quan tự chú ý đa đầu giữa các token trong Encoder Layer 0.',
        status: 'available',
        inputs: [
          {
            name: 'x_in',
            shape: [encLen, dModel],
            values: enc?.embedded_positions || [],
            dtype: 'float32'
          }
        ],
        intermediate: [
          {
            name: 'q_heads',
            shape: [numHeads, encLen, dK],
            values: l?.q || [],
            dtype: 'float32'
          },
          {
            name: 'k_heads',
            shape: [numHeads, encLen, dK],
            values: l?.k || [],
            dtype: 'float32'
          },
          {
            name: 'v_heads',
            shape: [numHeads, encLen, dK],
            values: l?.v || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'attention_weights',
            shape: [numHeads, encLen, encLen],
            values: l?.attention_weights || [],
            dtype: 'float32',
            description: 'Ma trận trọng số chú ý sau Softmax (tổng mỗi hàng = 1.0)'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'enc_layer_0_addnorm_1': {
      const l = enc?.layer_1;
      return {
        nodeId,
        nodeName: 'Encoder Layer 0: Add & Norm (Sublayer 1)',
        layerIndex: 0,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{Sublayer}(x))',
        description: 'Đường tắt Residual kết hợp chuẩn hóa tầng (LayerNorm) giúp gradient lưu thông không bị triệt tiêu.',
        status: 'available',
        inputs: [
          {
            name: 'residual_in (x)',
            shape: [encLen, dModel],
            values: enc?.embedded_positions || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'normalized_output',
            shape: [encLen, dModel],
            values: l?.residual_1 || l?.layer_output || [],
            dtype: 'float32',
            description: 'Đầu ra chuẩn hóa phân phối trung bình 0, độ lệch chuẩn 1'
          }
        ]
      };
    }

    case 'enc_layer_0_ffn': {
      const l = enc?.layer_1;
      return {
        nodeId,
        nodeName: 'Encoder Layer 0: Feed Forward',
        layerIndex: 0,
        operation: 'Position-wise Feed-Forward Network',
        formula: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
        description: 'Mạng nơ-ron truyền thẳng 2 tầng biến đổi phi tuyến độc lập tại từng vị trí token với kích thước ẩn d_ff.',
        status: 'available',
        inputs: [
          {
            name: 'norm1_out',
            shape: [encLen, dModel],
            values: l?.residual_1 || [],
            dtype: 'float32'
          }
        ],
        intermediate: [
          {
            name: 'ffn_hidden',
            shape: [encLen, trace.d_ff],
            values: l?.ffn_hidden || [],
            dtype: 'float32',
            description: 'Tầng ẩn kích hoạt hàm ReLU kích thước d_ff'
          }
        ],
        outputs: [
          {
            name: 'ffn_output',
            shape: [encLen, dModel],
            values: l?.ffn_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'enc_layer_0_addnorm_2': {
      const l = enc?.layer_1;
      return {
        nodeId,
        nodeName: 'Encoder Layer 0: Add & Norm (Sublayer 2)',
        layerIndex: 0,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{FFN}(x))',
        description: 'Đường tắt residual thứ hai kết hợp LayerNorm hoàn tất tính toán của Encoder Layer 0.',
        status: 'available',
        inputs: [
          {
            name: 'ffn_output',
            shape: [encLen, dModel],
            values: l?.ffn_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'layer_0_output',
            shape: [encLen, dModel],
            values: l?.layer_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    // ---------------- ENCODER LAYER 1 ----------------
    case 'enc_layer_1_mha': {
      const l = enc?.layer_2;
      const headDetails: AttentionHeadDetail[] = [];
      if (l?.attention_weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: l.q?.[h] || [],
            k: l.k?.[h] || [],
            v: l.v?.[h] || [],
            rawScores: l.raw_scores?.[h] || [],
            weights: l.attention_weights?.[h] || [],
            queryTokens: encTokens,
            keyTokens: encTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Encoder Layer 1: Multi-Head Self-Attention',
        layerIndex: 1,
        operation: 'Deep Self-Attention Extraction',
        formula: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
        description: 'Tự chú ý tầng sâu của Encoder Layer 1, nhận biểu diễn cấp cao từ Layer 0.',
        status: 'available',
        inputs: [
          {
            name: 'layer_0_output',
            shape: [encLen, dModel],
            values: enc?.layer_1?.layer_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'layer_1_attention_weights',
            shape: [numHeads, encLen, encLen],
            values: l?.attention_weights || [],
            dtype: 'float32'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'enc_layer_1_addnorm_1': {
      const l = enc?.layer_2;
      return {
        nodeId,
        nodeName: 'Encoder Layer 1: Add & Norm (Sublayer 1)',
        layerIndex: 1,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{Sublayer}(x))',
        description: 'Đường tắt Residual và LayerNorm tầng 1 của Encoder.',
        status: 'available',
        inputs: [
          {
            name: 'layer_0_output',
            shape: [encLen, dModel],
            values: enc?.layer_1?.layer_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'normalized_output',
            shape: [encLen, dModel],
            values: l?.residual_1 || l?.layer_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'enc_layer_1_ffn': {
      const l = enc?.layer_2;
      return {
        nodeId,
        nodeName: 'Encoder Layer 1: Feed Forward',
        layerIndex: 1,
        operation: 'Position-wise Feed-Forward Network',
        formula: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
        description: 'Mạng nơ-ron truyền thẳng biến đổi biểu diễn ngữ nghĩa trừu tượng cấp cao.',
        status: 'available',
        inputs: [
          {
            name: 'norm1_out',
            shape: [encLen, dModel],
            values: l?.residual_1 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'ffn_output',
            shape: [encLen, dModel],
            values: l?.ffn_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'enc_layer_1_addnorm_2': {
      const l = enc?.layer_2;
      return {
        nodeId,
        nodeName: 'Encoder Layer 1: Add & Norm (Final Encoder Memory)',
        layerIndex: 1,
        operation: 'Final Residual LayerNorm',
        formula: '\\text{Final Memory} = \\text{LayerNorm}(x + \\text{FFN}(x))',
        description: 'Đầu ra hoàn chỉnh của toàn bộ Encoder Stack, đóng vai trò Memory (Keys & Values) chuyển qua Decoder.',
        status: 'available',
        inputs: [
          {
            name: 'ffn_output',
            shape: [encLen, dModel],
            values: l?.ffn_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'encoder_memory',
            shape: [encLen, dModel],
            values: l?.layer_output || [],
            dtype: 'float32',
            description: 'Bộ nhớ ngữ cảnh trọn vẹn kết nối tới tất cả các khối Cross-Attention của Decoder'
          }
        ]
      };
    }

    // ---------------- DECODER INPUT & EMBEDDING ----------------
    case 'dec_outputs': {
      const decTokens = currentDecStep?.current_tokens || ['<s>'];
      return {
        nodeId,
        nodeName: 'Outputs (Shifted Right)',
        operation: 'Target Sequence Shifted Right by 1 Position',
        formula: 'y_{<t} = [\\text{<s>}, y_1, y_2, \\dots, y_{t-1}]',
        description: 'Chuỗi mục tiêu được dịch phải 1 vị trí (bắt đầu bằng token <s>) để đảm bảo mô hình chỉ dự đoán từ tiếp theo.',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'step_index',
            shape: [1],
            values: [validStepIdx],
            dtype: 'int32'
          }
        ],
        outputs: [
          {
            name: 'shifted_tokens',
            shape: [decTokens.length],
            values: decTokens as any,
            dtype: 'string[]',
            description: 'Các token tiền tố đã sinh cho đến bước hiện tại'
          },
          {
            name: 'shifted_token_ids',
            shape: [decTokens.length],
            values: currentDecStep?.current_token_ids || [],
            dtype: 'int32'
          }
        ]
      };
    }

    case 'dec_output_embedding': {
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Output Embedding',
        operation: 'Decoder Token Embedding Lookup',
        formula: 'E(y) \\times \\sqrt{d_{model}}',
        description: 'Tra cứu vector nhúng cho các token mục tiêu của Decoder.',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'target_token_ids',
            shape: [curLen],
            values: currentDecStep?.current_token_ids || [],
            dtype: 'int32'
          }
        ],
        outputs: [
          {
            name: 'decoder_embeddings',
            shape: [curLen, dModel],
            values: currentDecStep?.embeddings || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_pe': {
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Positional Encoding',
        operation: 'Sinusoidal Positional Encoding for Target Sequence',
        formula: 'PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d})',
        description: 'Mã vị trí điều hòa sin/cos cho chuỗi Decoder.',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'target_positions',
            shape: [curLen],
            values: Array.from({ length: curLen }, (_, i) => i),
            dtype: 'int32'
          }
        ],
        outputs: [
          {
            name: 'dec_pe_matrix',
            shape: [curLen, dModel],
            values: currentDecStep?.positional_encodings || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_pe_add': {
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Add Positional Encoding (Decoder)',
        operation: 'Target Embedding + PE Addition',
        formula: 'y_0 = Embedding(y) + PE',
        description: 'Cộng vector nhúng mục tiêu với mã vị trí chuẩn bị đưa vào Decoder Stack.',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'target_embeddings',
            shape: [curLen, dModel],
            values: currentDecStep?.embeddings || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'decoder_embedded_positions',
            shape: [curLen, dModel],
            values: currentDecStep?.embedded_positions || currentDecStep?.embeddings || [],
            dtype: 'float32'
          }
        ]
      };
    }

    // ---------------- DECODER LAYER 0 ----------------
    case 'dec_layer_0_masked_mha': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      const decTokens = currentDecStep?.current_tokens || ['<s>'];

      const headDetails: AttentionHeadDetail[] = [];
      const weights = decL0?.masked_self_attention?.attention_weights || currentDecStep?.masked_self_attention?.attention_weights;
      if (weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: decL0?.masked_self_attention?.q?.[h] || currentDecStep?.masked_self_attention?.q?.[h] || [],
            k: decL0?.masked_self_attention?.k?.[h] || currentDecStep?.masked_self_attention?.k?.[h] || [],
            v: decL0?.masked_self_attention?.v?.[h] || currentDecStep?.masked_self_attention?.v?.[h] || [],
            rawScores: decL0?.masked_self_attention?.raw_scores?.[h] || [],
            weights: weights[h] || [],
            queryTokens: decTokens,
            keyTokens: decTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Masked Multi-Head Attention',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Causal Masked Self-Attention',
        formula: '\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V, \\; M_{ij} = -\\infty \\; (j > i)',
        description: 'Tự chú ý có mặt nạ nhân quả (Causal Mask) chặn hoàn toàn thông tin từ các vị trí tương lai.',
        status: 'available',
        inputs: [
          {
            name: 'causal_mask',
            shape: [curLen, curLen],
            values: currentDecStep?.causal_mask || [],
            dtype: 'float32',
            description: 'Ma trận tam giác dưới (1 = được chú ý, 0 = bị chặn -1e9)'
          }
        ],
        outputs: [
          {
            name: 'masked_attention_weights',
            shape: [numHeads, curLen, curLen],
            values: weights || [],
            dtype: 'float32',
            description: 'Trọng số chú ý có giá trị chính xác 0.0 tại tất cả các vị trí tương lai'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'dec_layer_0_addnorm_1': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Add & Norm (Sublayer 1)',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(y + \\text{MaskedAttention}(y))',
        description: 'Đường tắt kết hợp chuẩn hóa tầng sau khối Masked Self-Attention.',
        status: 'available',
        inputs: [
          {
            name: 'masked_attention_out',
            shape: [curLen, dModel],
            values: decL0?.residual_1 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'norm1_out',
            shape: [curLen, dModel],
            values: decL0?.residual_1 || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_0_cross_mha': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      const decTokens = currentDecStep?.current_tokens || ['<s>'];

      const headDetails: AttentionHeadDetail[] = [];
      const weights = decL0?.cross_attention?.attention_weights || currentDecStep?.cross_attention?.attention_weights;
      if (weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: decL0?.cross_attention?.q?.[h] || currentDecStep?.cross_attention?.q?.[h] || [],
            k: decL0?.cross_attention?.k?.[h] || currentDecStep?.cross_attention?.k?.[h] || [],
            v: decL0?.cross_attention?.v?.[h] || currentDecStep?.cross_attention?.v?.[h] || [],
            rawScores: decL0?.cross_attention?.raw_scores?.[h] || [],
            weights: weights[h] || [],
            queryTokens: decTokens,
            keyTokens: encTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Multi-Head Cross-Attention',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Encoder-Decoder Cross-Attention',
        formula: 'Q_{\\text{dec}} \\times K_{\\text{enc}}^T / \\sqrt{d_k} \\rightarrow \\text{softmax} \\times V_{\\text{enc}}',
        description: 'Cầu nối cốt lõi: Decoder truy vấn (Queries) thông tin từ bộ nhớ ngữ cảnh của Encoder (Keys & Values).',
        status: 'available',
        inputs: [
          {
            name: 'q_from_decoder',
            shape: [curLen, dModel],
            values: decL0?.residual_1 || [],
            dtype: 'float32',
            description: 'Queries lấy từ đầu ra chuẩn hóa của Masked Self-Attention Decoder'
          },
          {
            name: 'k_v_from_encoder',
            shape: [encLen, dModel],
            values: enc?.layer_2?.layer_output || [],
            dtype: 'float32',
            description: 'Keys và Values lấy từ đầu ra cuối cùng của Encoder Stack'
          }
        ],
        outputs: [
          {
            name: 'cross_attention_weights',
            shape: [numHeads, curLen, encLen],
            values: weights || [],
            dtype: 'float32',
            description: 'Ma trận chú ý thể hiện mức độ liên kết giữa từng token đích và token nguồn'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'dec_layer_0_addnorm_2': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Add & Norm (Sublayer 2)',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{CrossAttention}(x))',
        description: 'Đường tắt kết hợp LayerNorm sau khối Cross-Attention.',
        status: 'available',
        inputs: [
          {
            name: 'cross_attn_out',
            shape: [curLen, dModel],
            values: decL0?.residual_2 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'norm2_out',
            shape: [curLen, dModel],
            values: decL0?.residual_2 || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_0_ffn': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Feed Forward',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Position-wise FFN',
        formula: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
        description: 'Mạng truyền thẳng vị trí của Decoder Layer 0.',
        status: 'available',
        inputs: [
          {
            name: 'norm2_out',
            shape: [curLen, dModel],
            values: decL0?.residual_2 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'ffn_output',
            shape: [curLen, dModel],
            values: decL0?.ffn_output || currentDecStep?.ffn_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_0_addnorm_3': {
      const decL0 = currentDecStep?.layer_0;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 0: Add & Norm (Sublayer 3)',
        layerIndex: 0,
        decoderStep: validStepIdx,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{FFN}(x))',
        description: 'Hoàn tất bước tính toán của Decoder Layer 0.',
        status: 'available',
        inputs: [
          {
            name: 'ffn_out',
            shape: [curLen, dModel],
            values: decL0?.ffn_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'layer_0_output',
            shape: [curLen, dModel],
            values: decL0?.layer_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    // ---------------- DECODER LAYER 1 ----------------
    case 'dec_layer_1_masked_mha': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      const decTokens = currentDecStep?.current_tokens || ['<s>'];

      const headDetails: AttentionHeadDetail[] = [];
      const weights = decL1?.masked_self_attention?.attention_weights;
      if (weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: decL1?.masked_self_attention?.q?.[h] || [],
            k: decL1?.masked_self_attention?.k?.[h] || [],
            v: decL1?.masked_self_attention?.v?.[h] || [],
            rawScores: decL1?.masked_self_attention?.raw_scores?.[h] || [],
            weights: weights[h] || [],
            queryTokens: decTokens,
            keyTokens: decTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Masked Multi-Head Attention',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Deep Masked Self-Attention',
        formula: '\\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}} + M\\right)V',
        description: 'Tự chú ý có mặt nạ nhân quả tầng 1 của Decoder.',
        status: 'available',
        inputs: [
          {
            name: 'layer_0_output',
            shape: [curLen, dModel],
            values: currentDecStep?.layer_0?.layer_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'layer_1_masked_weights',
            shape: [numHeads, curLen, curLen],
            values: weights || [],
            dtype: 'float32'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'dec_layer_1_addnorm_1': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Add & Norm (Sublayer 1)',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{Sublayer}(x))',
        description: 'Đường tắt Residual và LayerNorm tầng 1 của Decoder.',
        status: 'available',
        inputs: [
          {
            name: 'masked_attn_out',
            shape: [curLen, dModel],
            values: decL1?.residual_1 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'norm1_out',
            shape: [curLen, dModel],
            values: decL1?.residual_1 || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_1_cross_mha': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      const decTokens = currentDecStep?.current_tokens || ['<s>'];

      const headDetails: AttentionHeadDetail[] = [];
      const weights = decL1?.cross_attention?.attention_weights;
      if (weights) {
        for (let h = 0; h < numHeads; h++) {
          headDetails.push({
            headIndex: h,
            q: decL1?.cross_attention?.q?.[h] || [],
            k: decL1?.cross_attention?.k?.[h] || [],
            v: decL1?.cross_attention?.v?.[h] || [],
            rawScores: decL1?.cross_attention?.raw_scores?.[h] || [],
            weights: weights[h] || [],
            queryTokens: decTokens,
            keyTokens: encTokens
          });
        }
      }

      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Multi-Head Cross-Attention',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Deep Encoder-Decoder Cross-Attention',
        formula: 'Q_{\\text{dec1}} \\times K_{\\text{enc}}^T / \\sqrt{d_k} \\rightarrow \\text{softmax} \\times V_{\\text{enc}}',
        description: 'Khối Cross-Attention tầng 1 kết nối biểu diễn sâu của Decoder với toàn bộ Encoder.',
        status: 'available',
        inputs: [
          {
            name: 'q_from_decoder_l1',
            shape: [curLen, dModel],
            values: decL1?.residual_1 || [],
            dtype: 'float32'
          },
          {
            name: 'encoder_memory',
            shape: [encLen, dModel],
            values: enc?.layer_2?.layer_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'cross_weights_l1',
            shape: [numHeads, curLen, encLen],
            values: weights || [],
            dtype: 'float32'
          }
        ],
        attentionHeads: headDetails
      };
    }

    case 'dec_layer_1_addnorm_2': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Add & Norm (Sublayer 2)',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Residual Addition & Layer Normalization',
        formula: '\\text{LayerNorm}(x + \\text{CrossAttention}(x))',
        description: 'Đường tắt Residual sau Cross-Attention tầng 1.',
        status: 'available',
        inputs: [
          {
            name: 'cross_attn_out',
            shape: [curLen, dModel],
            values: decL1?.residual_2 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'norm2_out',
            shape: [curLen, dModel],
            values: decL1?.residual_2 || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_1_ffn': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Feed Forward',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Position-wise FFN',
        formula: '\\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2',
        description: 'Mạng truyền thẳng tầng 1 tạo biểu diễn ẩn cuối cùng.',
        status: 'available',
        inputs: [
          {
            name: 'norm2_out',
            shape: [curLen, dModel],
            values: decL1?.residual_2 || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'ffn_output',
            shape: [curLen, dModel],
            values: decL1?.ffn_output || [],
            dtype: 'float32'
          }
        ]
      };
    }

    case 'dec_layer_1_addnorm_3': {
      const decL1 = currentDecStep?.layer_1;
      const curLen = currentDecStep?.current_tokens?.length || 1;
      return {
        nodeId,
        nodeName: 'Decoder Layer 1: Add & Norm (Final Decoder State)',
        layerIndex: 1,
        decoderStep: validStepIdx,
        operation: 'Final Decoder Layer Normalization',
        formula: 'h_{\\text{final}} = \\text{LayerNorm}(x + \\text{FFN}(x))',
        description: 'Trạng thái ẩn hoàn chỉnh của Decoder Stack chuẩn bị đưa vào phép chiếu tuyến tính Linear.',
        status: 'available',
        inputs: [
          {
            name: 'ffn_output',
            shape: [curLen, dModel],
            values: decL1?.ffn_output || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'decoder_final_hidden',
            shape: [curLen, dModel],
            values: decL1?.layer_output || currentDecStep?.final_hidden || [],
            dtype: 'float32'
          }
        ]
      };
    }

    // ---------------- FINAL LINEAR & SOFTMAX ----------------
    case 'dec_linear': {
      const vocabSize = trace.vocab?.length || 25;
      return {
        nodeId,
        nodeName: 'Linear (Output Projection)',
        operation: 'Dense Linear Projection to Vocabulary Logits',
        formula: '\\text{logits} = h_{\\text{final}} W_{\\text{vocab}} + b_{\\text{vocab}}',
        description: 'Chiếu vector d_model của vị trí cuối cùng sang không gian kích thước bằng kích thước từ điển (vocab_size).',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'last_token_hidden',
            shape: [1, dModel],
            values: (currentDecStep?.final_hidden ? [currentDecStep.final_hidden[currentDecStep.final_hidden.length - 1]] : []) as any,
            dtype: 'float32',
            description: 'Vector ẩn của token cuối cùng'
          }
        ],
        outputs: [
          {
            name: 'logits',
            shape: [vocabSize],
            values: currentDecStep?.logits || currentDecStep?.top_candidates?.map(c => c.logit) || [],
            dtype: 'float32',
            description: 'Điểm số thô (unnormalized log-odds) cho từng từ trong từ điển'
          }
        ]
      };
    }

    case 'dec_softmax': {
      const vocabSize = trace.vocab?.length || 25;
      return {
        nodeId,
        nodeName: 'Softmax',
        operation: 'Softmax Probability Normalization',
        formula: 'P(y_t = w) = \\frac{\\exp(\\text{logit}_w)}{\\sum_{v} \\exp(\\text{logit}_v)}',
        description: 'Chuyển đổi điểm số logits thành phân phối xác suất hợp lệ (tổng = 1.0, các giá trị >= 0).',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'logits',
            shape: [vocabSize],
            values: currentDecStep?.logits || [],
            dtype: 'float32'
          }
        ],
        outputs: [
          {
            name: 'probabilities',
            shape: [vocabSize],
            values: currentDecStep?.all_probabilities || [],
            dtype: 'float32',
            description: 'Xác suất của từng token trong từ điển'
          }
        ]
      };
    }

    case 'dec_probabilities': {
      return {
        nodeId,
        nodeName: 'Output Probabilities & Next Token',
        operation: 'Argmax / Greedy Sampling & Sequence Update',
        formula: 'y_t = \\arg\\max_w P(y_t = w)',
        description: 'Chọn token có xác suất cao nhất đưa vào chuỗi đầu ra và tiếp tục sinh tự hồi quy.',
        status: 'available',
        decoderStep: validStepIdx,
        inputs: [
          {
            name: 'top_5_candidates',
            shape: [5],
            values: currentDecStep?.top_candidates as any || [],
            dtype: 'TokenCandidate[]',
            description: 'Top các từ có xác suất cao nhất'
          }
        ],
        outputs: [
          {
            name: 'selected_token',
            shape: [1],
            values: [currentDecStep?.next_token || ''] as any,
            dtype: 'string',
            description: `Token chiến thắng: "${currentDecStep?.next_token}" (ID: ${currentDecStep?.next_token_id})`
          },
          {
            name: 'current_sentence',
            shape: [1],
            values: [trace.output_sentence] as any,
            dtype: 'string',
            description: 'Câu được sinh hoàn chỉnh'
          }
        ]
      };
    }

    default:
      return {
        nodeId,
        nodeName: nodeId,
        status: 'unavailable',
        unavailableReason: `Node ${nodeId} không có dữ liệu ánh xạ trong mô phỏng hiện tại.`,
        operation: 'Unknown',
        inputs: [],
        outputs: []
      };
  }
}
