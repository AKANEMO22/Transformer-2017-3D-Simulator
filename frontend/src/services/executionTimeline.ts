import type { TimelineStep, TransformerTrace } from '../types/trace';

/**
 * Build a structured execution timeline adhering to Vaswani et al. 2017 Figure 1 dependencies.
 * For each decoder step, executes the full decoder forward-pass stack and outputs the next token.
 */
export function buildExecutionTimeline(
  trace: TransformerTrace | null,
  maxDecoderSteps = 4
): TimelineStep[] {
  const steps: TimelineStep[] = [];
  let seqIdx = 0;

  // ---------------- PART 1: ENCODER PIPELINE ----------------
  // 1. Source Tokens Input
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_inputs',
    decoderStep: 0,
    activeEdgeIds: ['enc_inputs->enc_input_embedding'],
    duration: 1200,
    label: '1. Tokenization',
    description: 'Tách chuỗi câu đầu vào thành các token từ vựng và tra cứu mã số định danh Token ID.'
  });

  // 2. Input Embedding
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_input_embedding',
    decoderStep: 0,
    activeEdgeIds: ['enc_inputs->enc_input_embedding', 'enc_input_embedding->enc_pe_add'],
    duration: 1200,
    label: '2. Input Embedding',
    description: 'Tra cứu vector không gian dày đặc d_model cho từng token và nhân căn bậc hai của d_model.'
  });

  // 3. Positional Encoding
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_pe',
    decoderStep: 0,
    activeEdgeIds: ['enc_pe->enc_pe_add'],
    duration: 1200,
    label: '3. Sinusoidal PE',
    description: 'Sinh ma trận hàm sóng điều hòa sin/cos thể hiện thứ tự vị trí trong chuỗi.'
  });

  // 4. Positional Addition
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_pe_add',
    decoderStep: 0,
    activeEdgeIds: ['enc_input_embedding->enc_pe_add', 'enc_pe->enc_pe_add', 'enc_pe_add->enc_layer_0_mha'],
    duration: 1200,
    label: '4. Embedding + PE',
    description: 'Cộng từng phần tử giữa Embedding và Positional Encoding tạo vector đầu vào cho Encoder Stack.'
  });

  // 5. Encoder Layer 0 - Self-Attention
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_0_mha',
    layerIndex: 0,
    decoderStep: 0,
    activeEdgeIds: ['enc_pe_add->enc_layer_0_mha', 'enc_layer_0_mha->enc_layer_0_addnorm_1'],
    duration: 1500,
    label: '5. Enc L0 Self-Attn',
    description: 'Tính toán quan hệ tương hỗ tự chú ý đa đầu giữa các token trong tầng 0.'
  });

  // 6. Encoder Layer 0 - Add & Norm 1
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_0_addnorm_1',
    layerIndex: 0,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_0_mha->enc_layer_0_addnorm_1', 'enc_pe_add->enc_layer_0_addnorm_1', 'enc_layer_0_addnorm_1->enc_layer_0_ffn'],
    duration: 1200,
    label: '6. Enc L0 Add & Norm 1',
    description: 'Đường tắt Residual bypass cộng kết quả tự chú ý và chuẩn hóa phân phối LayerNorm.'
  });

  // 7. Encoder Layer 0 - FFN
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_0_ffn',
    layerIndex: 0,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_0_addnorm_1->enc_layer_0_ffn', 'enc_layer_0_ffn->enc_layer_0_addnorm_2'],
    duration: 1200,
    label: '7. Enc L0 Feed-Forward',
    description: 'Mạng truyền thẳng vị trí kích hoạt ReLU tạo biến đổi phi tuyến độc lập tại từng vị trí.'
  });

  // 8. Encoder Layer 0 - Add & Norm 2
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_0_addnorm_2',
    layerIndex: 0,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_0_ffn->enc_layer_0_addnorm_2', 'enc_layer_0_addnorm_1->enc_layer_0_addnorm_2', 'enc_layer_0_addnorm_2->enc_layer_1_mha'],
    duration: 1200,
    label: '8. Enc L0 Add & Norm 2',
    description: 'Hoàn tất tầng Encoder 0, chuyển biểu diễn sang Encoder Layer 1.'
  });

  // 9. Encoder Layer 1 - Self-Attention
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_1_mha',
    layerIndex: 1,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_0_addnorm_2->enc_layer_1_mha', 'enc_layer_1_mha->enc_layer_1_addnorm_1'],
    duration: 1500,
    label: '9. Enc L1 Self-Attn',
    description: 'Tự chú ý tầng sâu của Encoder Layer 1 trích xuất liên kết ngữ cảnh trừu tượng.'
  });

  // 10. Encoder Layer 1 - Add & Norm 1
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_1_addnorm_1',
    layerIndex: 1,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_1_mha->enc_layer_1_addnorm_1', 'enc_layer_0_addnorm_2->enc_layer_1_addnorm_1', 'enc_layer_1_addnorm_1->enc_layer_1_ffn'],
    duration: 1200,
    label: '10. Enc L1 Add & Norm 1',
    description: 'Đường tắt Residual và chuẩn hóa LayerNorm tầng 1.'
  });

  // 11. Encoder Layer 1 - FFN
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_1_ffn',
    layerIndex: 1,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_1_addnorm_1->enc_layer_1_ffn', 'enc_layer_1_ffn->enc_layer_1_addnorm_2'],
    duration: 1200,
    label: '11. Enc L1 Feed-Forward',
    description: 'Mạng truyền thẳng tầng 1 xử lý đặc trưng phi tuyến cao cấp.'
  });

  // 12. Encoder Layer 1 - Add & Norm 2 (Final Encoder Memory Output)
  steps.push({
    stepId: `step_${seqIdx}`,
    sequenceIndex: seqIdx++,
    nodeId: 'enc_layer_1_addnorm_2',
    layerIndex: 1,
    decoderStep: 0,
    activeEdgeIds: ['enc_layer_1_ffn->enc_layer_1_addnorm_2', 'enc_layer_1_addnorm_1->enc_layer_1_addnorm_2', 'enc_layer_1_addnorm_2->dec_layer_0_cross_mha', 'enc_layer_1_addnorm_2->dec_layer_1_cross_mha'],
    duration: 1400,
    label: '12. Final Encoder Memory',
    description: 'Bộ nhớ ngữ cảnh toàn vẹn của Encoder, chuyển sang làm Keys & Values cho Decoder.'
  });

  // ---------------- PART 2: AUTOREGRESSIVE DECODER LOOP ----------------
  const actualDecStepsCount = trace?.decoder_steps?.length || maxDecoderSteps;

  for (let dStep = 0; dStep < actualDecStepsCount; dStep++) {
    // 13. Shifted-right Target Inputs
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_outputs',
      decoderStep: dStep,
      activeEdgeIds: ['dec_outputs->dec_output_embedding'],
      duration: 1000,
      label: `Dec Step ${dStep}: Shifted Inputs`,
      description: `Nhận chuỗi token mục tiêu đã sinh trước đó làm tiền đề sinh từ tiếp theo.`
    });

    // 14. Target Embedding & PE
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_output_embedding',
      decoderStep: dStep,
      activeEdgeIds: ['dec_outputs->dec_output_embedding', 'dec_output_embedding->dec_pe_add'],
      duration: 1000,
      label: `Dec Step ${dStep}: Embedding`,
      description: 'Nhúng vector mục tiêu và chuẩn bị cộng mã hóa vị trí.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_pe',
      decoderStep: dStep,
      activeEdgeIds: ['dec_pe->dec_pe_add'],
      duration: 1000,
      label: `Dec Step ${dStep}: Sinusoidal PE`,
      description: 'Mã vị trí của chuỗi mục tiêu Decoder.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_pe_add',
      decoderStep: dStep,
      activeEdgeIds: ['dec_output_embedding->dec_pe_add', 'dec_pe->dec_pe_add', 'dec_pe_add->dec_layer_0_masked_mha'],
      duration: 1000,
      label: `Dec Step ${dStep}: Add PE`,
      description: 'Cộng embedding và mã vị trí đưa vào Decoder Stack.'
    });

    // 15. Decoder Layer 0 - Masked Self-Attention
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_masked_mha',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_pe_add->dec_layer_0_masked_mha', 'dec_layer_0_masked_mha->dec_layer_0_addnorm_1'],
      duration: 1400,
      label: `Dec Step ${dStep}: L0 Masked Attn`,
      description: 'Tự chú ý có mặt nạ Causal Mask chặn hoàn toàn các token ở vị trí tương lai.'
    });

    // 16. Decoder Layer 0 - Add & Norm 1
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_addnorm_1',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_masked_mha->dec_layer_0_addnorm_1', 'dec_pe_add->dec_layer_0_addnorm_1', 'dec_layer_0_addnorm_1->dec_layer_0_cross_mha'],
      duration: 1000,
      label: `Dec Step ${dStep}: L0 Add & Norm 1`,
      description: 'Residual bypass và LayerNorm chuẩn bị Queries cho khối Cross-Attention.'
    });

    // 17. Decoder Layer 0 - Cross-Attention (Memory from Encoder)
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_cross_mha',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_addnorm_1->dec_layer_0_cross_mha', 'enc_layer_1_addnorm_2->dec_layer_0_cross_mha', 'dec_layer_0_cross_mha->dec_layer_0_addnorm_2'],
      duration: 1600,
      label: `Dec Step ${dStep}: L0 Cross-Attn`,
      description: 'Cross-Attention: Queries từ Decoder đối chiếu với Keys và Values từ Encoder.'
    });

    // 18. Decoder Layer 0 - Add & Norm 2
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_addnorm_2',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_cross_mha->dec_layer_0_addnorm_2', 'dec_layer_0_addnorm_1->dec_layer_0_addnorm_2', 'dec_layer_0_addnorm_2->dec_layer_0_ffn'],
      duration: 1000,
      label: `Dec Step ${dStep}: L0 Add & Norm 2`,
      description: 'Đường tắt sau Cross-Attention.'
    });

    // 19. Decoder Layer 0 - FFN & Add & Norm 3
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_ffn',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_addnorm_2->dec_layer_0_ffn', 'dec_layer_0_ffn->dec_layer_0_addnorm_3'],
      duration: 1000,
      label: `Dec Step ${dStep}: L0 FFN`,
      description: 'Mạng truyền thẳng của Decoder Layer 0.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_0_addnorm_3',
      layerIndex: 0,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_ffn->dec_layer_0_addnorm_3', 'dec_layer_0_addnorm_2->dec_layer_0_addnorm_3', 'dec_layer_0_addnorm_3->dec_layer_1_masked_mha'],
      duration: 1000,
      label: `Dec Step ${dStep}: L0 Add & Norm 3`,
      description: 'Chuyển biểu diễn Decoder Layer 0 lên Decoder Layer 1.'
    });

    // 20. Decoder Layer 1 - Masked Attention, Cross Attention, FFN
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_masked_mha',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_0_addnorm_3->dec_layer_1_masked_mha', 'dec_layer_1_masked_mha->dec_layer_1_addnorm_1'],
      duration: 1200,
      label: `Dec Step ${dStep}: L1 Masked Attn`,
      description: 'Masked Attention tầng 1 của Decoder.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_addnorm_1',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_masked_mha->dec_layer_1_addnorm_1', 'dec_layer_0_addnorm_3->dec_layer_1_addnorm_1', 'dec_layer_1_addnorm_1->dec_layer_1_cross_mha'],
      duration: 1000,
      label: `Dec Step ${dStep}: L1 Add & Norm 1`,
      description: 'Residual bypass tầng 1.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_cross_mha',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_addnorm_1->dec_layer_1_cross_mha', 'enc_layer_1_addnorm_2->dec_layer_1_cross_mha', 'dec_layer_1_cross_mha->dec_layer_1_addnorm_2'],
      duration: 1500,
      label: `Dec Step ${dStep}: L1 Cross-Attn`,
      description: 'Cross-Attention sâu kết hợp thông tin với Encoder memory.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_addnorm_2',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_cross_mha->dec_layer_1_addnorm_2', 'dec_layer_1_addnorm_1->dec_layer_1_addnorm_2', 'dec_layer_1_addnorm_2->dec_layer_1_ffn'],
      duration: 1000,
      label: `Dec Step ${dStep}: L1 Add & Norm 2`,
      description: 'Residual bypass sau Cross-Attention tầng 1.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_ffn',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_addnorm_2->dec_layer_1_ffn', 'dec_layer_1_ffn->dec_layer_1_addnorm_3'],
      duration: 1000,
      label: `Dec Step ${dStep}: L1 FFN`,
      description: 'Mạng truyền thẳng tầng 1 của Decoder.'
    });

    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_layer_1_addnorm_3',
      layerIndex: 1,
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_ffn->dec_layer_1_addnorm_3', 'dec_layer_1_addnorm_2->dec_layer_1_addnorm_3', 'dec_layer_1_addnorm_3->dec_linear'],
      duration: 1000,
      label: `Dec Step ${dStep}: Final Decoder State`,
      description: 'Trạng thái ẩn cuối cùng của Decoder đưa vào tầng Linear.'
    });

    // 21. Linear Layer
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_linear',
      decoderStep: dStep,
      activeEdgeIds: ['dec_layer_1_addnorm_3->dec_linear', 'dec_linear->dec_softmax'],
      duration: 1200,
      label: `Dec Step ${dStep}: Linear Logits`,
      description: 'Chiếu trạng thái ẩn d_model sang điểm số logits trên toàn bộ từ điển.'
    });

    // 22. Softmax Layer
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_softmax',
      decoderStep: dStep,
      activeEdgeIds: ['dec_linear->dec_softmax', 'dec_softmax->dec_probabilities'],
      duration: 1200,
      label: `Dec Step ${dStep}: Softmax`,
      description: 'Chuẩn hóa hàm Softmax tạo phân phối xác suất hợp lệ.'
    });

    // 23. Next Token Output Probabilities
    const currentStepTrace = trace?.decoder_steps?.[dStep];
    const generatedToken = currentStepTrace?.next_token || '...';
    steps.push({
      stepId: `step_${seqIdx}`,
      sequenceIndex: seqIdx++,
      nodeId: 'dec_probabilities',
      decoderStep: dStep,
      activeEdgeIds: ['dec_softmax->dec_probabilities'],
      duration: 1400,
      label: `Dec Step ${dStep}: Token "${generatedToken}"`,
      description: `Chọn token "${generatedToken}" (xác suất cao nhất) và cập nhật chuỗi sinh.`
    });
  }

  return steps;
}
