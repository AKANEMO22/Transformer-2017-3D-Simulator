import { describe, it, expect } from 'vitest';
import { generateDemoTrace } from '../services/demoTrace';
import { validateTransformerTrace } from '../services/traceValidator';
import { getNodeDetail } from '../utils/nodeDetailMapper';
import { buildExecutionTimeline } from '../services/executionTimeline';
import specData from '../spec/transformer_2017_spec.json';

describe('Transformer Frontend Test Suite', () => {
  // Test 1: Same input -> same simulated trace (deterministic)
  it('produces identical deterministic traces for identical input', () => {
    const input = 'Lắng nghe ý kiến nhóm';
    const trace1 = generateDemoTrace(input);
    const trace2 = generateDemoTrace(input);

    expect(trace1.encoder.source_ids).toEqual(trace2.encoder.source_ids);
    expect(trace1.encoder.embeddings[0]).toEqual(trace2.encoder.embeddings[0]);
    expect(trace1.encoder.layer_1.attention_weights[0][0]).toEqual(
      trace2.encoder.layer_1.attention_weights[0][0]
    );
    expect(trace1.decoder_steps[0].next_token_id).toEqual(
      trace2.decoder_steps[0].next_token_id
    );
  });

  // Test 2: Different valid input -> appropriately different trace
  it('produces distinct traces for different inputs', () => {
    const traceA = generateDemoTrace('Lắng nghe ý kiến');
    const traceB = generateDemoTrace('Python sql code');

    expect(traceA.encoder.source_tokens).not.toEqual(traceB.encoder.source_tokens);
    expect(traceA.encoder.source_ids).not.toEqual(traceB.encoder.source_ids);
    expect(traceA.encoder.embeddings).not.toEqual(traceB.encoder.embeddings);
  });

  // Test 3: Attention rows normalize correctly (sum to 1.0)
  it('ensures attention weight matrix rows sum approximately to 1.0', () => {
    const trace = generateDemoTrace('Thảo luận giải quyết');
    const encL1Weights = trace.encoder.layer_1.attention_weights;

    for (const head of encL1Weights) {
      for (const row of head) {
        const sum = row.reduce((acc, v) => acc + v, 0);
        expect(sum).toBeCloseTo(1.0, 2);
      }
    }
  });

  // Test 4: Causal mask blocks future positions
  it('strictly blocks future token attention via causal mask', () => {
    const trace = generateDemoTrace('Lắng nghe');
    // Inspect later decoder step with cur_len >= 2
    const step = trace.decoder_steps[1];
    expect(step.current_tokens.length).toBeGreaterThanOrEqual(2);

    const maskedWeights = step.masked_self_attention.attention_weights;
    for (const head of maskedWeights) {
      // Position 0 cannot attend to position 1
      expect(head[0][1]).toBe(0);
    }
  });

  // Test 5: Cross-attention shapes match encoder length
  it('matches cross-attention key length to encoder sequence length', () => {
    const trace = generateDemoTrace('Ý kiến nhóm giải quyết');
    const encLen = trace.encoder.source_tokens.length;
    const step = trace.decoder_steps[0];
    const crossWeights = step.cross_attention.attention_weights;

    for (const head of crossWeights) {
      for (const row of head) {
        expect(row.length).toBe(encLen);
      }
    }
  });

  // Test 6: All 31 architecture nodes are inspectable
  it('supports inspection for all 31 architecture nodes in spec', () => {
    const trace = generateDemoTrace('Lắng nghe ý kiến nhóm');
    expect(specData.nodes.length).toBe(31);

    for (const node of specData.nodes) {
      const detail = getNodeDetail(node.id, trace, 0);
      expect(detail.status).toBe('available');
      expect(detail.nodeId).toBe(node.id);
      expect(detail.operation).toBeDefined();
      expect(detail.inputs.length).toBeGreaterThanOrEqual(1);
      expect(detail.outputs.length).toBeGreaterThanOrEqual(1);
    }
  });

  // Test 7: Each node maps to the correct layer
  it('correctly associates nodes with their designated layer indices', () => {
    const trace = generateDemoTrace('Lắng nghe');
    const encL0 = getNodeDetail('enc_layer_0_mha', trace, 0);
    const encL1 = getNodeDetail('enc_layer_1_mha', trace, 0);
    const decL0 = getNodeDetail('dec_layer_0_cross_mha', trace, 0);
    const decL1 = getNodeDetail('dec_layer_1_cross_mha', trace, 0);

    expect(encL0.layerIndex).toBe(0);
    expect(encL1.layerIndex).toBe(1);
    expect(decL0.layerIndex).toBe(0);
    expect(decL1.layerIndex).toBe(1);

    // Verify tensors are not identical between layer 0 and layer 1
    expect(encL0.outputs[0].values).not.toEqual(encL1.outputs[0].values);
  });

  // Test 8: Timeline order follows architectural dependencies
  it('generates a timeline adhering strictly to execution order', () => {
    const trace = generateDemoTrace('Lắng nghe ý kiến');
    const timeline = buildExecutionTimeline(trace, 4);

    const nodeOrder = timeline.map(s => s.nodeId);
    const encInputIdx = nodeOrder.indexOf('enc_inputs');
    const encEmbIdx = nodeOrder.indexOf('enc_input_embedding');
    const encL0MhaIdx = nodeOrder.indexOf('enc_layer_0_mha');
    const encL0NormIdx = nodeOrder.indexOf('enc_layer_0_addnorm_1');
    const encL1Norm2Idx = nodeOrder.indexOf('enc_layer_1_addnorm_2');
    const decLinearIdx = nodeOrder.indexOf('dec_linear');
    const decSoftmaxIdx = nodeOrder.indexOf('dec_softmax');
    const decProbIdx = nodeOrder.indexOf('dec_probabilities');

    expect(encInputIdx).toBeLessThan(encEmbIdx);
    expect(encEmbIdx).toBeLessThan(encL0MhaIdx);
    expect(encL0MhaIdx).toBeLessThan(encL0NormIdx);
    expect(encL0NormIdx).toBeLessThan(encL1Norm2Idx);
    expect(encL1Norm2Idx).toBeLessThan(decLinearIdx);
    expect(decLinearIdx).toBeLessThan(decSoftmaxIdx);
    expect(decSoftmaxIdx).toBeLessThan(decProbIdx);
  });

  // Test 9: Changing decoder step updates inspected tensors
  it('updates inspected tensors when changing decoder step', () => {
    const trace = generateDemoTrace('Lắng nghe ý kiến');
    const step0Detail = getNodeDetail('dec_outputs', trace, 0);
    const step1Detail = getNodeDetail('dec_outputs', trace, 1);

    expect(step0Detail.decoderStep).toBe(0);
    expect(step1Detail.decoderStep).toBe(1);

    const step0Len = (step0Detail.outputs[0].values as unknown as string[]).length;
    const step1Len = (step1Detail.outputs[0].values as unknown as string[]).length;
    expect(step1Len).toBeGreaterThan(step0Len);
  });

  // Test 10: Simulated traces cannot be labeled as genuine Keras output
  it('enforces transparent provenance and validation rejection of malformed traces', () => {
    const simTrace = generateDemoTrace('Lắng nghe');
    expect(simTrace.metadata?.source).toBe('simulated');
    expect(simTrace.metadata?.isUntrained).toBe(true);

    // Validation passes for compliant trace
    const validation = validateTransformerTrace(simTrace);
    expect(validation.valid).toBe(true);
    expect(validation.errors.length).toBe(0);

    // Malformed trace with NaN is rejected
    const badTrace = JSON.parse(JSON.stringify(simTrace));
    badTrace.encoder.embeddings[0][0] = NaN;
    const badValidation = validateTransformerTrace(badTrace);
    expect(badValidation.valid).toBe(false);
    expect(badValidation.errors.some(e => e.includes('NaN'))).toBe(true);
  });

  // Test 11: Explicit unavailable state for missing trace or unknown node
  it('returns explicit unavailable state when trace is null or node is unknown', () => {
    const nullTraceDetail = getNodeDetail('enc_inputs', null, 0);
    expect(nullTraceDetail.status).toBe('unavailable');

    const trace = generateDemoTrace('Lắng nghe');
    const unknownNodeDetail = getNodeDetail('non_existent_node_99', trace, 0);
    expect(unknownNodeDetail.status).toBe('unavailable');
  });
});
