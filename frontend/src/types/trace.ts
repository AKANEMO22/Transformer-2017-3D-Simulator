export type TraceSource = 'keras' | 'simulated';

export interface TraceMetadata {
  runId: string;
  source: TraceSource;
  inputText: string;
  modelName: string;
  modelConfig: {
    numEncoderLayers: number;
    numDecoderLayers: number;
    dModel: number;
    numHeads: number;
    dFF: number;
    vocabSize?: number;
  };
  createdAt: string;
  isUntrained?: boolean;
  disclaimer?: string;
}

export interface TensorData {
  name: string;
  shape: number[];
  values: number[] | number[][] | number[][][];
  dtype: string;
  description?: string;
  source?: string;
  layerIndex?: number;
  executionStep?: number;
}

export interface AttentionHeadDetail {
  headIndex: number;
  q: number[][]; // [seq_len, d_k]
  k: number[][]; // [seq_len, d_k]
  v: number[][]; // [seq_len, d_k]
  rawScores: number[][]; // [seq_len, seq_len]
  weights: number[][]; // [seq_len, seq_len]
  queryTokens: string[];
  keyTokens: string[];
}

export interface NodeExecutionTrace {
  nodeId: string;
  nodeName?: string;
  layerIndex?: number;
  decoderStep?: number;
  operation: string;
  formula?: string;
  description?: string;
  status: 'available' | 'unavailable' | 'idle';
  unavailableReason?: string;
  inputs: TensorData[];
  outputs: TensorData[];
  intermediate?: TensorData[];
  attentionHeads?: AttentionHeadDetail[];
}

export interface TimelineStep {
  stepId: string;
  nodeId: string;
  sequenceIndex: number;
  decoderStep: number;
  activeEdgeIds: string[];
  duration: number;
  label: string;
  layerIndex?: number;
  description?: string;
}

export interface TokenCandidate {
  id: number;
  token: string;
  prob: number;
  logit: number;
}

export interface LayerAttentionTrace {
  q: number[][][]; // [num_heads, seq_len, d_k]
  k: number[][][];
  v: number[][][];
  raw_scores?: number[][][];
  attention_weights: number[][][]; // [num_heads, seq_len, seq_len]
  residual_1?: number[][]; // [seq_len, d_model]
  ffn_hidden?: number[][]; // [seq_len, d_ff]
  ffn_output: number[][]; // [seq_len, d_model]
  residual_2?: number[][]; // [seq_len, d_model]
  layer_output: number[][]; // [seq_len, d_model]
}

export interface EncoderTrace {
  source_tokens: string[];
  source_ids: number[];
  embeddings: number[][]; // [seq_len, d_model]
  positional_encodings: number[][]; // [seq_len, d_model]
  embedded_positions: number[][]; // [seq_len, d_model]
  layer_1: LayerAttentionTrace;
  layer_2: LayerAttentionTrace;
}

export interface DecoderLayerTrace {
  masked_self_attention: {
    q: number[][][];
    k: number[][][];
    v: number[][][];
    raw_scores?: number[][][];
    attention_weights: number[][][]; // [num_heads, cur_len, cur_len]
  };
  residual_1?: number[][];
  cross_attention: {
    q: number[][][]; // [num_heads, cur_len, d_k]
    k: number[][][]; // [num_heads, enc_len, d_k]
    v: number[][][];
    raw_scores?: number[][][];
    attention_weights: number[][][]; // [num_heads, cur_len, enc_len]
  };
  residual_2?: number[][];
  ffn_hidden?: number[][];
  ffn_output: number[][];
  residual_3?: number[][];
  layer_output?: number[][];
}

export interface DecoderStepTrace {
  step_index: number;
  current_tokens: string[];
  current_token_ids: number[];
  embeddings: number[][];
  positional_encodings?: number[][];
  embedded_positions?: number[][];
  causal_mask: number[][]; // [cur_len, cur_len]
  // Layer 0 / Layer 1 detailed traces
  layer_0?: DecoderLayerTrace;
  layer_1?: DecoderLayerTrace;
  // Backward compatibility fields
  masked_self_attention: {
    q: number[][][];
    k: number[][][];
    v: number[][][];
    attention_weights: number[][][];
  };
  cross_attention: {
    q: number[][][];
    k: number[][][];
    v: number[][][];
    attention_weights: number[][][];
  };
  ffn_output: number[][];
  final_hidden?: number[][];
  logits?: number[];
  top_candidates: TokenCandidate[];
  next_token: string;
  next_token_id: number;
  all_probabilities: number[];
}

export interface TransformerTrace {
  metadata?: TraceMetadata;
  source_text: string;
  d_model: number;
  num_heads: number;
  d_k: number;
  d_ff: number;
  vocab: string[];
  encoder: EncoderTrace;
  decoder_steps: DecoderStepTrace[];
  final_generated_tokens: string[];
  output_sentence: string;
}

export type StageId = 
  | 'input' 
  | 'embedding' 
  | 'positional' 
  | 'encoder_attn' 
  | 'encoder_ffn' 
  | 'decoder_masked' 
  | 'cross_attn' 
  | 'softmax' 
  | 'autoregressive';

export interface StageInfo {
  id: StageId;
  index: number;
  name: string;
  shortName: string;
  formula: string;
  description: string;
  cameraPos: [number, number, number];
  cameraTarget: [number, number, number];
}
