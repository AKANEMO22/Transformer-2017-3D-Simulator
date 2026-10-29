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
  ffn_output: number[][]; // [seq_len, d_model]
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

export interface DecoderStepTrace {
  step_index: number;
  current_tokens: string[];
  current_token_ids: number[];
  embeddings: number[][];
  causal_mask: number[][]; // [cur_len, cur_len]
  masked_self_attention: {
    q: number[][][];
    k: number[][][];
    v: number[][][];
    attention_weights: number[][][]; // [num_heads, cur_len, cur_len]
  };
  cross_attention: {
    q: number[][][]; // [num_heads, cur_len, d_k]
    k: number[][][]; // [num_heads, enc_len, d_k]
    v: number[][][];
    attention_weights: number[][][]; // [num_heads, cur_len, enc_len]
  };
  ffn_output: number[][];
  top_candidates: TokenCandidate[];
  next_token: string;
  next_token_id: number;
  all_probabilities: number[];
}

export interface TransformerTrace {
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
