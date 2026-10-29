import type { TransformerTrace } from '../types/trace';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

function hasInvalidNumber(matrix: any): boolean {
  if (typeof matrix === 'number') {
    return isNaN(matrix) || !isFinite(matrix);
  }
  if (Array.isArray(matrix)) {
    return matrix.some(item => hasInvalidNumber(item));
  }
  return false;
}

export function validateTransformerTrace(trace: TransformerTrace): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  if (!trace) {
    return { valid: false, errors: ['Trace is null or undefined'], warnings: [] };
  }

  // 1. Basic configuration checks
  if (trace.d_model <= 0 || trace.num_heads <= 0) {
    errors.push(`Invalid model dimensions: d_model=${trace.d_model}, num_heads=${trace.num_heads}`);
  }
  if (trace.d_model % trace.num_heads !== 0) {
    errors.push(`d_model (${trace.d_model}) must be divisible by num_heads (${trace.num_heads})`);
  }

  // 2. Encoder checks
  if (!trace.encoder) {
    errors.push('Trace missing encoder object');
  } else {
    const enc = trace.encoder;
    if (!enc.source_tokens || !enc.source_ids) {
      errors.push('Encoder missing source_tokens or source_ids');
    } else if (enc.source_tokens.length !== enc.source_ids.length) {
      errors.push(
        `Source tokens length (${enc.source_tokens.length}) does not match source IDs length (${enc.source_ids.length})`
      );
    }

    const seqLen = enc.source_tokens ? enc.source_tokens.length : 0;
    if (seqLen === 0) {
      errors.push('Source sequence length is 0');
    }

    // Embedding shapes
    if (enc.embeddings) {
      if (enc.embeddings.length !== seqLen) {
        errors.push(`Embeddings length (${enc.embeddings.length}) != seqLen (${seqLen})`);
      }
      if (enc.embeddings[0] && enc.embeddings[0].length !== trace.d_model) {
        errors.push(`Embedding dimension (${enc.embeddings[0].length}) != d_model (${trace.d_model})`);
      }
    }

    // NaN / Inf in encoder
    if (hasInvalidNumber(enc.embeddings)) errors.push('NaN or Infinity detected in encoder embeddings');
    if (hasInvalidNumber(enc.positional_encodings)) errors.push('NaN or Infinity detected in positional encodings');

    // Encoder Layer 1 Attention weights
    if (enc.layer_1 && enc.layer_1.attention_weights) {
      const weights = enc.layer_1.attention_weights;
      if (weights.length !== trace.num_heads) {
        errors.push(`Encoder Layer 1 attention heads (${weights.length}) != num_heads (${trace.num_heads})`);
      }
      for (let h = 0; h < weights.length; h++) {
        for (let r = 0; r < weights[h].length; r++) {
          const sum = weights[h][r].reduce((acc, v) => acc + v, 0);
          if (Math.abs(sum - 1.0) > 0.05) {
            errors.push(`Encoder Layer 1 Head ${h} row ${r} attention sum (${sum.toFixed(3)}) does not equal 1.0`);
          }
        }
      }
    }
  }

  // 3. Decoder steps checks
  if (!trace.decoder_steps || trace.decoder_steps.length === 0) {
    errors.push('Decoder steps are empty');
  } else {
    const encLen = trace.encoder?.source_tokens?.length || 0;

    trace.decoder_steps.forEach((step, sIdx) => {
      const curLen = step.current_tokens?.length || 0;
      if (curLen === 0) {
        errors.push(`Decoder step ${sIdx} current_tokens is empty`);
      }

      // Causal mask check
      if (step.causal_mask) {
        for (let r = 0; r < step.causal_mask.length; r++) {
          for (let c = r + 1; c < step.causal_mask[r].length; c++) {
            if (step.causal_mask[r][c] > 0) {
              errors.push(`Decoder step ${sIdx} causal mask allows future position [${r}, ${c}]`);
            }
          }
        }
      }

      // Masked attention future zero check
      const selfWeights = step.masked_self_attention?.attention_weights;
      if (selfWeights) {
        for (let h = 0; h < selfWeights.length; h++) {
          for (let r = 0; r < selfWeights[h].length; r++) {
            for (let c = r + 1; c < selfWeights[h][r].length; c++) {
              if (selfWeights[h][r][c] > 0.001) {
                errors.push(`Decoder step ${sIdx} Head ${h} attends to future token at [${r}, ${c}] with weight ${selfWeights[h][r][c]}`);
              }
            }
          }
        }
      }

      // Cross-attention dimension check: key dimension must match Encoder output length
      const crossWeights = step.cross_attention?.attention_weights;
      if (crossWeights && crossWeights[0] && crossWeights[0][0]) {
        const keyDim = crossWeights[0][0].length;
        if (keyDim !== encLen) {
          errors.push(`Decoder step ${sIdx} Cross-Attention key dim (${keyDim}) != Encoder length (${encLen})`);
        }
      }

      // Probabilities check
      if (step.all_probabilities) {
        const pSum = step.all_probabilities.reduce((acc, v) => acc + v, 0);
        if (Math.abs(pSum - 1.0) > 0.05) {
          errors.push(`Decoder step ${sIdx} probabilities sum (${pSum.toFixed(3)}) does not equal 1.0`);
        }
        for (let p of step.all_probabilities) {
          if (p < 0 || p > 1.0) {
            errors.push(`Decoder step ${sIdx} probability out of range [0, 1]: ${p}`);
          }
        }
      }

      if (hasInvalidNumber(step)) {
        errors.push(`NaN or Infinity detected in decoder step ${sIdx}`);
      }
    });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}
