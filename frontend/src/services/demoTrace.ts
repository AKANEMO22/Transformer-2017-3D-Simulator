import type {
  TransformerTrace,
  EncoderTrace,
  DecoderStepTrace,
  DecoderLayerTrace,
  LayerAttentionTrace,
  TraceMetadata,
  TokenCandidate
} from '../types/trace';

// Seeded pseudo-random number generator (Mulberry32) for deterministic reproducible weights
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Matrix operations
function createRandomMatrix(rows: number, cols: number, rng: () => number, scale = 0.5): number[][] {
  const mat: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      // Normal-ish distribution via sum of uniforms
      const u = (rng() + rng() + rng() - 1.5) * scale;
      row.push(parseFloat(u.toFixed(4)));
    }
    mat.push(row);
  }
  return mat;
}

function matmul(a: number[][], b: number[][]): number[][] {
  const rows = a.length;
  const cols = b[0].length;
  const inner = b.length;
  const out: number[][] = [];

  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) {
        sum += a[r][k] * b[k][c];
      }
      row.push(parseFloat(sum.toFixed(4)));
    }
    out.push(row);
  }
  return out;
}

function transpose(m: number[][]): number[][] {
  const rows = m.length;
  const cols = m[0].length;
  const out: number[][] = [];
  for (let c = 0; c < cols; c++) {
    const row: number[] = [];
    for (let r = 0; r < rows; r++) {
      row.push(m[r][c]);
    }
    out.push(row);
  }
  return out;
}

function vectorAdd(a: number[], b: number[]): number[] {
  return a.map((val, i) => parseFloat((val + (b[i] || 0)).toFixed(4)));
}

function matrixAdd(a: number[][], b: number[][]): number[][] {
  return a.map((row, r) => row.map((val, c) => parseFloat((val + b[r][c]).toFixed(4))));
}

function layerNorm(matrix: number[][], eps = 1e-5): number[][] {
  return matrix.map(row => {
    const mean = row.reduce((acc, v) => acc + v, 0) / row.length;
    const variance = row.reduce((acc, v) => acc + (v - mean) ** 2, 0) / row.length;
    const std = Math.sqrt(variance + eps);
    return row.map(v => parseFloat(((v - mean) / std).toFixed(4)));
  });
}

function softmax(arr: number[]): number[] {
  const max = Math.max(...arr);
  const exp = arr.map(v => Math.exp(v - max));
  const sum = exp.reduce((acc, v) => acc + v, 0) || 1e-9;
  return exp.map(v => parseFloat((v / sum).toFixed(4)));
}

// Fixed Vocabulary
export const EDUCATIONAL_VOCAB = [
  "<pad>", "<s>", "</s>",
  "lắng", "nghe", "ý", "kiến", "nhóm", "giải", "quyết", "mâu", "thuẫn",
  "em", "sẽ", "cùng", "mọi", "người", "thảo", "luận", "tìm", "hiểu", "nguyên", "nhân",
  "python", "sql", "code", "mô", "hình", "machine", "learning",
  "thấu", "hiểu", "đồng", "thuận", "hòa", "hợp", "kết", "quả", "hợp", "tác"
];

const VOCAB_MAP = new Map<string, number>();
EDUCATIONAL_VOCAB.forEach((word, idx) => VOCAB_MAP.set(word.toLowerCase(), idx));

export function tokenizeEducational(text: string, maxLen = 5): { tokens: string[]; tokenIds: number[] } {
  if (!text || !text.trim()) {
    const defaultTokens = ["lắng", "nghe", "ý", "kiến"];
    return {
      tokens: defaultTokens,
      tokenIds: defaultTokens.map(t => VOCAB_MAP.get(t) ?? 3)
    };
  }

  const cleaned = text
    .toLowerCase()
    .replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?"'–—]/g, " ")
    .trim()
    .split(/\s+/)
    .filter(Boolean);

  const tokens: string[] = [];
  const tokenIds: number[] = [];

  for (const word of cleaned) {
    if (tokens.length >= maxLen) break;
    if (VOCAB_MAP.has(word)) {
      tokens.push(word);
      tokenIds.push(VOCAB_MAP.get(word)!);
    } else {
      // Deterministically map arbitrary word to vocabulary using character code sum
      let charSum = 0;
      for (let i = 0; i < word.length; i++) {
        charSum += word.charCodeAt(i);
      }
      const mappedIdx = 3 + (charSum % (EDUCATIONAL_VOCAB.length - 3));
      tokens.push(word);
      tokenIds.push(mappedIdx);
    }
  }

  if (tokens.length === 0) {
    const fallback = ["lắng", "nghe", "ý", "kiến"];
    return {
      tokens: fallback,
      tokenIds: fallback.map(t => VOCAB_MAP.get(t) ?? 3)
    };
  }

  return { tokens, tokenIds };
}

// Compute Sinusoidal Positional Encoding (Vaswani et al. 2017)
export function computeSinusoidalPE(seqLen: number, dModel: number): number[][] {
  const pe: number[][] = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row: number[] = [];
    for (let i = 0; i < dModel; i++) {
      if (i % 2 === 0) {
        const div = Math.pow(10000, i / dModel);
        row.push(parseFloat(Math.sin(pos / div).toFixed(4)));
      } else {
        const div = Math.pow(10000, (i - 1) / dModel);
        row.push(parseFloat(Math.cos(pos / div).toFixed(4)));
      }
    }
    pe.push(row);
  }
  return pe;
}

// Multi-Head Attention computation
function computeMHA(
  qIn: number[][],
  kIn: number[][],
  vIn: number[][],
  dModel: number,
  numHeads: number,
  wQ: number[][],
  wK: number[][],
  wV: number[][],
  wO: number[][],
  mask: number[][] | null
): {
  qHeads: number[][][];
  kHeads: number[][][];
  vHeads: number[][][];
  rawScores: number[][][];
  attnWeights: number[][][];
  output: number[][];
} {
  const dK = dModel / numHeads;
  const qProj = matmul(qIn, wQ);
  const kProj = matmul(kIn, wK);
  const vProj = matmul(vIn, wV);

  const qHeads: number[][][] = [];
  const kHeads: number[][][] = [];
  const vHeads: number[][][] = [];
  const rawScores: number[][][] = [];
  const attnWeights: number[][][] = [];
  const headOutputs: number[][][] = [];

  const qLen = qIn.length;
  const kLen = kIn.length;

  for (let h = 0; h < numHeads; h++) {
    const startCol = h * dK;
    const endCol = (h + 1) * dK;

    const qH = qProj.map(row => row.slice(startCol, endCol));
    const kH = kProj.map(row => row.slice(startCol, endCol));
    const vH = vProj.map(row => row.slice(startCol, endCol));

    qHeads.push(qH);
    kHeads.push(kH);
    vHeads.push(vH);

    // Q * K^T / sqrt(d_k)
    const kH_T = transpose(kH);
    const scores = matmul(qH, kH_T);
    const scale = Math.sqrt(dK);
    const scaledScores: number[][] = [];
    const weights: number[][] = [];

    for (let r = 0; r < qLen; r++) {
      const scaledRow: number[] = [];
      const scoreForSoftmax: number[] = [];

      for (let c = 0; c < kLen; c++) {
        const val = scores[r][c] / scale;
        scaledRow.push(parseFloat(val.toFixed(4)));

        if (mask && mask[r] && mask[r][c] === 0) {
          scoreForSoftmax.push(-1e9); // Causal mask
        } else {
          scoreForSoftmax.push(val);
        }
      }
      scaledScores.push(scaledRow);
      weights.push(softmax(scoreForSoftmax));
    }

    rawScores.push(scaledScores);
    attnWeights.push(weights);

    // Weighted sum with V
    const headContext = matmul(weights, vH);
    headOutputs.push(headContext);
  }

  // Concatenate head outputs along embedding dimension
  const concatContext: number[][] = [];
  for (let r = 0; r < qLen; r++) {
    const row: number[] = [];
    for (let h = 0; h < numHeads; h++) {
      row.push(...headOutputs[h][r]);
    }
    concatContext.push(row);
  }

  // Final linear projection
  const output = matmul(concatContext, wO);

  return {
    qHeads,
    kHeads,
    vHeads,
    rawScores,
    attnWeights,
    output
  };
}

// Position-wise Feed Forward Network: max(0, xW1 + b1)W2 + b2
function computeFFN(
  x: number[][],
  w1: number[][],
  b1: number[],
  w2: number[][],
  b2: number[]
): { hidden: number[][]; output: number[][] } {
  const hidden: number[][] = [];
  for (const row of x) {
    const projected = vectorAdd(matmul([row], w1)[0], b1);
    // ReLU activation
    const reluRow = projected.map(v => Math.max(0, v));
    hidden.push(reluRow);
  }

  const output: number[][] = [];
  for (const hRow of hidden) {
    const outRow = vectorAdd(matmul([hRow], w2)[0], b2);
    output.push(outRow);
  }

  return { hidden, output };
}

/**
 * Generate a mathematically coherent educational Transformer trace.
 * Uses a documented deterministic seed (42) to establish stable, persistent model weights.
 */
export function generateDemoTrace(inputText: string): TransformerTrace {
  const dModel = 8;
  const numHeads = 2;
  const dK = dModel / numHeads;
  const dFF = 16;
  const vocab = [...EDUCATIONAL_VOCAB];
  const vocabSize = vocab.length;

  // Initialize deterministic PRNG with fixed seed for stable model weights across requests
  const modelWeightRng = mulberry32(4242);

  // Model parameters (weights)
  const embeddingTable = createRandomMatrix(vocabSize, dModel, modelWeightRng, 0.4);

  // Encoder Layer 0 weights
  const enc0_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc0_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc0_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc0_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc0_wFF1 = createRandomMatrix(dModel, dFF, modelWeightRng);
  const enc0_bFF1 = new Array(dFF).fill(0.05);
  const enc0_wFF2 = createRandomMatrix(dFF, dModel, modelWeightRng);
  const enc0_bFF2 = new Array(dModel).fill(0.0);

  // Encoder Layer 1 weights
  const enc1_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc1_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc1_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc1_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const enc1_wFF1 = createRandomMatrix(dModel, dFF, modelWeightRng);
  const enc1_bFF1 = new Array(dFF).fill(0.05);
  const enc1_wFF2 = createRandomMatrix(dFF, dModel, modelWeightRng);
  const enc1_bFF2 = new Array(dModel).fill(0.0);

  // Decoder Layer 0 weights
  const dec0_self_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_self_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_self_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_self_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_cross_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_cross_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_cross_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_cross_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec0_wFF1 = createRandomMatrix(dModel, dFF, modelWeightRng);
  const dec0_bFF1 = new Array(dFF).fill(0.05);
  const dec0_wFF2 = createRandomMatrix(dFF, dModel, modelWeightRng);
  const dec0_bFF2 = new Array(dModel).fill(0.0);

  // Decoder Layer 1 weights
  const dec1_self_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_self_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_self_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_self_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_cross_wQ = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_cross_wK = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_cross_wV = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_cross_wO = createRandomMatrix(dModel, dModel, modelWeightRng);
  const dec1_wFF1 = createRandomMatrix(dModel, dFF, modelWeightRng);
  const dec1_bFF1 = new Array(dFF).fill(0.05);
  const dec1_wFF2 = createRandomMatrix(dFF, dModel, modelWeightRng);
  const dec1_bFF2 = new Array(dModel).fill(0.0);

  // Final Linear Projection
  const finalLinearW = createRandomMatrix(dModel, vocabSize, modelWeightRng, 0.3);
  const finalLinearB = new Array(vocabSize).fill(0.0);

  // 1. Tokenize Input
  const { tokens: srcTokens, tokenIds: srcIds } = tokenizeEducational(inputText, 5);
  const seqLen = srcTokens.length;

  // 2. Embedding Lookup with scaling sqrt(d_model)
  const embeddingScale = Math.sqrt(dModel);
  const embeddings: number[][] = srcIds.map(id => {
    const raw = embeddingTable[id % vocabSize];
    return raw.map(v => parseFloat((v * embeddingScale).toFixed(4)));
  });

  // 3. Positional Encoding
  const pe = computeSinusoidalPE(seqLen, dModel);
  const embeddedPositions = matrixAdd(embeddings, pe);

  // 4. Encoder Layer 0 (MHA -> Add & Norm -> FFN -> Add & Norm)
  const enc0_mha = computeMHA(
    embeddedPositions,
    embeddedPositions,
    embeddedPositions,
    dModel,
    numHeads,
    enc0_wQ,
    enc0_wK,
    enc0_wV,
    enc0_wO,
    null
  );
  const enc0_res1 = matrixAdd(embeddedPositions, enc0_mha.output);
  const enc0_norm1 = layerNorm(enc0_res1);

  const enc0_ffn = computeFFN(enc0_norm1, enc0_wFF1, enc0_bFF1, enc0_wFF2, enc0_bFF2);
  const enc0_res2 = matrixAdd(enc0_norm1, enc0_ffn.output);
  const enc0_norm2 = layerNorm(enc0_res2);

  const encLayer0Trace: LayerAttentionTrace = {
    q: enc0_mha.qHeads,
    k: enc0_mha.kHeads,
    v: enc0_mha.vHeads,
    raw_scores: enc0_mha.rawScores,
    attention_weights: enc0_mha.attnWeights,
    residual_1: enc0_res1,
    ffn_hidden: enc0_ffn.hidden,
    ffn_output: enc0_ffn.output,
    residual_2: enc0_res2,
    layer_output: enc0_norm2
  };

  // 5. Encoder Layer 1 (Takes Layer 0 output)
  const enc1_mha = computeMHA(
    enc0_norm2,
    enc0_norm2,
    enc0_norm2,
    dModel,
    numHeads,
    enc1_wQ,
    enc1_wK,
    enc1_wV,
    enc1_wO,
    null
  );
  const enc1_res1 = matrixAdd(enc0_norm2, enc1_mha.output);
  const enc1_norm1 = layerNorm(enc1_res1);

  const enc1_ffn = computeFFN(enc1_norm1, enc1_wFF1, enc1_bFF1, enc1_wFF2, enc1_bFF2);
  const enc1_res2 = matrixAdd(enc1_norm1, enc1_ffn.output);
  const enc1_norm2 = layerNorm(enc1_res2);

  const encLayer1Trace: LayerAttentionTrace = {
    q: enc1_mha.qHeads,
    k: enc1_mha.kHeads,
    v: enc1_mha.vHeads,
    raw_scores: enc1_mha.rawScores,
    attention_weights: enc1_mha.attnWeights,
    residual_1: enc1_res1,
    ffn_hidden: enc1_ffn.hidden,
    ffn_output: enc1_ffn.output,
    residual_2: enc1_res2,
    layer_output: enc1_norm2
  };

  const encFinalOutput = enc1_norm2;

  const encoderTrace: EncoderTrace = {
    source_tokens: srcTokens,
    source_ids: srcIds,
    embeddings,
    positional_encodings: pe,
    embedded_positions: embeddedPositions,
    layer_1: encLayer0Trace,
    layer_2: encLayer1Trace
  };

  // 6. Autoregressive Decoder Simulation (Max 4 steps)
  const startId = VOCAB_MAP.get("<s>") ?? 1;
  const currentTgtIds = [startId];
  const decoderSteps: DecoderStepTrace[] = [];
  const maxDecodeSteps = 4;

  for (let step = 0; step < maxDecodeSteps; step++) {
    const curLen = currentTgtIds.length;
    const currentTokens = currentTgtIds.map(id => vocab[id] || `<unk_${id}>`);

    // Target Embeddings + Positional Encoding
    const tgtEmbeddings = currentTgtIds.map(id => {
      const raw = embeddingTable[id % vocabSize];
      return raw.map(v => parseFloat((v * embeddingScale).toFixed(4)));
    });
    const decPE = computeSinusoidalPE(curLen, dModel);
    const decEmbeddedPositions = matrixAdd(tgtEmbeddings, decPE);

    // Causal Mask: Lower triangular matrix
    const causalMask: number[][] = [];
    for (let r = 0; r < curLen; r++) {
      const maskRow: number[] = [];
      for (let c = 0; c < curLen; c++) {
        maskRow.push(c <= r ? 1 : 0);
      }
      causalMask.push(maskRow);
    }

    // Decoder Layer 0
    // 1. Masked Self-Attention
    const dec0_self = computeMHA(
      decEmbeddedPositions,
      decEmbeddedPositions,
      decEmbeddedPositions,
      dModel,
      numHeads,
      dec0_self_wQ,
      dec0_self_wK,
      dec0_self_wV,
      dec0_self_wO,
      causalMask
    );
    const dec0_res1 = matrixAdd(decEmbeddedPositions, dec0_self.output);
    const dec0_norm1 = layerNorm(dec0_res1);

    // 2. Cross-Attention: Q from Decoder, K & V from Encoder Final Output!
    const dec0_cross = computeMHA(
      dec0_norm1,
      encFinalOutput,
      encFinalOutput,
      dModel,
      numHeads,
      dec0_cross_wQ,
      dec0_cross_wK,
      dec0_cross_wV,
      dec0_cross_wO,
      null
    );
    const dec0_res2 = matrixAdd(dec0_norm1, dec0_cross.output);
    const dec0_norm2 = layerNorm(dec0_res2);

    // 3. FFN
    const dec0_ffn = computeFFN(dec0_norm2, dec0_wFF1, dec0_bFF1, dec0_wFF2, dec0_bFF2);
    const dec0_res3 = matrixAdd(dec0_norm2, dec0_ffn.output);
    const dec0_norm3 = layerNorm(dec0_res3);

    const dec0Trace: DecoderLayerTrace = {
      masked_self_attention: {
        q: dec0_self.qHeads,
        k: dec0_self.kHeads,
        v: dec0_self.vHeads,
        raw_scores: dec0_self.rawScores,
        attention_weights: dec0_self.attnWeights
      },
      residual_1: dec0_res1,
      cross_attention: {
        q: dec0_cross.qHeads,
        k: dec0_cross.kHeads,
        v: dec0_cross.vHeads,
        raw_scores: dec0_cross.rawScores,
        attention_weights: dec0_cross.attnWeights
      },
      residual_2: dec0_res2,
      ffn_hidden: dec0_ffn.hidden,
      ffn_output: dec0_ffn.output,
      residual_3: dec0_res3,
      layer_output: dec0_norm3
    };

    // Decoder Layer 1
    const dec1_self = computeMHA(
      dec0_norm3,
      dec0_norm3,
      dec0_norm3,
      dModel,
      numHeads,
      dec1_self_wQ,
      dec1_self_wK,
      dec1_self_wV,
      dec1_self_wO,
      causalMask
    );
    const dec1_res1 = matrixAdd(dec0_norm3, dec1_self.output);
    const dec1_norm1 = layerNorm(dec1_res1);

    const dec1_cross = computeMHA(
      dec1_norm1,
      encFinalOutput,
      encFinalOutput,
      dModel,
      numHeads,
      dec1_cross_wQ,
      dec1_cross_wK,
      dec1_cross_wV,
      dec1_cross_wO,
      null
    );
    const dec1_res2 = matrixAdd(dec1_norm1, dec1_cross.output);
    const dec1_norm2 = layerNorm(dec1_res2);

    const dec1_ffn = computeFFN(dec1_norm2, dec1_wFF1, dec1_bFF1, dec1_wFF2, dec1_bFF2);
    const dec1_res3 = matrixAdd(dec1_norm2, dec1_ffn.output);
    const dec1_norm3 = layerNorm(dec1_res3);

    const dec1Trace: DecoderLayerTrace = {
      masked_self_attention: {
        q: dec1_self.qHeads,
        k: dec1_self.kHeads,
        v: dec1_self.vHeads,
        raw_scores: dec1_self.rawScores,
        attention_weights: dec1_self.attnWeights
      },
      residual_1: dec1_res1,
      cross_attention: {
        q: dec1_cross.qHeads,
        k: dec1_cross.kHeads,
        v: dec1_cross.vHeads,
        raw_scores: dec1_cross.rawScores,
        attention_weights: dec1_cross.attnWeights
      },
      residual_2: dec1_res2,
      ffn_hidden: dec1_ffn.hidden,
      ffn_output: dec1_ffn.output,
      residual_3: dec1_res3,
      layer_output: dec1_norm3
    };

    // Linear projection on the last token's representation
    const lastHidden = dec1_norm3[curLen - 1];
    const logits = vectorAdd(matmul([lastHidden], finalLinearW)[0], finalLinearB);
    const probs = softmax(logits);

    // Candidates
    const candidates: TokenCandidate[] = probs
      .map((p, idx) => ({
        id: idx,
        token: vocab[idx],
        prob: parseFloat(p.toFixed(4)),
        logit: parseFloat(logits[idx].toFixed(4))
      }))
      .sort((a, b) => b.prob - a.prob);

    // Choose next token (exclude special tokens <pad>, <s> for text continuation)
    let selectedCandidate = candidates[0];
    if (selectedCandidate.token === "<pad>" || selectedCandidate.token === "<s>") {
      selectedCandidate = candidates[1] || candidates[0];
    }

    const nextToken = selectedCandidate.token;
    const nextTokenId = selectedCandidate.id;

    decoderSteps.push({
      step_index: step,
      current_tokens: currentTokens,
      current_token_ids: [...currentTgtIds],
      embeddings: tgtEmbeddings,
      positional_encodings: decPE,
      embedded_positions: decEmbeddedPositions,
      causal_mask: causalMask,
      layer_0: dec0Trace,
      layer_1: dec1Trace,
      // Backward compatibility top-level fields
      masked_self_attention: {
        q: dec0_self.qHeads,
        k: dec0_self.kHeads,
        v: dec0_self.vHeads,
        attention_weights: dec0_self.attnWeights
      },
      cross_attention: {
        q: dec0_cross.qHeads,
        k: dec0_cross.kHeads,
        v: dec0_cross.vHeads,
        attention_weights: dec0_cross.attnWeights
      },
      ffn_output: dec0_ffn.output,
      final_hidden: dec1_norm3,
      logits,
      top_candidates: candidates.slice(0, 5),
      next_token: nextToken,
      next_token_id: nextTokenId,
      all_probabilities: probs
    });

    currentTgtIds.push(nextTokenId);
    if (nextToken === "</s>") {
      break;
    }
  }

  const finalGeneratedTokens = currentTgtIds.map(id => vocab[id] || `<unk_${id}>`);
  const outputSentence = finalGeneratedTokens
    .filter(t => t !== "<s>" && t !== "</s>" && t !== "<pad>")
    .join(" ");

  const metadata: TraceMetadata = {
    runId: `sim_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`,
    source: 'simulated',
    inputText,
    modelName: 'Educational Simulator (Vaswani 2017)',
    modelConfig: {
      numEncoderLayers: 2,
      numDecoderLayers: 2,
      dModel,
      numHeads,
      dFF,
      vocabSize
    },
    createdAt: new Date().toISOString(),
    isUntrained: true,
    disclaimer: 'Educational simulation with deterministic seeded weights. Demonstrates exact Figure 1 tensor operations and causal masking.'
  };

  return {
    metadata,
    source_text: inputText,
    d_model: dModel,
    num_heads: numHeads,
    d_k: dK,
    d_ff: dFF,
    vocab,
    encoder: encoderTrace,
    decoder_steps: decoderSteps,
    final_generated_tokens: finalGeneratedTokens,
    output_sentence: outputSentence || "đồng thuận"
  };
}
