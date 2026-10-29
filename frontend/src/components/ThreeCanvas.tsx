import React, { useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import type { TransformerTrace, StageInfo } from '../types/trace';

import { StageInputTokens } from './stages/StageInputTokens';
import { StageEmbedding } from './stages/StageEmbedding';
import { StagePositionalEncoding } from './stages/StagePositionalEncoding';
import { StageEncoderAttention } from './stages/StageEncoderAttention';
import { StageEncoderFFN } from './stages/StageEncoderFFN';
import { StageDecoderMasked } from './stages/StageDecoderMasked';
import { StageCrossAttention } from './stages/StageCrossAttention';
import { StageSoftmaxGeneration } from './stages/StageSoftmaxGeneration';
import { StageAutoregressiveSummary } from './stages/StageAutoregressiveSummary';

export const STAGES: StageInfo[] = [
  {
    id: 'input',
    index: 0,
    name: '1. Input & Tokenization',
    shortName: 'Input',
    formula: 'Sentence → [t_0, t_1, ..., t_n]',
    description: 'Raw text is segmented into discrete vocabulary indices.',
    cameraPos: [0, 1.2, 7.5],
    cameraTarget: [0, 0.4, 0]
  },
  {
    id: 'embedding',
    index: 1,
    name: '2. Token Embedding',
    shortName: 'Embedding',
    formula: 'x_i = Embedding(t_i) ∈ ℝ^{d_model}',
    description: 'Each discrete token is projected into an 8-dimensional dense vector.',
    cameraPos: [0, 2.0, 9.5],
    cameraTarget: [0, 0.6, 0]
  },
  {
    id: 'positional',
    index: 2,
    name: '3. Positional Encoding',
    shortName: 'PE (sin/cos)',
    formula: 'PE(pos, 2i) = sin(pos/10000^{2i/d}), PE(pos, 2i+1) = cos(pos/10000^{2i/d})',
    description: 'Harmonic wave signals inject positional order into vectors.',
    cameraPos: [0, 2.2, 10.0],
    cameraTarget: [0, 0.6, 0]
  },
  {
    id: 'encoder_attn',
    index: 3,
    name: '4. Encoder Multi-Head Self-Attention',
    shortName: 'Encoder Attn',
    formula: 'Attention(Q, K, V) = softmax(Q K^T / √d_k) V',
    description: 'Tokens compute pairwise attention scores in parallel across multiple heads.',
    cameraPos: [0, 2.4, 9.5],
    cameraTarget: [0, 0.5, 0]
  },
  {
    id: 'encoder_ffn',
    index: 4,
    name: '5. Encoder Feed-Forward & Residuals',
    shortName: 'Encoder FFN',
    formula: 'FFN(x) = max(0, x W_1 + b_1) W_2 + b_2 · LayerNorm(x + Sublayer(x))',
    description: 'Nonlinear MLP projection and layer normalization stabilize representations.',
    cameraPos: [0, 2.0, 10.0],
    cameraTarget: [0, 0.5, 0]
  },
  {
    id: 'decoder_masked',
    index: 5,
    name: '6. Decoder Masked Self-Attention',
    shortName: 'Decoder Masked',
    formula: 'M_{ij} = -∞ for j > i · Softmax zeros out future attention',
    description: 'Causal mask energy shield ensures decoder cannot see future tokens.',
    cameraPos: [0, 2.2, 9.5],
    cameraTarget: [0, 0.5, 0]
  },
  {
    id: 'cross_attn',
    index: 6,
    name: '7. Multi-Head Cross-Attention',
    shortName: 'Cross-Attention',
    formula: 'CrossAttention(Q_dec, K_enc, V_enc) = softmax(Q_dec K_enc^T / √d_k) V_enc',
    description: 'Decoder queries cross 3D space to attend to rich Encoder representations.',
    cameraPos: [0, 3.5, 12.5],
    cameraTarget: [0, 0.5, 0]
  },
  {
    id: 'softmax',
    index: 7,
    name: '8. Linear Logits & Softmax Sampling',
    shortName: 'Linear & Softmax',
    formula: 'P(w | context) = softmax(W_v z + b_v)',
    description: 'Dense projection to vocabulary probabilities; greedy winner is sampled.',
    cameraPos: [0, 2.2, 9.5],
    cameraTarget: [0, 0.6, 0]
  },
  {
    id: 'autoregressive',
    index: 8,
    name: '9. Autoregressive Output Formation',
    shortName: 'Generated Loop',
    formula: 'y_t = Model(y_{<t}, x) · Loop until </s>',
    description: 'Tokens assemble sequentially into the completed output sentence.',
    cameraPos: [0, 2.0, 9.5],
    cameraTarget: [0, 0.5, 0]
  }
];

function CameraController({
  stageInfo,
  resetTrigger
}: {
  stageInfo: StageInfo;
  resetTrigger: number;
}) {
  const controlsRef = useRef<any>(null);

  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.target.set(...stageInfo.cameraTarget);
      controlsRef.current.object.position.set(...stageInfo.cameraPos);
      controlsRef.current.update();
    }
  }, [stageInfo, resetTrigger]);

  useFrame((state) => {
    // Subtle gentle camera breathing
    state.camera.lookAt(stageInfo.cameraTarget[0], stageInfo.cameraTarget[1], stageInfo.cameraTarget[2]);
  });

  return (
    <OrbitControls
      ref={controlsRef}
      enableDamping={true}
      dampingFactor={0.05}
      minDistance={3}
      maxDistance={25}
    />
  );
}

interface ThreeCanvasProps {
  trace: TransformerTrace;
  currentStageIndex: number;
  activeHead: number;
  resetTrigger: number;
  activeDecoderStep?: number;
}

export const ThreeCanvas: React.FC<ThreeCanvasProps> = ({
  trace,
  currentStageIndex,
  activeHead,
  resetTrigger,
  activeDecoderStep = 0
}) => {
  const stage = STAGES[currentStageIndex] || STAGES[0];
  const decStepIdx = Math.min(activeDecoderStep, (trace.decoder_steps?.length || 1) - 1);
  const currentDecStep = trace.decoder_steps?.[decStepIdx] || trace.decoder_steps?.[0];

  return (
    <div className="w-full h-full relative bg-slate-950 overflow-hidden">
      <Canvas
        camera={{ position: stage.cameraPos, fov: 45 }}
        gl={{ antialias: true, alpha: false }}
        className="w-full h-full"
      >
        <color attach="background" args={['#030712']} />
        
        {/* Deep Space Background Stars */}
        <Stars radius={60} depth={40} count={2500} factor={3} saturation={0.5} fade={true} />

        {/* Cinematic Research Lighting */}
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 15, 10]} intensity={1.2} castShadow />
        <directionalLight position={[-10, 10, -5]} intensity={0.4} color="#38bdf8" />
        <pointLight position={[0, 4, 2]} intensity={1.5} color="#00f0ff" distance={15} />

        {/* Camera Controller & OrbitControls */}
        <CameraController stageInfo={stage} resetTrigger={resetTrigger} />

        {/* Active Stage Rendering */}
        {stage.id === 'input' && (
          <StageInputTokens
            tokens={trace.encoder.source_tokens}
            tokenIds={trace.encoder.source_ids}
          />
        )}

        {stage.id === 'embedding' && (
          <StageEmbedding
            tokens={trace.encoder.source_tokens}
            embeddings={trace.encoder.embeddings}
          />
        )}

        {stage.id === 'positional' && (
          <StagePositionalEncoding
            tokens={trace.encoder.source_tokens}
            embeddings={trace.encoder.embeddings}
            posEncodings={trace.encoder.positional_encodings}
            embeddedPositions={trace.encoder.embedded_positions}
          />
        )}

        {stage.id === 'encoder_attn' && (
          <StageEncoderAttention
            tokens={trace.encoder.source_tokens}
            layerTrace={trace.encoder.layer_1}
            activeHead={activeHead}
          />
        )}

        {stage.id === 'encoder_ffn' && (
          <StageEncoderFFN
            tokens={trace.encoder.source_tokens}
            layerTrace={trace.encoder.layer_1}
          />
        )}

        {stage.id === 'decoder_masked' && currentDecStep && (
          <StageDecoderMasked
            decoderStep={currentDecStep}
            activeHead={activeHead}
          />
        )}

        {stage.id === 'cross_attn' && currentDecStep && (
          <StageCrossAttention
            encoderTokens={trace.encoder.source_tokens}
            decoderStep={currentDecStep}
            activeHead={activeHead}
          />
        )}

        {stage.id === 'softmax' && currentDecStep && (
          <StageSoftmaxGeneration decoderStep={currentDecStep} />
        )}

        {stage.id === 'autoregressive' && (
          <StageAutoregressiveSummary trace={trace} />
        )}
      </Canvas>
    </div>
  );
};
