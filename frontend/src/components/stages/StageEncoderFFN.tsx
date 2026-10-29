import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { VectorPillars } from '../3d/VectorPillars';
import { Text } from '@react-three/drei';
import type { LayerAttentionTrace } from '../../types/trace';

interface StageEncoderFFNProps {
  tokens: string[];
  layerTrace: LayerAttentionTrace;
}

export const StageEncoderFFN: React.FC<StageEncoderFFNProps> = ({ tokens, layerTrace }) => {
  const spacing = 2.4;
  const startX = -((tokens.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.2, 0]}
        fontSize={0.32}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        ENCODER FEED-FORWARD NETWORK & RESIDUALS
      </Text>
      <Text
        position={[0, 2.7, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        FFN(x) = max(0, x W1 + b1) W2 + b2 · LayerNorm(x + Sublayer(x))
      </Text>

      {tokens.map((token, i) => {
        const xPos = startX + i * spacing;
        const ffnVec = layerTrace.ffn_output?.[i] || [0, 0, 0, 0, 0, 0, 0, 0];
        const outVec = layerTrace.layer_output?.[i] || [0, 0, 0, 0, 0, 0, 0, 0];

        return (
          <group key={i} position={[xPos, 0, 0]}>
            <GlowingToken
              text={token}
              subText={`token ${i}`}
              position={[0, -1.2, 0]}
              color="#0284c7"
              scale={0.85}
            />

            {/* FFN intermediate projection */}
            <VectorPillars
              vector={ffnVec}
              position={[-0.45, -0.1, 0]}
              color="#a855f7"
              label="FFN(x)"
              scale={0.55}
            />

            {/* Add & Norm Symbol */}
            <Text position={[0, 0.2, 0]} fontSize={0.22} color="#38bdf8" anchorX="center">
              + LN
            </Text>

            {/* Final contextualized encoder representation */}
            <VectorPillars
              vector={outVec}
              position={[0, 1.4, 0]}
              color="#00f0ff"
              label={`H_enc[${token}]`}
              scale={0.75}
            />
          </group>
        );
      })}

      <gridHelper args={[18, 18, '#0284c7', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
