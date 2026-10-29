import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { VectorPillars } from '../3d/VectorPillars';
import { Text } from '@react-three/drei';

interface StagePositionalEncodingProps {
  tokens: string[];
  embeddings: number[][];
  posEncodings: number[][];
  embeddedPositions: number[][];
}

export const StagePositionalEncoding: React.FC<StagePositionalEncodingProps> = ({
  tokens,
  embeddings,
  posEncodings,
  embeddedPositions
}) => {
  const spacing = 2.4;
  const startX = -((tokens.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.0, 0]}
        fontSize={0.32}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        SINUSOIDAL POSITIONAL ENCODING
      </Text>
      <Text
        position={[0, 2.5, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        x_pos = Token_Embedding + PE(pos) [sin/cos harmonics inject order into permutation-invariant attention]
      </Text>

      {tokens.map((token, idx) => {
        const xPos = startX + idx * spacing;
        const peVec = posEncodings[idx] || [0, 0, 0, 0, 0, 0, 0, 0];
        const finalVec = embeddedPositions[idx] || embeddings[idx] || [0, 0, 0, 0, 0, 0, 0, 0];

        return (
          <group key={idx} position={[xPos, 0, 0]}>
            {/* Token anchor */}
            <GlowingToken
              text={token}
              subText={`pos: ${idx}`}
              position={[0, -1.2, 0]}
              color="#0284c7"
              scale={0.85}
            />

            {/* Positional Encoding vector (amber/yellow) */}
            <VectorPillars
              vector={peVec}
              position={[-0.45, -0.1, 0]}
              color="#f59e0b"
              label="PE(pos)"
              scale={0.55}
            />

            <Text
              position={[0, 0.2, 0]}
              fontSize={0.24}
              color="#e2e8f0"
              anchorX="center"
              anchorY="middle"
            >
              +
            </Text>

            {/* Combined Result Vector (glowing emerald) */}
            <VectorPillars
              vector={finalVec}
              position={[0, 1.4, 0]}
              color="#10b981"
              label="Input Vector x0"
              scale={0.75}
            />
          </group>
        );
      })}

      <gridHelper args={[16, 16, '#0284c7', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
