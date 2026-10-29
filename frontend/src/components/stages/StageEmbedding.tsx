import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { VectorPillars } from '../3d/VectorPillars';
import { Text } from '@react-three/drei';

interface StageEmbeddingProps {
  tokens: string[];
  embeddings: number[][]; // [seq_len, d_model]
}

export const StageEmbedding: React.FC<StageEmbeddingProps> = ({ tokens, embeddings }) => {
  const spacing = 2.2;
  const startX = -((tokens.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 2.8, 0]}
        fontSize={0.32}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        EMBEDDING LOOKUP (d_model = 8)
      </Text>
      <Text
        position={[0, 2.3, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        Discrete token IDs are projected into continuous 8-dimensional dense vectors
      </Text>

      {tokens.map((token, idx) => {
        const xPos = startX + idx * spacing;
        const vector = embeddings[idx] || [0, 0, 0, 0, 0, 0, 0, 0];

        return (
          <group key={idx} position={[xPos, 0, 0]}>
            {/* Source Token at bottom */}
            <GlowingToken
              text={token}
              subText={`idx: ${idx}`}
              position={[0, -1.0, 0]}
              color="#0284c7"
              glowColor="#0369a1"
              scale={0.9}
            />

            {/* Glowing vertical connector beam */}
            <mesh position={[0, -0.2, 0]}>
              <cylinderGeometry args={[0.02, 0.02, 1.2, 8]} />
              <meshBasicMaterial color="#38bdf8" transparent={true} opacity={0.6} />
            </mesh>

            {/* 3D Vector Pillars representing embedding dimensions */}
            <VectorPillars
              vector={vector}
              position={[0, 0.6, 0]}
              color="#00f0ff"
              label={`E[${token}]`}
              scale={0.9}
            />
          </group>
        );
      })}

      <gridHelper args={[14, 14, '#0369a1', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
