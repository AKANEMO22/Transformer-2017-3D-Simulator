import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { Text } from '@react-three/drei';

interface StageInputTokensProps {
  tokens: string[];
  tokenIds: number[];
}

export const StageInputTokens: React.FC<StageInputTokensProps> = ({ tokens, tokenIds }) => {
  const spacing = 1.8;
  const startX = -((tokens.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      {/* Stage Title in 3D Space */}
      <Text
        position={[0, 2.2, 0]}
        fontSize={0.32}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        INPUT SEQUENCE & TOKENIZATION
      </Text>
      <Text
        position={[0, 1.7, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        Sentence is segmented into discrete vocabulary indices
      </Text>

      {/* Floating 3D Token Blocks */}
      {tokens.map((token, idx) => {
        const xPos = startX + idx * spacing;
        return (
          <GlowingToken
            key={idx}
            text={token}
            subText={`pos: ${idx} | id: ${tokenIds[idx] ?? idx}`}
            position={[xPos, 0, 0]}
            color="#00f0ff"
            glowColor="#0284c7"
            scale={1.1}
            highlighted={true}
          />
        );
      })}

      {/* Ground Grid Platform */}
      <gridHelper args={[10, 10, '#0284c7', '#0f172a']} position={[0, -1.2, 0]} />
    </group>
  );
};
