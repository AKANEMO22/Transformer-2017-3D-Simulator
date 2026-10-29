import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { Text } from '@react-three/drei';
import type { DecoderStepTrace } from '../../types/trace';

interface StageSoftmaxGenerationProps {
  decoderStep: DecoderStepTrace;
}

export const StageSoftmaxGeneration: React.FC<StageSoftmaxGenerationProps> = ({ decoderStep }) => {
  const candidates = decoderStep.top_candidates || [];
  const winnerToken = decoderStep.next_token;
  const spacing = 2.0;
  const startX = -((candidates.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.8, 0]}
        fontSize={0.32}
        color="#34d399"
        anchorX="center"
        anchorY="middle"
      >
        LINEAR PROJECTION & SOFTMAX VOCABULARY SAMPLING
      </Text>
      <Text
        position={[0, 3.3, 0]}
        fontSize={0.16}
        color="#a7f3d0"
        anchorX="center"
        anchorY="middle"
      >
        P(w_i | context) = exp(z_i) / Σ exp(z_j) · Winning token is greedily selected or sampled
      </Text>

      {/* Top Candidates 3D Probability Towers */}
      {candidates.map((cand, idx) => {
        const xPos = startX + idx * spacing;
        const isWinner = cand.token === winnerToken || idx === 0;
        const barHeight = Math.max(0.2, cand.prob * 3.8);
        const barColor = isWinner ? '#10b981' : '#0284c7';

        return (
          <group key={idx} position={[xPos, 0, 0]}>
            {/* Probability Tower */}
            <mesh position={[0, barHeight / 2 - 0.8, 0]} castShadow>
              <boxGeometry args={[0.9, barHeight, 0.5]} />
              <meshStandardMaterial
                color={barColor}
                emissive={barColor}
                emissiveIntensity={isWinner ? 0.6 : 0.2}
                roughness={0.2}
              />
            </mesh>

            {/* Probability Badge */}
            <Text
              position={[0, barHeight - 0.5, 0.3]}
              fontSize={0.18}
              color={isWinner ? '#6ee7b7' : '#94a3b8'}
              anchorX="center"
              anchorY="middle"
            >
              {(cand.prob * 100).toFixed(1)}%
            </Text>

            {/* Token Tile */}
            <GlowingToken
              text={cand.token}
              subText={`logit: ${cand.logit.toFixed(2)}`}
              position={[0, -1.2, 0]}
              color={isWinner ? '#10b981' : '#1e293b'}
              glowColor={isWinner ? '#059669' : '#0f172a'}
              highlighted={isWinner}
              scale={isWinner ? 1.15 : 0.9}
            />

            {/* Winner Crown / Halo */}
            {isWinner && (
              <group position={[0, barHeight + 0.3, 0]}>
                <mesh>
                  <torusGeometry args={[0.35, 0.04, 16, 32]} />
                  <meshBasicMaterial color="#34d399" />
                </mesh>
                <Text position={[0, 0.35, 0]} fontSize={0.14} color="#34d399" anchorX="center">
                  SELECTED NEXT TOKEN
                </Text>
              </group>
            )}
          </group>
        );
      })}

      <gridHelper args={[18, 18, '#10b981', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
