import React, { useState } from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { AttentionBeam } from '../3d/AttentionBeam';
import { Text } from '@react-three/drei';
import type { LayerAttentionTrace } from '../../types/trace';

interface StageEncoderAttentionProps {
  tokens: string[];
  layerTrace: LayerAttentionTrace;
  activeHead: number; // 0, 1, or -1 for all
}

export const StageEncoderAttention: React.FC<StageEncoderAttentionProps> = ({
  tokens,
  layerTrace,
  activeHead
}) => {
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number | null>(null);

  const spacing = 2.4;
  const startX = -((tokens.length - 1) * spacing) / 2;
  const weights = layerTrace.attention_weights; // [num_heads, seq_len, seq_len]

  // Head colors
  const headColors = ['#00f0ff', '#f59e0b'];

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.2, 0]}
        fontSize={0.32}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        ENCODER MULTI-HEAD SELF-ATTENTION
      </Text>
      <Text
        position={[0, 2.7, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        Attention(Q, K, V) = softmax(Q K^T / √d_k) V · Each token attends to all tokens in parallel
      </Text>

      {/* Floating 3D Tokens */}
      {tokens.map((token, i) => {
        const xPos = startX + i * spacing;
        const isHovered = selectedTokenIdx === i;

        return (
          <group
            key={i}
            position={[xPos, -0.6, 0]}
            onClick={() => setSelectedTokenIdx(selectedTokenIdx === i ? null : i)}
          >
            <GlowingToken
              text={token}
              subText={`token ${i}`}
              position={[0, 0, 0]}
              color={isHovered ? '#10b981' : '#0284c7'}
              glowColor={isHovered ? '#059669' : '#0369a1'}
              highlighted={isHovered}
              scale={1.05}
            />

            {/* Q, K, V labels hovering above token */}
            <Text
              position={[0, 0.65, 0]}
              fontSize={0.11}
              color="#38bdf8"
              anchorX="center"
              anchorY="middle"
            >
              Q, K, V
            </Text>
          </group>
        );
      })}

      {/* 3D Attention Beams connecting token pairs */}
      {weights && weights.map((headWeights, hIdx) => {
        if (activeHead !== -1 && activeHead !== hIdx) return null;
        const color = headColors[hIdx % headColors.length];

        return (
          <group key={`head-${hIdx}`}>
            {headWeights.map((row, i) => {
              if (selectedTokenIdx !== null && selectedTokenIdx !== i) return null;
              const startXPos = startX + i * spacing;

              return row.map((weight, j) => {
                const endXPos = startX + j * spacing;
                // Offset curve slightly in Z and Y depending on head
                const zOffset = (hIdx - 0.5) * 0.4;
                const archHeight = 0.8 + Math.abs(i - j) * 0.45;

                return (
                  <AttentionBeam
                    key={`beam-${hIdx}-${i}-${j}`}
                    start={[startXPos, -0.2, zOffset]}
                    end={[endXPos, -0.2, zOffset]}
                    weight={weight}
                    color={color}
                    archHeight={archHeight}
                  />
                );
              });
            })}
          </group>
        );
      })}

      {/* 3D Mini Attention Heatmap floating on the right side */}
      <group position={[startX + tokens.length * spacing + 0.8, 0.4, 0]}>
        <Text position={[0, 1.4, 0]} fontSize={0.14} color="#e2e8f0" anchorX="center">
          {activeHead === -1 ? 'HEAD 0 ATTENTION MATRIX' : `HEAD ${activeHead} ATTENTION MATRIX`}
        </Text>
        {weights && weights[activeHead === -1 ? 0 : activeHead]?.map((row, r) => (
          <group key={`row-${r}`} position={[0, 1.0 - r * 0.35, 0]}>
            {row.map((val, c) => (
              <group key={`cell-${r}-${c}`} position={[(c - row.length / 2) * 0.45, 0, 0]}>
                <mesh>
                  <planeGeometry args={[0.4, 0.3]} />
                  <meshBasicMaterial
                    color={headColors[(activeHead === -1 ? 0 : activeHead) % headColors.length]}
                    transparent={true}
                    opacity={Math.max(0.1, val * 1.5)}
                  />
                </mesh>
                <Text position={[0, 0, 0.02]} fontSize={0.09} color="#ffffff" anchorX="center" anchorY="middle">
                  {val.toFixed(2)}
                </Text>
              </group>
            ))}
          </group>
        ))}
      </group>

      <gridHelper args={[18, 18, '#0284c7', '#0f172a']} position={[0, -1.6, 0]} />
    </group>
  );
};
