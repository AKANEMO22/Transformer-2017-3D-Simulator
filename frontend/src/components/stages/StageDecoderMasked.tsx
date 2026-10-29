import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { AttentionBeam } from '../3d/AttentionBeam';
import { CausalMaskWall } from '../3d/CausalMaskWall';
import { Text } from '@react-three/drei';
import type { DecoderStepTrace } from '../../types/trace';

interface StageDecoderMaskedProps {
  decoderStep: DecoderStepTrace;
  activeHead: number;
}

export const StageDecoderMasked: React.FC<StageDecoderMaskedProps> = ({
  decoderStep,
  activeHead
}) => {
  const tokens = decoderStep.current_tokens; // e.g. ["<s>", "thấu", "hiểu"]
  const spacing = 2.4;
  const startX = -((tokens.length - 1) * spacing) / 2;
  const weights = decoderStep.masked_self_attention?.attention_weights; // [H, cur_len, cur_len]
  const headColor = activeHead === 1 ? '#ec4899' : '#a855f7'; // Violet/Pink

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.3, 0]}
        fontSize={0.32}
        color="#c084fc"
        anchorX="center"
        anchorY="middle"
      >
        DECODER MASKED SELF-ATTENTION (CAUSAL MASK)
      </Text>
      <Text
        position={[0, 2.8, 0]}
        fontSize={0.16}
        color="#e9d5ff"
        anchorX="center"
        anchorY="middle"
      >
        M_ij = -∞ for j &gt; i · Upper triangle is zeroed out by softmax to prevent lookahead
      </Text>

      {/* Decoder Tokens */}
      {tokens.map((token, i) => {
        const xPos = startX + i * spacing;
        const isCurrent = i === tokens.length - 1;

        return (
          <group key={i} position={[xPos, -0.6, 0]}>
            <GlowingToken
              text={token}
              subText={i === 0 ? 'START <S>' : `step ${i}`}
              position={[0, 0, 0]}
              color={isCurrent ? '#f43f5e' : '#a855f7'}
              glowColor={isCurrent ? '#e11d48' : '#7c3aed'}
              highlighted={isCurrent}
              scale={1.05}
            />
          </group>
        );
      })}

      {/* Valid Causal Attention Beams (Only j <= i) */}
      {weights && weights[activeHead === -1 ? 0 : activeHead]?.map((row, i) => {
        const startXPos = startX + i * spacing;

        return row.map((weight, j) => {
          if (j > i) return null; // strictly past only
          const endXPos = startX + j * spacing;
          const archHeight = 0.7 + Math.abs(i - j) * 0.4;

          return (
            <AttentionBeam
              key={`dec-beam-${i}-${j}`}
              start={[startXPos, -0.2, 0]}
              end={[endXPos, -0.2, 0]}
              weight={weight}
              color={headColor}
              archHeight={archHeight}
            />
          );
        });
      })}

      {/* 3D Holographic Causal Shield blocking future positions */}
      <CausalMaskWall
        position={[startX + tokens.length * spacing + 0.3, 0.4, 0.2]}
        size={[3.2, 2.2]}
      />

      {/* 3D Causal Masked Attention Matrix */}
      <group position={[startX - 3.2, 0.4, 0]}>
        <Text position={[0, 1.4, 0]} fontSize={0.14} color="#e9d5ff" anchorX="center">
          LOWER-TRIANGULAR MATRIX
        </Text>
        {weights && weights[activeHead === -1 ? 0 : activeHead]?.map((row, r) => (
          <group key={`drow-${r}`} position={[0, 1.0 - r * 0.38, 0]}>
            {row.map((val, c) => {
              const isMasked = c > r;
              return (
                <group key={`dcell-${r}-${c}`} position={[(c - row.length / 2) * 0.55, 0, 0]}>
                  <mesh>
                    <planeGeometry args={[0.5, 0.32]} />
                    <meshBasicMaterial
                      color={isMasked ? '#450a0a' : '#7c3aed'}
                      transparent={true}
                      opacity={isMasked ? 0.4 : Math.max(0.2, val * 1.5)}
                    />
                  </mesh>
                  <Text position={[0, 0, 0.02]} fontSize={0.09} color={isMasked ? '#ef4444' : '#ffffff'} anchorX="center" anchorY="middle">
                    {isMasked ? '0.00' : val.toFixed(2)}
                  </Text>
                </group>
              );
            })}
          </group>
        ))}
      </group>

      <gridHelper args={[18, 18, '#7c3aed', '#0f172a']} position={[0, -1.6, 0]} />
    </group>
  );
};
