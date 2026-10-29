import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { AttentionBeam } from '../3d/AttentionBeam';
import { Text } from '@react-three/drei';
import type { DecoderStepTrace } from '../../types/trace';

interface StageCrossAttentionProps {
  encoderTokens: string[];
  decoderStep: DecoderStepTrace;
  activeHead: number;
}

export const StageCrossAttention: React.FC<StageCrossAttentionProps> = ({
  encoderTokens,
  decoderStep,
  activeHead
}) => {
  const encCount = encoderTokens.length;
  const decTokens = decoderStep.current_tokens;
  const decCount = decTokens.length;

  // Encoder placed on LEFT (x = -4.5), Decoder placed on RIGHT (x = 4.5)
  const encSpacing = 1.8;
  const decSpacing = 1.8;
  const encStartZ = -((encCount - 1) * encSpacing) / 2;
  const decStartZ = -((decCount - 1) * decSpacing) / 2;

  const weights = decoderStep.cross_attention?.attention_weights; // [num_heads, cur_len, enc_len]
  const currentDecIdx = decCount - 1; // latest token generating next attention

  return (
    <group position={[0, 0, 0]}>
      {/* Title */}
      <Text
        position={[0, 3.4, 0]}
        fontSize={0.34}
        color="#fbbf24"
        anchorX="center"
        anchorY="middle"
      >
        MULTI-HEAD CROSS-ATTENTION (ENCODER → DECODER BRIDGE)
      </Text>
      <Text
        position={[0, 2.9, 0]}
        fontSize={0.16}
        color="#fef08a"
        anchorX="center"
        anchorY="middle"
      >
        Queries (Q) from Decoder attend to Keys (K) and Values (V) from Encoder representations
      </Text>

      {/* ENCODER STACK (LEFT - CYAN) */}
      <group position={[-4.5, 0, 0]}>
        <Text position={[0, 2.0, 0]} fontSize={0.22} color="#38bdf8" anchorX="center">
          ENCODER CONTEXT (K, V)
        </Text>
        {encoderTokens.map((token, j) => {
          const zPos = encStartZ + j * encSpacing;
          return (
            <GlowingToken
              key={`enc-${j}`}
              text={token}
              subText={`enc pos ${j}`}
              position={[0, 0, zPos]}
              color="#00f0ff"
              glowColor="#0284c7"
              scale={1.0}
            />
          );
        })}
      </group>

      {/* DECODER STACK (RIGHT - VIOLET) */}
      <group position={[4.5, 0, 0]}>
        <Text position={[0, 2.0, 0]} fontSize={0.22} color="#c084fc" anchorX="center">
          DECODER QUERY (Q)
        </Text>
        {decTokens.map((token, i) => {
          const zPos = decStartZ + i * decSpacing;
          const isLatest = i === currentDecIdx;

          return (
            <GlowingToken
              key={`dec-${i}`}
              text={token}
              subText={isLatest ? 'CURRENT QUERY' : `target ${i}`}
              position={[0, 0, zPos]}
              color={isLatest ? '#fbbf24' : '#a855f7'}
              glowColor={isLatest ? '#d97706' : '#7c3aed'}
              highlighted={isLatest}
              scale={isLatest ? 1.15 : 0.95}
            />
          );
        })}
      </group>

      {/* GLOWING AMBER CROSS-ATTENTION BEAMS BRIDGING DECODER TO ENCODER */}
      {weights && weights[activeHead === -1 ? 0 : activeHead]?.map((row, i) => {
        // We highlight beams coming from the active/latest decoder token
        const decZ = decStartZ + i * decSpacing;
        const startPos: [number, number, number] = [4.5, 0.4, decZ];

        return row.map((weight, j) => {
          const encZ = encStartZ + j * encSpacing;
          const endPos: [number, number, number] = [-4.5, 0.4, encZ];

          return (
            <AttentionBeam
              key={`cross-beam-${i}-${j}`}
              start={startPos}
              end={endPos}
              weight={weight}
              color="#fbbf24"
              archHeight={1.8 + weight * 1.5}
              showPulse={i === currentDecIdx}
            />
          );
        });
      })}

      {/* Floating Center Cross-Attention Matrix */}
      <group position={[0, 0.2, 0]}>
        <Text position={[0, 1.4, 0]} fontSize={0.14} color="#fef08a" anchorX="center">
          CROSS-ATTENTION WEIGHTS (Q_dec × K_enc^T)
        </Text>
        {weights && weights[activeHead === -1 ? 0 : activeHead]?.map((row, r) => (
          <group key={`crow-${r}`} position={[0, 0.9 - r * 0.4, 0]}>
            {row.map((val, c) => (
              <group key={`ccell-${r}-${c}`} position={[(c - row.length / 2) * 0.55, 0, 0]}>
                <mesh>
                  <planeGeometry args={[0.5, 0.35]} />
                  <meshBasicMaterial
                    color="#b45309"
                    transparent={true}
                    opacity={Math.max(0.15, val * 1.6)}
                  />
                </mesh>
                <Text position={[0, 0, 0.02]} fontSize={0.10} color="#ffffff" anchorX="center" anchorY="middle">
                  {val.toFixed(2)}
                </Text>
              </group>
            ))}
          </group>
        ))}
      </group>

      <gridHelper args={[22, 22, '#f59e0b', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
