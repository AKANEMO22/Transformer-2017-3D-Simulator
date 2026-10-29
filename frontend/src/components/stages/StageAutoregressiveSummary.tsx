import React from 'react';
import { GlowingToken } from '../3d/GlowingToken';
import { AttentionBeam } from '../3d/AttentionBeam';
import { Text } from '@react-three/drei';
import type { TransformerTrace } from '../../types/trace';

interface StageAutoregressiveSummaryProps {
  trace: TransformerTrace;
}

export const StageAutoregressiveSummary: React.FC<StageAutoregressiveSummaryProps> = ({ trace }) => {
  const outputTokens = trace.final_generated_tokens || [];

  const spacing = 1.9;
  const startX = -((outputTokens.length - 1) * spacing) / 2;

  return (
    <group position={[0, 0, 0]}>
      <Text
        position={[0, 3.4, 0]}
        fontSize={0.34}
        color="#38bdf8"
        anchorX="center"
        anchorY="middle"
      >
        AUTOREGRESSIVE GENERATION LOOP (SEQUENCE FORMATION)
      </Text>
      <Text
        position={[0, 2.9, 0]}
        fontSize={0.16}
        color="#94a3b8"
        anchorX="center"
        anchorY="middle"
      >
        Outputs are fed back into the decoder at step t+1 until &lt;/s&gt; is generated
      </Text>

      {/* Final Assembled Sentence */}
      <group position={[0, 1.2, 0]}>
        <Text position={[0, 0.8, 0]} fontSize={0.2} color="#34d399" anchorX="center">
          GENERATED OUTPUT SENTENCE:
        </Text>
        <Text position={[0, 0.35, 0]} fontSize={0.3} color="#ffffff" anchorX="center">
          "{trace.output_sentence}"
        </Text>
      </group>

      {/* Sequential Token Assembly in 3D */}
      {outputTokens.map((tok, idx) => {
        const xPos = startX + idx * spacing;
        const isStart = tok === '<s>';
        const isEnd = tok === '</s>';

        return (
          <group key={idx} position={[xPos, -0.6, 0]}>
            <GlowingToken
              text={tok}
              subText={`step ${idx}`}
              position={[0, 0, 0]}
              color={isStart ? '#a855f7' : isEnd ? '#f43f5e' : '#10b981'}
              glowColor={isStart ? '#7c3aed' : isEnd ? '#e11d48' : '#059669'}
              highlighted={true}
              scale={1.1}
            />

            {/* Step badge */}
            <Text position={[0, -0.6, 0]} fontSize={0.12} color="#94a3b8" anchorX="center">
              {idx === 0 ? 'Prompt Init' : `Generated +${idx}`}
            </Text>
          </group>
        );
      })}

      {/* Autoregressive Feedback Loop Beams */}
      {outputTokens.slice(1).map((_, idx) => {
        const prevX = startX + idx * spacing;
        const nextX = startX + (idx + 1) * spacing;

        return (
          <AttentionBeam
            key={`loop-${idx}`}
            start={[prevX, -0.1, 0]}
            end={[nextX, -0.1, 0]}
            weight={0.8}
            color="#34d399"
            archHeight={0.6}
            showPulse={true}
          />
        );
      })}

      <gridHelper args={[18, 18, '#38bdf8', '#0f172a']} position={[0, -1.8, 0]} />
    </group>
  );
};
