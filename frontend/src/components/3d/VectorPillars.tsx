import React from 'react';
import { Text } from '@react-three/drei';

interface VectorPillarsProps {
  vector: number[];
  position: [number, number, number];
  color?: string;
  label?: string;
  scale?: number;
}

export const VectorPillars: React.FC<VectorPillarsProps> = ({
  vector,
  position,
  color = '#38bdf8',
  label,
  scale = 1
}) => {
  const dModel = vector.length;
  const pillarWidth = 0.12;
  const spacing = 0.18;
  const totalWidth = dModel * spacing;

  return (
    <group position={position} scale={[scale, scale, scale]}>
      {/* Label for the vector */}
      {label && (
        <Text
          position={[0, 1.4, 0]}
          fontSize={0.16}
          color="#e2e8f0"
          anchorX="center"
          anchorY="middle"
        >
          {label}
        </Text>
      )}

      {/* Ground reference plate */}
      <mesh position={[0, -0.05, 0]} receiveShadow>
        <boxGeometry args={[totalWidth + 0.2, 0.04, 0.35]} />
        <meshStandardMaterial color="#1e293b" roughness={0.5} metalness={0.2} />
      </mesh>

      {/* Individual dimension bars */}
      {vector.map((val, i) => {
        const height = Math.max(0.1, Math.min(1.8, Math.abs(val) * 0.8 + 0.1));
        const yPos = val >= 0 ? height / 2 : -height / 2;
        const xPos = (i - (dModel - 1) / 2) * spacing;
        const barColor = val >= 0 ? color : '#f43f5e'; // cyan/blue if positive, rose if negative

        return (
          <group key={i} position={[xPos, 0, 0]}>
            <mesh position={[0, yPos, 0]} castShadow>
              <boxGeometry args={[pillarWidth, height, pillarWidth]} />
              <meshStandardMaterial
                color={barColor}
                emissive={barColor}
                emissiveIntensity={0.35}
                roughness={0.3}
              />
            </mesh>

            {/* Dimension index tiny label */}
            <Text
              position={[0, -0.15, 0.12]}
              fontSize={0.08}
              color="#94a3b8"
              anchorX="center"
              anchorY="middle"
            >
              {`d${i}`}
            </Text>
          </group>
        );
      })}
    </group>
  );
};
