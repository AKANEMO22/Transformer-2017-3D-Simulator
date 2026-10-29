import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { QuadraticBezierLine } from '@react-three/drei';
import * as THREE from 'three';

interface AttentionBeamProps {
  start: [number, number, number];
  end: [number, number, number];
  weight: number; // 0.0 to 1.0
  color?: string;
  archHeight?: number;
  showPulse?: boolean;
}

export const AttentionBeam: React.FC<AttentionBeamProps> = ({
  start,
  end,
  weight,
  color = '#00f0ff',
  archHeight = 1.2,
  showPulse = true
}) => {
  const pulseRef = useRef<THREE.Mesh>(null);

  // Compute bezier midpoint arching in Y/Z
  const mid: [number, number, number] = [
    (start[0] + end[0]) / 2,
    Math.max(start[1], end[1]) + archHeight * Math.max(0.3, weight),
    (start[2] + end[2]) / 2 + 0.3
  ];

  // Animated traveling pulse along quadratic bezier curve
  useFrame((state) => {
    if (showPulse && pulseRef.current && weight > 0.08) {
      const speed = 1.5;
      const t = (state.clock.getElapsedTime() * speed) % 1;
      // Quadratic Bezier formula: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
      const x = (1 - t) * (1 - t) * start[0] + 2 * (1 - t) * t * mid[0] + t * t * end[0];
      const y = (1 - t) * (1 - t) * start[1] + 2 * (1 - t) * t * mid[1] + t * t * end[1];
      const z = (1 - t) * (1 - t) * start[2] + 2 * (1 - t) * t * mid[2] + t * t * end[2];
      pulseRef.current.position.set(x, y, z);
    }
  });

  if (weight < 0.02) return null;

  const lineWidth = Math.max(1, weight * 5);
  const opacity = Math.min(0.9, Math.max(0.15, weight * 1.2));

  return (
    <group>
      <QuadraticBezierLine
        start={start}
        end={end}
        mid={mid}
        color={color}
        lineWidth={lineWidth}
        transparent={true}
        opacity={opacity}
      />
      {showPulse && weight > 0.08 && (
        <mesh ref={pulseRef}>
          <sphereGeometry args={[0.06 * Math.max(0.8, weight * 1.5), 12, 12]} />
          <meshBasicMaterial color={color} />
        </mesh>
      )}
    </group>
  );
};
