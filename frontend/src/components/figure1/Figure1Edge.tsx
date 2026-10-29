import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

interface Figure1EdgeProps {
  id?: string;
  fromId: string;
  toId: string;
  fromPos: [number, number, number];
  toPos: [number, number, number];
  type: 'feedforward' | 'residual' | 'memory';
  isActive?: boolean;
  isHighlighted?: boolean;
  isDimmed?: boolean;
  progress?: number;
  onClick?: () => void;
}

export const Figure1Edge: React.FC<Figure1EdgeProps> = ({
  fromPos,
  toPos,
  type,
  isActive = false,
  isHighlighted = false,
  isDimmed = false,
  progress = 0,
  onClick
}) => {
  const pulseRef = useRef<THREE.Mesh>(null);

  // Compute precise routing path points based on edge type
  let points: [number, number, number][] = [];

  if (type === 'feedforward') {
    // Vertical straight direct connection
    points = [
      [fromPos[0], fromPos[1] + 0.35, fromPos[2]],
      [toPos[0], toPos[1] - 0.35, toPos[2]]
    ];
  } else if (type === 'residual') {
    // Iconic bypass path: branches out sideways, goes up, enters Add & Norm
    const isEncoder = fromPos[0] < 0;
    const sideX = isEncoder ? fromPos[0] - 2.5 : fromPos[0] + 2.5;

    points = [
      [fromPos[0], fromPos[1] + 0.3, fromPos[2]],
      [sideX, fromPos[1] + 0.3, fromPos[2]],
      [sideX, toPos[1], toPos[2]],
      [isEncoder ? toPos[0] - 1.85 : toPos[0] + 1.85, toPos[1], toPos[2]]
    ];
  } else if (type === 'memory') {
    // Cross-Attention Memory Bus: leaves Encoder top, crosses over to Decoder Cross-Attn
    const midX = (fromPos[0] + toPos[0]) / 2;
    points = [
      [fromPos[0], fromPos[1] + 0.35, fromPos[2]],
      [fromPos[0], fromPos[1] + 1.2, fromPos[2]],
      [midX, fromPos[1] + 1.2, fromPos[2]],
      [midX, toPos[1], toPos[2]],
      [toPos[0] - 1.85, toPos[1], toPos[2]]
    ];
  }

  // Color according to edge state
  let edgeColor = '#64748b';
  if (isActive) {
    edgeColor = type === 'memory' ? '#f59e0b' : '#38bdf8';
  } else if (isHighlighted) {
    edgeColor = '#00f0ff';
  } else if (isDimmed) {
    edgeColor = '#1e293b';
  } else {
    edgeColor = type === 'residual' ? '#0ea5e9' : type === 'memory' ? '#d97706' : '#64748b';
  }

  const lineWidth = isActive ? 3 : isHighlighted ? 2.5 : type === 'memory' ? 2.2 : type === 'residual' ? 1.8 : 1.2;

  // Animate photon along path when active
  useFrame((state) => {
    if (isActive && pulseRef.current && points.length >= 2) {
      // Use timeline progress if provided (> 0), otherwise smooth looping pulse
      const t = progress > 0 ? progress : (state.clock.getElapsedTime() * 2) % 1;
      const totalSegments = points.length - 1;
      const segmentIndex = Math.min(totalSegments - 1, Math.floor(t * totalSegments));
      const segT = (t * totalSegments) - segmentIndex;

      const p0 = points[segmentIndex];
      const p1 = points[segmentIndex + 1];

      const x = THREE.MathUtils.lerp(p0[0], p1[0], segT);
      const y = THREE.MathUtils.lerp(p0[1], p1[1], segT);
      const z = THREE.MathUtils.lerp(p0[2], p1[2], segT);

      pulseRef.current.position.set(x, y, z);
    }
  });

  return (
    <group onClick={(e) => { e.stopPropagation(); onClick?.(); }}>
      <Line
        points={points}
        color={edgeColor}
        lineWidth={lineWidth}
        transparent={true}
        opacity={isDimmed ? 0.2 : isActive ? 1.0 : isHighlighted ? 0.9 : 0.65}
      />

      {/* Terminal Arrow Head at toPos */}
      <mesh
        position={points[points.length - 1]}
        rotation={
          type === 'residual'
            ? [0, 0, fromPos[0] < 0 ? -Math.PI / 2 : Math.PI / 2]
            : type === 'memory'
            ? [0, 0, -Math.PI / 2]
            : [0, 0, 0]
        }
      >
        <coneGeometry args={[0.09, 0.22, 8]} />
        <meshBasicMaterial
          color={edgeColor}
          transparent={isDimmed}
          opacity={isDimmed ? 0.2 : 1.0}
        />
      </mesh>

      {/* Traveling Photon Indicator for active execution */}
      {isActive && (
        <mesh ref={pulseRef}>
          <sphereGeometry args={[0.09, 12, 12]} />
          <meshBasicMaterial color={type === 'memory' ? '#fde047' : '#ffffff'} />
        </mesh>
      )}
    </group>
  );
};
