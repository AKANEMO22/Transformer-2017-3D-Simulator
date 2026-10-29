import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

interface Figure1EdgeProps {
  fromPos: [number, number, number];
  toPos: [number, number, number];
  type: 'feedforward' | 'residual' | 'memory';
  isExecuting?: boolean;
  color?: string;
}

export const Figure1Edge: React.FC<Figure1EdgeProps> = ({
  fromPos,
  toPos,
  type,
  isExecuting = false,
  color
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

  // Edge styling
  const edgeColor = color || (
    type === 'residual' ? '#38bdf8' :
    type === 'memory' ? '#f59e0b' : '#94a3b8'
  );
  const lineWidth = type === 'memory' ? 2.5 : type === 'residual' ? 2.0 : 1.5;

  // Animate pulse along the points array during execution
  useFrame((state) => {
    if (isExecuting && pulseRef.current && points.length >= 2) {
      const speed = 1.8;
      const t = (state.clock.getElapsedTime() * speed) % 1;
      
      // Interpolate along multi-segment path
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
    <group>
      <Line
        points={points}
        color={edgeColor}
        lineWidth={lineWidth}
        transparent={true}
        opacity={type === 'residual' ? 0.85 : type === 'memory' ? 0.95 : 0.6}
      />

      {/* Terminal Arrow Head at toPos */}
      <mesh position={points[points.length - 1]} rotation={
        type === 'residual' 
          ? [0, 0, fromPos[0] < 0 ? -Math.PI / 2 : Math.PI / 2]
          : type === 'memory'
          ? [0, 0, -Math.PI / 2]
          : [0, 0, 0]
      }>
        <coneGeometry args={[0.1, 0.25, 8]} />
        <meshBasicMaterial color={edgeColor} />
      </mesh>

      {/* Animated Traveling Photon Packet */}
      {isExecuting && (
        <mesh ref={pulseRef}>
          <sphereGeometry args={[0.08, 12, 12]} />
          <meshBasicMaterial color={type === 'memory' ? '#fbbf24' : '#00f0ff'} />
        </mesh>
      )}
    </group>
  );
};
