import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface CausalMaskWallProps {
  position: [number, number, number];
  size?: [number, number];
}

export const CausalMaskWall: React.FC<CausalMaskWallProps> = ({
  position,
  size = [3.5, 2.2]
}) => {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshBasicMaterial;
      const t = state.clock.getElapsedTime();
      mat.opacity = 0.25 + Math.sin(t * 4) * 0.08;
    }
  });

  return (
    <group position={position}>
      {/* Translucent Energy Shield */}
      <mesh ref={meshRef}>
        <planeGeometry args={[size[0], size[1]]} />
        <meshBasicMaterial
          color="#ef4444"
          transparent={true}
          opacity={0.3}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Grid Wireframe Outline */}
      <lineSegments>
        <edgesGeometry args={[new THREE.PlaneGeometry(size[0], size[1], 4, 3)]} />
        <lineBasicMaterial color="#f87171" transparent={true} opacity={0.6} />
      </lineSegments>

      {/* Hologram Warning Label */}
      <Text
        position={[0, 0, 0.05]}
        fontSize={0.22}
        color="#fca5a5"
        anchorX="center"
        anchorY="middle"
      >
        CAUSAL MASK [-inf]
      </Text>
      <Text
        position={[0, -0.28, 0.05]}
        fontSize={0.12}
        color="#fee2e2"
        anchorX="center"
        anchorY="middle"
      >
        Future Tokens Blocked
      </Text>
    </group>
  );
};
