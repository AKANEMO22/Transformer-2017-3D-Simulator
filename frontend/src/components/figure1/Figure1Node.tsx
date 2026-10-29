import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface Figure1NodeProps {
  id: string;
  label: string;
  subLabel?: string;
  type: string;
  position: [number, number, number];
  size: [number, number, number];
  color: string;
  isHighlighted?: boolean;
  isExecuting?: boolean;
  onClick?: () => void;
}

export const Figure1Node: React.FC<Figure1NodeProps> = ({
  label,
  subLabel,
  type,
  position,
  size,
  color,
  isHighlighted = false,
  isExecuting = false,
  onClick
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const borderRef = useRef<THREE.LineSegments>(null);

  // Subtle breathing animation during execution
  useFrame((state) => {
    if (isExecuting && meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      const t = state.clock.getElapsedTime();
      mat.emissiveIntensity = 0.2 + Math.sin(t * 3) * 0.15;
    }
  });

  const isCircular = type === 'add';
  const width = size[0];
  const height = size[1];
  const depth = size[2];

  return (
    <group position={position} onClick={onClick}>
      {/* 3D Geometry Body */}
      {isCircular ? (
        <mesh ref={meshRef} castShadow receiveShadow rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[width / 2, width / 2, depth, 24]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={isHighlighted ? 0.6 : 0.15}
            roughness={0.25}
            metalness={0.8}
          />
        </mesh>
      ) : (
        <mesh ref={meshRef} castShadow receiveShadow>
          <boxGeometry args={[width, height, depth]} />
          <meshStandardMaterial
            color={isHighlighted ? '#38bdf8' : color}
            emissive={color}
            emissiveIntensity={isHighlighted ? 0.5 : 0.15}
            roughness={0.3}
            metalness={0.7}
          />
        </mesh>
      )}

      {/* Precision Metallic Chamfer Edge */}
      {!isCircular && (
        <lineSegments ref={borderRef}>
          <edgesGeometry args={[new THREE.BoxGeometry(width + 0.02, height + 0.02, depth + 0.02)]} />
          <lineBasicMaterial
            color={isHighlighted ? '#ffffff' : '#94a3b8'}
            transparent={true}
            opacity={0.7}
            linewidth={1.5}
          />
        </lineSegments>
      )}

      {/* Main Block Label */}
      <Text
        position={[0, subLabel ? 0.1 : 0, depth / 2 + 0.05]}
        fontSize={isCircular ? 0.24 : 0.22}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        font="https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjp-Ek-_EeA.woff"
      >
        {label}
      </Text>

      {/* Sub-label (e.g. "Self-Attention", "Cross-Attention", "shifted right") */}
      {subLabel && (
        <Text
          position={[0, -0.16, depth / 2 + 0.05]}
          fontSize={0.12}
          color="#cbd5e1"
          anchorX="center"
          anchorY="middle"
        >
          {subLabel}
        </Text>
      )}

      {/* Input / Output Signal Port Indicators */}
      {type.includes('attention') && (
        <group position={[0, -height / 2 - 0.05, 0]}>
          {/* Q, K, V Bottom Ports */}
          {[-0.8, 0, 0.8].map((xOffset, pIdx) => (
            <mesh key={`port-${pIdx}`} position={[xOffset, 0, 0]}>
              <cylinderGeometry args={[0.06, 0.06, 0.1, 8]} />
              <meshBasicMaterial color="#38bdf8" />
            </mesh>
          ))}
        </group>
      )}
    </group>
  );
};
