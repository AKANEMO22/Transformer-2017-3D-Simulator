import React, { useRef, useState } from 'react';
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
  isSelected?: boolean;
  isActive?: boolean;
  isCompleted?: boolean;
  isDimmed?: boolean;
  onClick?: () => void;
}

export const Figure1Node: React.FC<Figure1NodeProps> = ({
  label,
  subLabel,
  type,
  position,
  size,
  color,
  isSelected = false,
  isActive = false,
  isCompleted = false,
  isDimmed = false,
  onClick
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const borderRef = useRef<THREE.LineSegments>(null);
  const [hovered, setHovered] = useState(false);

  // Smooth animation for active execution state
  useFrame((state) => {
    if (meshRef.current) {
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      if (isActive) {
        const t = state.clock.getElapsedTime();
        mat.emissiveIntensity = 0.5 + Math.sin(t * 5) * 0.3;
      } else if (isSelected) {
        mat.emissiveIntensity = 0.5;
      } else if (hovered) {
        mat.emissiveIntensity = 0.35;
      } else if (isCompleted) {
        mat.emissiveIntensity = 0.2;
      } else {
        mat.emissiveIntensity = isDimmed ? 0.05 : 0.15;
      }
    }
  });

  const isCircular = type === 'add';
  const width = size[0];
  const height = size[1];
  const depth = size[2];

  // Visual appearance based on states
  let nodeColor = color;
  if (isSelected) {
    nodeColor = '#38bdf8'; // Cyan
  } else if (isActive) {
    nodeColor = '#fbbf24'; // Amber pulse
  } else if (isCompleted) {
    nodeColor = color;
  } else if (isDimmed) {
    nodeColor = '#1e293b';
  }

  const borderLineColor = isSelected
    ? '#ffffff'
    : isActive
    ? '#f59e0b'
    : hovered
    ? '#38bdf8'
    : isDimmed
    ? '#334155'
    : '#64748b';

  return (
    <group
      position={position}
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      onPointerOver={(e) => {
        e.stopPropagation();
        setHovered(true);
        document.body.style.cursor = 'pointer';
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = 'auto';
      }}
    >
      {/* 3D Geometry Body */}
      {isCircular ? (
        <mesh ref={meshRef} castShadow receiveShadow rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[width / 2, width / 2, depth, 24]} />
          <meshStandardMaterial
            color={nodeColor}
            emissive={isSelected ? '#38bdf8' : isActive ? '#fbbf24' : color}
            emissiveIntensity={isSelected ? 0.6 : isActive ? 0.7 : 0.15}
            roughness={0.25}
            metalness={0.8}
            transparent={isDimmed}
            opacity={isDimmed ? 0.4 : 1.0}
          />
        </mesh>
      ) : (
        <mesh ref={meshRef} castShadow receiveShadow>
          <boxGeometry args={[width, height, depth]} />
          <meshStandardMaterial
            color={nodeColor}
            emissive={isSelected ? '#38bdf8' : isActive ? '#fbbf24' : color}
            emissiveIntensity={isSelected ? 0.5 : isActive ? 0.6 : 0.15}
            roughness={0.3}
            metalness={0.7}
            transparent={isDimmed}
            opacity={isDimmed ? 0.35 : 1.0}
          />
        </mesh>
      )}

      {/* Selected Halo / Active Ring */}
      {(isSelected || isActive) && (
        <mesh position={[0, 0, 0]}>
          <boxGeometry args={[width + 0.15, height + 0.15, depth + 0.1]} />
          <meshBasicMaterial
            color={isSelected ? '#38bdf8' : '#fbbf24'}
            wireframe={true}
            transparent={true}
            opacity={0.8}
          />
        </mesh>
      )}

      {/* Precision Metallic Chamfer Edge */}
      {!isCircular && (
        <lineSegments ref={borderRef}>
          <edgesGeometry args={[new THREE.BoxGeometry(width + 0.02, height + 0.02, depth + 0.02)]} />
          <lineBasicMaterial
            color={borderLineColor}
            transparent={true}
            opacity={isDimmed ? 0.2 : 0.75}
            linewidth={isSelected || isActive ? 2 : 1}
          />
        </lineSegments>
      )}

      {/* Main Block Label */}
      <Text
        position={[0, subLabel ? 0.1 : 0, depth / 2 + 0.05]}
        fontSize={isCircular ? 0.24 : 0.21}
        color={isDimmed ? '#64748b' : '#ffffff'}
        anchorX="center"
        anchorY="middle"
        font="https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjp-Ek-_EeA.woff"
      >
        {label}
      </Text>

      {/* Sub-label */}
      {subLabel && (
        <Text
          position={[0, -0.16, depth / 2 + 0.05]}
          fontSize={0.11}
          color={isDimmed ? '#475569' : '#cbd5e1'}
          anchorX="center"
          anchorY="middle"
        >
          {subLabel}
        </Text>
      )}

      {/* Attention Port Indicators */}
      {type.includes('attention') && (
        <group position={[0, -height / 2 - 0.05, 0]}>
          {[-0.8, 0, 0.8].map((xOffset, pIdx) => (
            <mesh key={`port-${pIdx}`} position={[xOffset, 0, 0]}>
              <cylinderGeometry args={[0.06, 0.06, 0.1, 8]} />
              <meshBasicMaterial
                color={isActive ? '#fbbf24' : isSelected ? '#38bdf8' : '#38bdf8'}
                transparent={isDimmed}
                opacity={isDimmed ? 0.3 : 1}
              />
            </mesh>
          ))}
        </group>
      )}
    </group>
  );
};
