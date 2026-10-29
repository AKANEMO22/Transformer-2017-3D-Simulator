import React, { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

interface GlowingTokenProps {
  text: string;
  subText?: string;
  position: [number, number, number];
  color?: string;
  glowColor?: string;
  scale?: number;
  highlighted?: boolean;
  opacity?: number;
}

export const GlowingToken: React.FC<GlowingTokenProps> = ({
  text,
  subText,
  position,
  color = '#00f0ff',
  glowColor = '#0284c7',
  scale = 1,
  highlighted = false,
  opacity = 1
}) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const borderRef = useRef<THREE.LineSegments>(null);

  useFrame((state) => {
    if (highlighted && meshRef.current) {
      const t = state.clock.getElapsedTime();
      meshRef.current.position.y = position[1] + Math.sin(t * 3) * 0.08;
    }
  });

  return (
    <group position={position} scale={[scale, scale, scale]}>
      {/* Base Glass Tile */}
      <mesh ref={meshRef} castShadow receiveShadow>
        <boxGeometry args={[1.5, 0.75, 0.15]} />
        <meshPhysicalMaterial
          color={highlighted ? '#10b981' : color}
          emissive={highlighted ? '#059669' : glowColor}
          emissiveIntensity={highlighted ? 0.8 : 0.25}
          roughness={0.2}
          metalness={0.1}
          transparent={true}
          opacity={opacity * 0.85}
          transmission={0.3}
        />
      </mesh>

      {/* Glowing Neon Outline */}
      <lineSegments ref={borderRef}>
        <edgesGeometry args={[new THREE.BoxGeometry(1.52, 0.77, 0.16)]} />
        <lineBasicMaterial
          color={highlighted ? '#34d399' : color}
          transparent={true}
          opacity={opacity * 0.9}
          linewidth={2}
        />
      </lineSegments>

      {/* Main Token Label (3D Text) */}
      <Text
        position={[0, subText ? 0.08 : 0, 0.12]}
        fontSize={0.24}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        font="https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjp-Ek-_EeA.woff"
      >
        {text}
      </Text>

      {/* Subtext / Position / ID */}
      {subText && (
        <Text
          position={[0, -0.18, 0.12]}
          fontSize={0.12}
          color={highlighted ? '#6ee7b7' : '#94a3b8'}
          anchorX="center"
          anchorY="middle"
        >
          {subText}
        </Text>
      )}
    </group>
  );
};
