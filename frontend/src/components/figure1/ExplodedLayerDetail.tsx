import React from 'react';
import { Text } from '@react-three/drei';
import type { LayerAttentionTrace, DecoderStepTrace } from '../../types/trace';

interface ExplodedLayerDetailProps {
  layerType: 'encoder' | 'decoder';
  layerIndex: number;
  encoderTrace?: LayerAttentionTrace;
  decoderStepTrace?: DecoderStepTrace;
  onClose: () => void;
}

export const ExplodedLayerDetail: React.FC<ExplodedLayerDetailProps> = ({
  layerType,
  layerIndex,
  onClose
}) => {
  const isEncoder = layerType === 'encoder';

  return (
    <group position={[0, 3.5, 3.5]}>
      {/* Background Dimming Backdrop Plate */}
      <mesh position={[0, 0, -1]}>
        <planeGeometry args={[14, 11]} />
        <meshBasicMaterial color="#020617" transparent={true} opacity={0.88} />
      </mesh>

      {/* Header Plaque */}
      <mesh position={[0, 4.8, 0]}>
        <boxGeometry args={[12, 0.7, 0.1]} />
        <meshStandardMaterial color="#0f172a" metalness={0.8} roughness={0.3} />
      </mesh>
      <Text position={[0, 4.8, 0.1]} fontSize={0.28} color="#38bdf8" anchorX="center" anchorY="middle">
        {isEncoder 
          ? `ENCODER BLOCK (LAYER ${layerIndex}) — EXPLODED SUBLAYER MECHANISM`
          : `DECODER BLOCK (LAYER ${layerIndex}) — EXPLODED SUBLAYER MECHANISM`
        }
      </Text>

      {/* Close Button in 3D */}
      <group position={[5.4, 4.8, 0.1]} onClick={onClose}>
        <mesh>
          <boxGeometry args={[0.9, 0.45, 0.1]} />
          <meshStandardMaterial color="#dc2626" />
        </mesh>
        <Text position={[0, 0, 0.08]} fontSize={0.16} color="#ffffff" anchorX="center" anchorY="middle">
          CLOSE [X]
        </Text>
      </group>

      {isEncoder ? (
        /* ENCODER SUBLAYER EXPLODED DETAIL */
        <group position={[0, 0, 0]}>
          {/* SUBLAYER 1: MULTI-HEAD ATTENTION INTERNAL MECHANISM */}
          <group position={[-3.6, 1.2, 0]}>
            <mesh position={[0, 1.8, 0]}>
              <boxGeometry args={[5.2, 0.5, 0.2]} />
              <meshStandardMaterial color="#0284c7" />
            </mesh>
            <Text position={[0, 1.8, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              1. MULTI-HEAD SELF-ATTENTION (h=2)
            </Text>

            {/* Q, K, V Projection Blocks */}
            {[-1.5, 0, 1.5].map((xOff, idx) => {
              const name = idx === 0 ? 'W_Q (Linear)' : idx === 1 ? 'W_K (Linear)' : 'W_V (Linear)';
              return (
                <group key={`qkv-${idx}`} position={[xOff, 0.9, 0]}>
                  <mesh>
                    <boxGeometry args={[1.2, 0.6, 0.2]} />
                    <meshStandardMaterial color="#1e3a8a" />
                  </mesh>
                  <Text position={[0, 0, 0.12]} fontSize={0.14} color="#60a5fa" anchorX="center">
                    {name}
                  </Text>
                </group>
              );
            })}

            {/* Scaled Dot-Product Core Cell */}
            <mesh position={[0, -0.2, 0]}>
              <boxGeometry args={[4.8, 1.0, 0.2]} />
              <meshStandardMaterial color="#0f172a" />
            </mesh>
            <Text position={[0, 0.1, 0.12]} fontSize={0.16} color="#38bdf8" anchorX="center">
              Scaled Dot-Product Attention
            </Text>
            <Text position={[0, -0.2, 0.12]} fontSize={0.13} color="#f59e0b" anchorX="center">
              Softmax( (Q · K^T) / √d_k ) · V
            </Text>

            {/* Output Projection W_O */}
            <mesh position={[0, -1.2, 0]}>
              <boxGeometry args={[3.2, 0.5, 0.2]} />
              <meshStandardMaterial color="#1e3a8a" />
            </mesh>
            <Text position={[0, -1.2, 0.12]} fontSize={0.14} color="#60a5fa" anchorX="center">
              W_O Output Projection
            </Text>
          </group>

          {/* SUBLAYER 2: ADD & NORM 1 */}
          <group position={[3.6, 2.5, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 1.2, 0.2]} />
              <meshStandardMaterial color="#0f766e" />
            </mesh>
            <Text position={[0, 0.2, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              ADD &amp; NORM 1 (POST-LN)
            </Text>
            <Text position={[0, -0.2, 0.15]} fontSize={0.14} color="#99f6e4" anchorX="center">
              x_norm = LayerNorm(x + Sublayer(x))
            </Text>
          </group>

          {/* SUBLAYER 3: POSITION-WISE FEED-FORWARD */}
          <group position={[3.6, -0.5, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 1.8, 0.2]} />
              <meshStandardMaterial color="#0369a1" />
            </mesh>
            <Text position={[0, 0.5, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              2. FEED-FORWARD NETWORK (FFN)
            </Text>
            <Text position={[0, 0.1, 0.15]} fontSize={0.14} color="#bae6fd" anchorX="center">
              Dense(d_ff=32, activation="relu")
            </Text>
            <Text position={[0, -0.3, 0.15]} fontSize={0.14} color="#bae6fd" anchorX="center">
              Dense(d_model=16, linear)
            </Text>
          </group>

          {/* SUBLAYER 4: ADD & NORM 2 */}
          <group position={[3.6, -2.6, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 1.0, 0.2]} />
              <meshStandardMaterial color="#0f766e" />
            </mesh>
            <Text position={[0, 0.15, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              ADD &amp; NORM 2
            </Text>
            <Text position={[0, -0.2, 0.15]} fontSize={0.13} color="#99f6e4" anchorX="center">
              Output to Next Layer / Memory Bus
            </Text>
          </group>
        </group>
      ) : (
        /* DECODER SUBLAYER EXPLODED DETAIL */
        <group position={[0, 0, 0]}>
          {/* Sublayer 1: Masked Self-Attention */}
          <group position={[-3.6, 1.8, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 1.6, 0.2]} />
              <meshStandardMaterial color="#b45309" />
            </mesh>
            <Text position={[0, 0.4, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              1. MASKED MULTI-HEAD ATTENTION
            </Text>
            <Text position={[0, 0.0, 0.15]} fontSize={0.13} color="#fde68a" anchorX="center">
              Causal Mask M_ij = -∞ for j &gt; i
            </Text>
            <Text position={[0, -0.35, 0.15]} fontSize={0.12} color="#fca5a5" anchorX="center">
              Zeros out future token lookahead
            </Text>
          </group>

          {/* Sublayer 2: Cross-Attention */}
          <group position={[-3.6, -1.0, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 1.8, 0.2]} />
              <meshStandardMaterial color="#ea580c" />
            </mesh>
            <Text position={[0, 0.5, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              2. ENCODER-DECODER CROSS-ATTENTION
            </Text>
            <Text position={[0, 0.1, 0.15]} fontSize={0.13} color="#ffedd5" anchorX="center">
              Q: from Decoder Sublayer 1
            </Text>
            <Text position={[0, -0.25, 0.15]} fontSize={0.13} color="#fed7aa" anchorX="center">
              K, V: from Final Encoder Stack Output
            </Text>
          </group>

          {/* Sublayer 3: FFN & Add Norms */}
          <group position={[3.6, 0.4, 0]}>
            <mesh>
              <boxGeometry args={[5.2, 2.4, 0.2]} />
              <meshStandardMaterial color="#0f766e" />
            </mesh>
            <Text position={[0, 0.8, 0.15]} fontSize={0.2} color="#ffffff" anchorX="center">
              3. FEED FORWARD &amp; 3 ADD-NORMS
            </Text>
            <Text position={[0, 0.35, 0.15]} fontSize={0.13} color="#99f6e4" anchorX="center">
              3 Residual Bypass Paths
            </Text>
            <Text position={[0, 0.0, 0.15]} fontSize={0.13} color="#ffffff" anchorX="center">
              FFN: max(0, y W_1 + b_1) W_2 + b_2
            </Text>
            <Text position={[0, -0.45, 0.15]} fontSize={0.13} color="#99f6e4" anchorX="center">
              Final LayerNorm before Linear &amp; Softmax
            </Text>
          </group>
        </group>
      )}
    </group>
  );
};
