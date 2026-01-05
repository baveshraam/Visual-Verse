import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Html } from '@react-three/drei';
import { EffectComposer, Bloom, Vignette } from '@react-three/postprocessing';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';
import './MindMapView.css';

// 3D Node component
function Node3D({ node, position, isCenter, onClick, isHovered, onHover }) {
    const meshRef = useRef();
    const glowRef = useRef();

    const color = isCenter ? '#00f5ff' :
        node.type === 'keyword' ? '#b829ff' : '#ff2d7a';

    const size = isCenter ? 0.5 : 0.3;

    useFrame((state) => {
        if (meshRef.current) {
            // Gentle floating animation
            meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime + position[0]) * 0.05;

            // Pulse when hovered
            if (isHovered) {
                const scale = 1 + Math.sin(state.clock.elapsedTime * 4) * 0.1;
                meshRef.current.scale.setScalar(scale);
            } else {
                meshRef.current.scale.setScalar(1);
            }
        }
    });

    return (
        <group position={position}>
            {/* Glow sphere (outer) */}
            <mesh ref={glowRef}>
                <sphereGeometry args={[size * 1.5, 16, 16]} />
                <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={isHovered ? 0.15 : 0.08}
                />
            </mesh>

            {/* Main sphere */}
            <mesh
                ref={meshRef}
                onClick={onClick}
                onPointerOver={() => onHover(node.id)}
                onPointerOut={() => onHover(null)}
            >
                <sphereGeometry args={[size, 32, 32]} />
                <meshPhysicalMaterial
                    color={color}
                    metalness={0.3}
                    roughness={0.2}
                    emissive={color}
                    emissiveIntensity={isHovered ? 0.8 : 0.4}
                    transparent
                    opacity={0.9}
                />
            </mesh>

            {/* Label */}
            <Html
                position={[0, size + 0.3, 0]}
                center
                style={{
                    pointerEvents: 'none',
                    userSelect: 'none'
                }}
            >
                <div className={`node-label ${isCenter ? 'center' : ''} ${isHovered ? 'hovered' : ''}`}>
                    {node.label}
                </div>
            </Html>
        </group>
    );
}

// 3D Edge component
function Edge3D({ start, end, color = '#4361ee' }) {
    const points = useMemo(() => [
        new THREE.Vector3(...start),
        new THREE.Vector3(...end)
    ], [start, end]);

    return (
        <Line
            points={points}
            color={color}
            lineWidth={2}
            transparent
            opacity={0.6}
        />
    );
}

// Calculate node positions in 3D space
function calculatePositions(nodes, edges) {
    const positions = {};
    const centerNode = nodes.find(n => n.isCenter) || nodes[0];

    if (!centerNode) return positions;

    // Center node at origin
    positions[centerNode.id] = [0, 0, 0];

    // Other nodes in a sphere around center
    const otherNodes = nodes.filter(n => n.id !== centerNode.id);
    const radius = 4;

    otherNodes.forEach((node, i) => {
        const phi = Math.acos(-1 + (2 * i) / otherNodes.length);
        const theta = Math.sqrt(otherNodes.length * Math.PI) * phi;

        positions[node.id] = [
            radius * Math.cos(theta) * Math.sin(phi),
            radius * Math.sin(theta) * Math.sin(phi) * 0.6, // Flatten Y
            radius * Math.cos(phi)
        ];
    });

    return positions;
}

// 3D Mind Map Scene
function MindMapScene({ nodes, edges, onNodeClick }) {
    const [hoveredNode, setHoveredNode] = useState(null);

    const positions = useMemo(() =>
        calculatePositions(nodes, edges),
        [nodes, edges]
    );

    return (
        <>
            {/* Lighting */}
            <ambientLight intensity={0.3} />
            <pointLight position={[10, 10, 10]} color="#00f5ff" intensity={0.5} />
            <pointLight position={[-10, -10, -10]} color="#b829ff" intensity={0.3} />

            {/* Edges first (behind nodes) */}
            {edges.map((edge, i) => {
                const startPos = positions[edge.from];
                const endPos = positions[edge.to];
                if (!startPos || !endPos) return null;

                return (
                    <Edge3D
                        key={i}
                        start={startPos}
                        end={endPos}
                        color={edge.color || '#4361ee'}
                    />
                );
            })}

            {/* Nodes */}
            {nodes.map((node) => {
                const pos = positions[node.id];
                if (!pos) return null;

                return (
                    <Node3D
                        key={node.id}
                        node={node}
                        position={pos}
                        isCenter={node.isCenter}
                        isHovered={hoveredNode === node.id}
                        onHover={setHoveredNode}
                        onClick={() => onNodeClick?.(node)}
                    />
                );
            })}

            {/* Controls */}
            <OrbitControls
                enablePan={true}
                enableZoom={true}
                enableRotate={true}
                autoRotate={true}
                autoRotateSpeed={0.5}
                minDistance={3}
                maxDistance={15}
            />

            {/* Post-processing */}
            <EffectComposer>
                <Bloom
                    intensity={0.5}
                    luminanceThreshold={0.3}
                    luminanceSmoothing={0.9}
                />
                <Vignette offset={0.3} darkness={0.5} />
            </EffectComposer>
        </>
    );
}

// Main component
export default function MindMapView({
    mindmap,
    keyphrases,
    relations,
    isGenerating
}) {
    const [selectedNode, setSelectedNode] = useState(null);

    // Convert mindmap data to 3D-friendly format
    const { nodes, edges } = useMemo(() => {
        if (!keyphrases || keyphrases.length === 0) {
            return { nodes: [], edges: [] };
        }

        // Create nodes from keyphrases
        const nodes = keyphrases.map((kp, i) => ({
            id: kp.phrase || kp,
            label: kp.phrase || kp,
            score: kp.score || 1,
            isCenter: i === 0,
            type: 'keyword'
        }));

        // Create edges from relations
        const edges = (relations || []).map(rel => ({
            from: rel.source || rel.from,
            to: rel.target || rel.to,
            label: rel.relation || rel.type,
            color: getRelationColor(rel.relation || rel.type)
        }));

        return { nodes, edges };
    }, [keyphrases, relations]);

    if (!mindmap && !isGenerating && nodes.length === 0) {
        return (
            <div className="mindmap-placeholder">
                <div className="placeholder-icon">🧠</div>
                <h3>Mind Map Output</h3>
                <p>Enter informational text and click Generate to create an interactive 3D mind map</p>
            </div>
        );
    }

    return (
        <div className="mindmap-view">
            {/* Header */}
            <div className="mindmap-header">
                <h2 className="mindmap-title">
                    <span className="title-icon">🧠</span>
                    3D Mind Map
                </h2>
                <div className="mindmap-controls">
                    <span className="control-hint">🖱️ Drag to rotate • Scroll to zoom</span>
                </div>
            </div>

            {/* 3D Canvas */}
            <div className="mindmap-canvas-container">
                {nodes.length > 0 ? (
                    <Canvas
                        camera={{ position: [0, 0, 8], fov: 50 }}
                        dpr={[1, 2]}
                        gl={{ antialias: true }}
                    >
                        <color attach="background" args={['#0a0a0f']} />
                        <fog attach="fog" args={['#0a0a0f', 10, 25]} />

                        <MindMapScene
                            nodes={nodes}
                            edges={edges}
                            onNodeClick={setSelectedNode}
                        />
                    </Canvas>
                ) : (
                    <div className="loading-state">
                        <div className="loading-spinner"></div>
                        <span>Generating mind map...</span>
                    </div>
                )}
            </div>

            {/* Legend */}
            <div className="mindmap-legend">
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#00f5ff' }}></span>
                    <span>Central Topic</span>
                </div>
                <div className="legend-item">
                    <span className="legend-dot" style={{ background: '#b829ff' }}></span>
                    <span>Keywords</span>
                </div>
                <div className="legend-item">
                    <span className="legend-line"></span>
                    <span>Relationships</span>
                </div>
            </div>

            {/* Selected Node Info */}
            <AnimatePresence>
                {selectedNode && (
                    <motion.div
                        className="node-info-panel glass-card"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                    >
                        <button
                            className="close-btn"
                            onClick={() => setSelectedNode(null)}
                        >
                            ✕
                        </button>
                        <h4>{selectedNode.label}</h4>
                        {selectedNode.score && (
                            <p className="node-score">
                                Relevance: {(selectedNode.score * 100).toFixed(0)}%
                            </p>
                        )}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function getRelationColor(relationType) {
    const colors = {
        'is_a': '#00f5ff',
        'part_of': '#b829ff',
        'related_to': '#4361ee',
        'uses': '#ff2d7a',
        'causes': '#ff9500'
    };
    return colors[relationType] || '#4361ee';
}
