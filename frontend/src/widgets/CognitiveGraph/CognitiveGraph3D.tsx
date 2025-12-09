import React, { useState, useEffect, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import * as THREE from 'three';

// Типы для когнитивного графа
interface CognitiveNode {
  id: string;
  label: string;
  x: number;
  y: number;
  z: number;
  type: 'belief' | 'knowledge' | 'experience' | 'context' | 'emotion' | 'goal';
  confidence: number;
  activation: number;
  connections: string[];
}

interface CognitiveLink {
  source: string;
  target: string;
  type: 'causal' | 'associative' | 'temporal' | 'inhibitory' | 'supportive';
  strength: number;
}

interface CognitiveGraphData {
  nodes: CognitiveNode[];
  links: CognitiveLink[];
}

interface CognitiveGraph3DProps {
  data: CognitiveGraphData;
  className?: string;
  onNodeClick?: (node: CognitiveNode) => void;
  onNodeHover?: (node: CognitiveNode | null) => void;
}

// Компонент узла когнитивного графа
const CognitiveNode3D: React.FC<{
  node: CognitiveNode;
  isSelected: boolean;
  isHighlighted: boolean;
  onClick: (node: CognitiveNode) => void;
  onHover: (node: CognitiveNode | null) => void;
}> = ({ node, isSelected, isHighlighted, onClick, onHover }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  // Анимация пульсации для активных узлов
  useFrame(() => {
    if (meshRef.current) {
      const baseScale = node.activation > 0.7 ? 1.2 : 1.0;
      const pulse = 1 + Math.sin(Date.now() * 0.005 + node.id.length) * 0.05;
      meshRef.current.scale.setScalar(baseScale * pulse);
    }
  });

  // Цвета для разных типов узлов
  const nodeColors: Record<CognitiveNode['type'], string> = {
    belief: '#3b82f6',      // blue
    knowledge: '#8b5cf6',   // violet
    experience: '#ec4899',  // pink
    context: '#10b981',     // emerald
    emotion: '#f59e0b',     // amber
    goal: '#ef4444'         // red
  };

  // Размер узла на основе важности
  const size = 0.2 + (node.confidence * 0.3);

  return (
    <group
      position={[node.x, node.y, node.z]}
      onPointerEnter={(e: any) => {
        e.stopPropagation();
        setHovered(true);
        onHover(node);
      }}
      onPointerLeave={(e: any) => {
        e.stopPropagation();
        setHovered(false);
        onHover(null);
      }}
      onClick={(e: any) => {
        e.stopPropagation();
        onClick(node);
      }}
    >
      <mesh ref={meshRef}>
        <sphereGeometry args={[size, 16]} />
        <meshStandardMaterial
          color={isSelected ? '#fbbf24' : isHighlighted ? '#60a5fa' : nodeColors[node.type]}
          emissive={hovered ? '#ffffff' : '#000'}
          emissiveIntensity={hovered ? 0.2 : 0}
          opacity={0.85}
          transparent
        />
      </mesh>

      {/* Подпись узла при наведении */}
      {hovered && (
        <Html
          position={[0, size + 0.2, 0]}
          center
          distanceFactor={5}
          style={{
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: 'white',
            padding: '6px 10px',
            borderRadius: '6px',
            fontSize: '14px',
            fontWeight: 'bold',
            pointerEvents: 'none',
            whiteSpace: 'nowrap'
          }}
        >
          {node.label}
        </Html>
      )}
    </group>
  );
};

// Компонент связи когнитивного графа
const CognitiveLink3D: React.FC<{
  link: CognitiveLink;
  sourceNode: CognitiveNode | undefined;
  targetNode: CognitiveNode | undefined;
  isHighlighted: boolean;
}> = ({ link, sourceNode, targetNode, isHighlighted }) => {
  if (!sourceNode || !targetNode) return null;

  const sourcePos: [number, number, number] = [sourceNode.x, sourceNode.y, sourceNode.z];
  const targetPos: [number, number, number] = [targetNode.x, targetNode.y, targetNode.z];

  // Цвета для разных типов связей
  const linkColors: Record<CognitiveLink['type'], string> = {
    causal: '#3b82f6',        // blue
    associative: '#10b981',   // green
    temporal: '#8b5cf6',      // purple
    inhibitory: '#ef4444',    // red
    supportive: '#f59e0b'     // amber
  };

  // Толщина линии в зависимости от силы связи
  const lineWidth = 0.5 + (link.strength * 1.5);

  // Создаем геометрию для линии
  const points = [sourcePos, targetPos] as const;

  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flat())}
          itemSize={3}
          args={[new Float32Array(points.flat()), 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial
        color={isHighlighted ? '#fbbf24' : linkColors[link.type]}
        linewidth={lineWidth}
        transparent
        opacity={0.6}
      />
    </line>
  );
};

// Основная сцена когнитивного графа
const CognitiveGraphScene: React.FC<{
  data: CognitiveGraphData;
  selectedNode: CognitiveNode | null;
  highlightNodes: Set<string>;
  onNodeClick: (node: CognitiveNode) => void;
  onNodeHover: (node: CognitiveNode | null) => void;
}> = ({ data, selectedNode, highlightNodes, onNodeClick, onNodeHover }) => {
  const { camera } = useThree();

  // Устанавливаем начальную позицию камеры
  useEffect(() => {
    camera.position.set(15, 15, 15);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  // Создаем сетку для ориентации
  const grid = React.useMemo(() => new THREE.GridHelper(30, 30, new THREE.Color('#4b5563'), new THREE.Color('#4b5563')), []);
  grid.position.y = -5;

  return (
    <>
      {/* Освещение */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 0]} intensity={0.8} />
      <pointLight position={[-10, -10, 0]} intensity={0.4} />

      {/* Сетка для ориентации */}
      <primitive object={grid} />

      {/* Узлы */}
      {data.nodes.map((node) => (
        <CognitiveNode3D
          key={node.id}
          node={node}
          isSelected={selectedNode?.id === node.id}
          isHighlighted={highlightNodes.has(node.id)}
          onClick={onNodeClick}
          onHover={onNodeHover}
        />
      ))}

      {/* Связи */}
      {data.links.map((link) => {
        const sourceNode = data.nodes.find(n => n.id === link.source);
        const targetNode = data.nodes.find(n => n.id === link.target);

        return (
          <CognitiveLink3D
            key={`${link.source}-${link.target}`}
            link={link}
            sourceNode={sourceNode}
            targetNode={targetNode}
            isHighlighted={
              (selectedNode &&
                (selectedNode.id === link.source || selectedNode.id === link.target)) ||
              highlightNodes.has(link.source) ||
              highlightNodes.has(link.target)
            }
          />
        );
      })}

      {/* Управление орбитой */}
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
      />
    </>
  );
};

const CognitiveGraph3D: React.FC<CognitiveGraph3DProps> = ({
  data,
  className = '',
  onNodeClick,
  onNodeHover
}) => {
  const [selectedNode, setSelectedNode] = useState<CognitiveNode | null>(null);
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());

  // Обработчик выбора узла
  const handleNodeClick = (node: CognitiveNode) => {
    setSelectedNode(node);
    if (onNodeClick) onNodeClick(node);

    // Подсвечиваем связанные узлы
    const connectedNodes = new Set<string>([node.id]);
    data.links.forEach(link => {
      if (link.source === node.id || link.target === node.id) {
        connectedNodes.add(link.source);
        connectedNodes.add(link.target);
      }
    });
    setHighlightNodes(connectedNodes);
  };

  // Обработчик наведения на узел
  const handleNodeHover = (node: CognitiveNode | null) => {
    if (node) {
      // Подсвечиваем связанные узлы при наведении
      const connectedNodes = new Set<string>([node.id]);
      data.links.forEach(link => {
        if (link.source === node.id || link.target === node.id) {
          connectedNodes.add(link.source);
          connectedNodes.add(link.target);
        }
      });
      setHighlightNodes(connectedNodes);
    } else {
      setHighlightNodes(new Set());
    }

    if (onNodeHover) onNodeHover(node);
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white">Когнитивный Граф</h2>
        <div className="text-sm text-gray-600 dark:text-gray-300 flex items-center space-x-4">
          <span>Узлов: {data.nodes.length} | Связей: {data.links.length}</span>
        </div>
      </div>

      <div className="h-96 md:h-[400px] lg:h-[500px] xl:h-[600px] relative">
        <Canvas
          camera={{ position: [15, 15, 15], fov: 50 }}
          onCreated={({ gl }) => {
            // Адаптивная настройка рендерера для разных экранов
            const handleResize = () => {
              gl.setSize(
                document.querySelector('.h-96')?.clientWidth || window.innerWidth,
                document.querySelector('.h-96')?.clientHeight || 400
              );
            };

            window.addEventListener('resize', handleResize);
            handleResize(); // Инициализация размера

            return () => window.removeEventListener('resize', handleResize);
          }}
        >
          <CognitiveGraphScene
            data={data}
            selectedNode={selectedNode}
            highlightNodes={highlightNodes}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
          />
        </Canvas>
      </div>

      {/* Информация о выбранном узле */}
      {selectedNode && (
        <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600">
          <h3 className="font-medium text-gray-900 dark:text-white mb-2">Детали узла</h3>
          <div className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
            <p><span className="font-medium">ID:</span> {selectedNode.id}</p>
            <p><span className="font-medium">Метка:</span> {selectedNode.label}</p>
            <p><span className="font-medium">Тип:</span> {selectedNode.type}</p>
            <p><span className="font-medium">Уверенность:</span> {selectedNode.confidence.toFixed(2)}</p>
            <p><span className="font-medium">Активация:</span> {selectedNode.activation.toFixed(2)}</p>
          </div>
        </div>
      )}

      {/* Легенда */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-blue-500 mr-1"></div>
          <span>Убеждение</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-violet-500 mr-1"></div>
          <span>Знание</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-pink-500 mr-1"></div>
          <span>Опыт</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-emerald-500 mr-1"></div>
          <span>Контекст</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-amber-500 mr-1"></div>
          <span>Эмоция</span>
        </div>
        <div className="flex items-center">
          <div className="w-3 h-3 rounded-full bg-red-500 mr-1"></div>
          <span>Цель</span>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-4 text-xs text-gray-600 dark:text-gray-400">
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-blue-50 mr-1"></div>
          <span>Причинная связь</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-emerald-500 mr-1"></div>
          <span>Ассоциативная связь</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-violet-500 mr-1"></div>
          <span>Временная связь</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-red-500 mr-1"></div>
          <span>Ингибирующая связь</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-0.5 bg-amber-500 mr-1"></div>
          <span>Поддерживающая связь</span>
        </div>
      </div>
    </div>
  );
};

export default CognitiveGraph3D;
