import React, { useState, useEffect, useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Line, Text, Html, useTexture } from '@react-three/drei';
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

interface OptimizedCognitiveGraph3DProps {
  data: CognitiveGraphData;
  className?: string;
  onNodeClick?: (node: CognitiveNode) => void;
  onNodeHover?: (node: CognitiveNode | null) => void;
  maxNodes?: number; // Максимальное количество узлов для отображения
}

// Оптимизированный компонент узла когнитивного графа
const OptimizedCognitiveNode3D: React.FC<{
  node: CognitiveNode;
  isSelected: boolean;
  isHighlighted: boolean;
  onClick: (node: CognitiveNode) => void;
  onHover: (node: CognitiveNode | null) => void;
}> = React.memo(({ node, isSelected, isHighlighted, onClick, onHover }) => {
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
      <mesh
        ref={meshRef}
        onClick={(e: any) => e.stopPropagation()}
      >
        <sphereGeometry args={[size, 8, 8]} /> {/* Уменьшено количество сегментов для производительности */}
        <meshStandardMaterial 
          color={isSelected ? '#fbbf24' : isHighlighted ? '#60a5fa' : nodeColors[node.type]}
          emissive={hovered ? '#ffffff' : '#0000'}
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
});

// Оптимизированный компонент связи когнитивного графа
const OptimizedCognitiveLink3D: React.FC<{
  link: CognitiveLink;
  sourceNode: CognitiveNode | undefined;
 targetNode: CognitiveNode | undefined;
  isHighlighted: boolean;
}> = React.memo(({ link, sourceNode, targetNode, isHighlighted }) => {
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

  // Создаем геометрию для стрелки
  const direction = new THREE.Vector3(...targetPos).sub(new THREE.Vector3(...sourcePos));
  const length = direction.length();
  const midpoint = new THREE.Vector3(...sourcePos).add(direction.clone().multiplyScalar(0.5));

  // Вычисляем ориентацию стрелки
  const arrowDirection = direction.clone().normalize();
  const arrowLength = 0.3;
  
  // Создаем стрелку для направления связи
  const arrowStart = new THREE.Vector3(...targetPos).sub(arrowDirection.clone().multiplyScalar(arrowLength));
  const arrowPoints = [arrowStart.toArray(), targetPos] as const;

  return (
    <group>
      {/* Основная линия связи */}
      <Line
        points={[sourcePos, targetPos]}
        color={isHighlighted ? '#fbbf24' : linkColors[link.type]}
        lineWidth={lineWidth}
        transparent
        opacity={0.6}
      />
      
      {/* Стрелка для направления связи */}
      <Line
        points={[arrowStart.toArray(), targetPos]}
        color={isHighlighted ? '#fbbf24' : linkColors[link.type]}
        lineWidth={lineWidth * 1.5}
        transparent
        opacity={0.8}
      />
    </group>
  );
});

// Оптимизированная сцена когнитивного графа
const OptimizedCognitiveGraphScene: React.FC<{
 data: CognitiveGraphData;
  selectedNode: CognitiveNode | null;
  highlightNodes: Set<string>;
  onNodeClick: (node: CognitiveNode) => void;
  onNodeHover: (node: CognitiveNode | null) => void;
  maxNodes: number;
}> = React.memo(({ data, selectedNode, highlightNodes, onNodeClick, onNodeHover, maxNodes }) => {
 const { camera } = useThree();
  
  // Устанавливаем начальную позицию камеры
  useEffect(() => {
    camera.position.set(15, 15, 15);
    camera.lookAt(0, 0, 0);
  }, [camera]);

  // Ограничиваем количество отображаемых узлов для производительности
  const nodesToRender = useMemo(() => {
    return data.nodes.slice(0, maxNodes);
  }, [data.nodes, maxNodes]);

  // Группируем связи по типу для оптимизации рендеринга
  const linksByType = useMemo(() => {
    const grouped: Record<CognitiveLink['type'], CognitiveLink[]> = {
      causal: [],
      associative: [],
      temporal: [],
      inhibitory: [],
      supportive: []
    };

    data.links.forEach(link => {
      if (nodesToRender.some(node => node.id === link.source) && 
          nodesToRender.some(node => node.id === link.target)) {
        grouped[link.type].push(link);
      }
    });

    return grouped;
  }, [data.links, nodesToRender]);

  return (
    <>
      {/* Освещение */}
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={0.8} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} />
      
      {/* Сетка для ориентации */}
      <gridHelper args={[30, 30, '#4b5563', '#4b563']} position={[0, -5, 0]} />
      
      {/* Узлы */}
      {nodesToRender.map((node) => (
        <OptimizedCognitiveNode3D
          key={node.id}
          node={node}
          isSelected={selectedNode?.id === node.id}
          isHighlighted={highlightNodes.has(node.id)}
          onClick={onNodeClick}
          onHover={onNodeHover}
        />
      ))}
      
      {/* Связи по типам */}
      {Object.entries(linksByType).map(([type, links]) => (
        <group key={type}>
          {links.map((link) => {
            const sourceNode = data.nodes.find(n => n.id === link.source);
            const targetNode = data.nodes.find(n => n.id === link.target);
            
            return (
              <OptimizedCognitiveLink3D
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
        </group>
      ))}
      
      {/* Управление орбитой */}
      <OrbitControls 
        enableDamping
        dampingFactor={0.05}
        minDistance={5}
        maxDistance={50}
      />
    </>
  );
});

const OptimizedCognitiveGraph3D: React.FC<OptimizedCognitiveGraph3DProps> = ({ 
  data, 
  className = '', 
  onNodeClick,
  onNodeHover,
  maxNodes = 100  // По умолчанию ограничиваем 100 узлами
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

  // Отображаем предупреждение, если данных слишком много
  const showWarning = data.nodes.length > maxNodes;

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white">Когнитивный Граф (Оптимизированный)</h2>
        <div className="text-sm text-gray-600 dark:text-gray-300 flex items-center space-x-4">
          <span>Узлов: {data.nodes.length} | Связей: {data.links.length}</span>
          {showWarning && (
            <span className="text-xs text-amber-500">
              Показано {maxNodes} из {data.nodes.length} узлов
            </span>
          )}
        </div>
      </div>

      <div className="h-96 relative">
        <Canvas 
          camera={{ position: [15, 15, 15], fov: 50 }}
          gl={{ 
            antialias: true,
            alpha: true,
            powerPreference: "high-performance" // Оптимизация для производительности
          }}
        >
          <OptimizedCognitiveGraphScene
            data={data}
            selectedNode={selectedNode}
            highlightNodes={highlightNodes}
            onNodeClick={handleNodeClick}
            onNodeHover={handleNodeHover}
            maxNodes={maxNodes}
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
          <div className="w-4 h-0.5 bg-blue-500 mr-1"></div>
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

export default OptimizedCognitiveGraph3D;