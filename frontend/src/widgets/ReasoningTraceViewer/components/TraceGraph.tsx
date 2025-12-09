import React, { useEffect, useRef, Suspense, lazy, useMemo } from 'react';
import { GraphNode, GraphLink, BaseNode, BaseLink } from '../types';
import { getNodeColor, getNodeSize } from '../utils/graphUtils';


// Тип для пропсов компонента TraceGraph
interface TraceGraphProps {
  nodes: BaseNode[];
  links: BaseLink[];
  onNodeClick?: (node: BaseNode) => void;
  width?: number;
  height?: number;
  className?: string;
}

// Динамический импорт с обработкой ошибок
const ForceGraph2D = lazy(() =>
  import('react-force-graph-2d')
    .then(module => ({
      default: module.default || (() => null)
    }))
    .catch(error => {
      console.error('Ошибка загрузки react-force-graph-2d:', error);
      return {
        default: () => (
          <div className="text-red-600 p-4">
            Не удалось загрузить визуализацию графа. Пожалуйста, обновите страницу.
          </div>
        )
      };
    })
);

const TraceGraph: React.FC<TraceGraphProps> = ({
  nodes: propNodes = [],
  links: propLinks = [],
  onNodeClick,
  width = 800,
  height = 600,
  className = ''
}) => {
  // Преобразуем узлы и связи в формат, совместимый с react-force-graph-2d
  const { nodes: graphNodes, links: graphLinks } = useMemo(() => {
    // Создаем карту узлов для быстрого доступа
    const nodesMap = new Map<string, any>();

    // Создаем узлы
    const nodesList: any[] = propNodes.map((node, idx) => {
      const nodeId = node.id || `node-${idx}`;
      const graphNode: GraphNode = {
        ...node,
        id: nodeId,
        label: node.label || `Узел ${idx + 1}`,
        type: node.type || 'default',
        description: node.description || '',
        confidence: node.confidence ?? 0.5,
        // Добавляем цвет и размер по умолчанию
        color: getNodeColor(node.confidence),
        size: getNodeSize(node.type)
      };

      nodesMap.set(nodeId, graphNode);
      return graphNode;
    });

    // Создаем связи
    const linksList: any[] = propLinks.map((link, idx) => {
      const source = typeof link.source === 'string' ? link.source :
        (typeof link.source === 'object' && 'id' in link.source ? link.source.id : `source-${idx}`);
      const target = typeof link.target === 'string' ? link.target :
        (typeof link.target === 'object' && 'id' in link.target ? link.target.id : `target-${idx}`);

      return {
        ...link,
        source,
        target,
        value: 1,
        color: 'rgba(148, 163, 184, 0.7)'
      } as GraphLink;
    });

    return { nodes: nodesList, links: linksList };
  }, [propNodes, propLinks]);

  const fgRef = useRef<any>(null);

  useEffect(() => {
    if (fgRef.current) {
      try {
        fgRef.current.zoomToFit(400);
      } catch (error) {
        console.error('Ошибка при масштабировании графа:', error);
      }
    }
  }, [graphNodes, graphLinks]);

  const handleNodeClick = (node: any) => {
    if (onNodeClick) {
      // Удаляем внутренние свойства графа передачей наружу
      const { x, y, vx, vy, fx, fy, size, color, ...rest } = node;
      onNodeClick(rest as BaseNode);
    }
  };

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-4 ${className}`}>
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-white">Граф рассуждений</h3>
      <div className="h-[500px] relative">
        <Suspense fallback={
          <div className="flex items-center justify-center h-full">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        }>
          <ForceGraph2D
            ref={fgRef}
            width={width}
            height={height}
            graphData={{ nodes: graphNodes, links: graphLinks }}
            nodeLabel={(node: any) => `${node.label || 'Узел'}<br/>${node.description || ''}`}
            nodeColor={(node: any) => getNodeColor(node.confidence)}
            nodeVal={(node: any) => getNodeSize(node.type)}
            linkColor={() => 'rgba(148, 163, 184, 0.7)'}
            linkWidth={1.5}
            linkDirectionalArrowLength={6}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.1}
            onNodeClick={handleNodeClick}
            onNodeRightClick={() => {
              try {
                fgRef.current?.zoomToFit(400);
              } catch (error) {
                console.error('Ошибка при масштабировании:', error);
              }
            }}
            cooldownTicks={100}
            onEngineStop={() => {
              try {
                fgRef.current?.zoomToFit(400);
              } catch (error) {
                console.error('Ошибка при остановке движка:', error);
              }
            }}
          />
        </Suspense>
      </div>
    </div>
  );
};

export default TraceGraph;
