import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Compass,
  Cpu,
  Layers,
  Sparkles,
  Info,
  ExternalLink,
  RotateCw,
  Sliders,
  Eye,
  Activity,
  Crosshair,
  Gamepad2,
  ChevronRight,
  ShieldAlert,
  HelpCircle,
  Maximize2
} from 'lucide-react';
import {
  api,
  GameplayEmbeddingsResponse,
  GameplayEmbeddingPoint,
  SequenceDetailResponse
} from '../services/api';

interface GameplayPatternExplorerProps {
  className?: string;
  defaultModel?: string;
}

export const GameplayPatternExplorer: React.FC<GameplayPatternExplorerProps> = ({
  className = '',
  defaultModel = 'transformer'
}) => {
  const navigate = useNavigate();

  // Configuration state
  const [selectedModel, setSelectedModel] = useState<string>(defaultModel);
  const [sampleSize, setSampleSize] = useState<number>(600);
  const [colorMode, setColorMode] = useState<'game' | 'action'>('game');
  const [selectedGameFilter, setSelectedGameFilter] = useState<string>('all');

  // Data state
  const [data, setData] = useState<GameplayEmbeddingsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Selection state
  const [selectedPoint, setSelectedPoint] = useState<GameplayEmbeddingPoint | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<GameplayEmbeddingPoint | null>(null);

  // Detailed frame sequence telemetry state
  const [seqDetail, setSeqDetail] = useState<SequenceDetailResponse | null>(null);
  const [loadingDetail, setLoadingDetail] = useState<boolean>(false);
  const [showFramesDrawer, setShowFramesDrawer] = useState<boolean>(false);

  // SVG dimensions
  const svgRef = useRef<SVGSVGElement | null>(null);
  const width = 640;
  const height = 480;
  const padding = 55;

  // Load embeddings from API
  const fetchEmbeddings = (forceRefresh = false) => {
    setLoading(true);
    setError(null);
    api.getGameplayEmbeddings(selectedModel, sampleSize, forceRefresh)
      .then((res) => {
        setData(res);
        setLoading(false);
        // Pre-select first point if none selected
        if (res.points && res.points.length > 0) {
          if (!selectedPoint || !res.points.some(p => p.sequence_index === selectedPoint.sequence_index)) {
            setSelectedPoint(res.points[0]);
          }
        }
      })
      .catch((err) => {
        console.error('Error fetching gameplay embeddings:', err);
        setError(err.message || 'Failed to load gameplay embeddings');
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchEmbeddings(false);
  }, [selectedModel, sampleSize]);

  // Fetch sequence frame detail when selected point changes
  useEffect(() => {
    if (selectedPoint) {
      setLoadingDetail(true);
      api.getSequenceDetail(selectedPoint.sequence_index)
        .then((detail) => {
          setSeqDetail(detail);
          setLoadingDetail(false);
        })
        .catch((err) => {
          console.warn('Failed to load sequence detail:', err);
          setSeqDetail(null);
          setLoadingDetail(false);
        });
    } else {
      setSeqDetail(null);
    }
  }, [selectedPoint]);

  // Filtered points
  const filteredPoints = useMemo(() => {
    if (!data?.points) return [];
    if (selectedGameFilter === 'all') return data.points;
    return data.points.filter((p) => p.game === selectedGameFilter);
  }, [data, selectedGameFilter]);

  // Compute 2D coordinate domain bounds
  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    if (!data?.points || data.points.length === 0) {
      return { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };
    }
    const xs = data.points.map((p) => p.x);
    const ys = data.points.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const xPad = (maxX - minX) * 0.08 || 1;
    const yPad = (maxY - minY) * 0.08 || 1;
    return {
      xMin: minX - xPad,
      xMax: maxX + xPad,
      yMin: minY - yPad,
      yMax: maxY + yPad
    };
  }, [data]);

  // Projection math
  const scaleX = (x: number) => {
    return padding + ((x - xMin) / (xMax - xMin)) * (width - 2 * padding);
  };
  const scaleY = (y: number) => {
    // Invert Y so higher is up
    return height - padding - ((y - yMin) / (yMax - yMin)) * (height - 2 * padding);
  };

  // Color mapper
  const getPointColor = (point: GameplayEmbeddingPoint) => {
    if (colorMode === 'game') {
      return data?.legend?.games?.[point.game] || '#C66A4E';
    } else {
      return data?.legend?.actions?.[point.dominant_action] || '#81786E';
    }
  };

  const handleOpenInPrediction = (point: GameplayEmbeddingPoint) => {
    // Navigate to /predict with sequence index and game
    navigate(`/predict?seq=${point.sequence_index}&game=${encodeURIComponent(point.game)}&model=${encodeURIComponent(selectedModel)}`);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Section Header */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1">
            <div className="inline-flex items-center space-x-2 text-xs font-bold text-[#C66A4E] uppercase tracking-wider">
              <Compass className="w-4 h-4 text-[#C66A4E]" />
              <span>Learned Latent Space Analysis</span>
            </div>
            <h2 className="text-xl font-bold text-[#302721] tracking-tight">
              Player Behaviour Embeddings & Gameplay Pattern Explorer
            </h2>
            <p className="text-xs text-[#81786E] max-w-2xl">
              Extracts 64-dimensional sequence embeddings directly from the trained Transformer encoder
              hidden states over genuine CrossFPS controller sequences (<code className="bg-[#F8F5EF] px-1 py-0.5 rounded text-[#302721]">data/CrossFPS-val/</code>) and visualizes them in 2D space using Principal Component Analysis (PCA).
            </p>
          </div>

          {/* Model & Refresh Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-xs font-semibold bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] rounded-lg px-3 py-2 outline-none focus:border-[#C66A4E]"
            >
              <option value="transformer">Transformer Encoder (Supervised, 91.5% Acc)</option>
              <option value="transformer_ssl">Self-Supervised Pretrained Transformer (SSL)</option>
            </select>

            <select
              value={sampleSize}
              onChange={(e) => setSampleSize(Number(e.target.value))}
              className="text-xs font-semibold bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] rounded-lg px-2.5 py-2 outline-none focus:border-[#C66A4E]"
            >
              <option value={300}>300 Sequences</option>
              <option value={600}>600 Sequences</option>
              <option value={1000}>1,000 Sequences</option>
            </select>

            <button
              onClick={() => fetchEmbeddings(true)}
              disabled={loading}
              className="inline-flex items-center space-x-1.5 px-3 py-2 rounded-lg text-xs font-bold bg-[#C66A4E] text-white hover:bg-[#B95D3B] disabled:opacity-50 transition-colors shadow-xs"
              title="Force recomputation from checkpoint"
            >
              <RotateCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>{loading ? 'Embedding...' : 'Recompute'}</span>
            </button>
          </div>
        </div>

        {/* Technical Provenance Chips */}
        {data && (
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-[#EFE7DB]/60 text-[11px]">
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] font-medium">
              <Cpu className="w-3.5 h-3.5 mr-1.5 text-[#C66A4E]" />
              <strong>Checkpoint:</strong>&nbsp;{data.checkpoint}
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] font-medium">
              <Layers className="w-3.5 h-3.5 mr-1.5 text-[#82917A]" />
              <strong>Latent Dim:</strong>&nbsp;{data.embedding_dim}-D pooled hidden state
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] font-medium">
              <Sparkles className="w-3.5 h-3.5 mr-1.5 text-[#B95D3B]" />
              <strong>PCA Projection:</strong>&nbsp;PC1 ({((data.explained_variance_ratio[0] || 0) * 100).toFixed(1)}%) + PC2 ({((data.explained_variance_ratio[1] || 0) * 100).toFixed(1)}%) = {data.total_variance_explained_pct}% Variance
            </span>
            <span className="inline-flex items-center px-2.5 py-1 rounded-md bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] font-medium">
              <Activity className="w-3.5 h-3.5 mr-1.5 text-[#302721]" />
              <strong>Sampled:</strong>&nbsp;{data.total_sequences_embedded} / {data.total_sequences_available.toLocaleString()} verified sequences
            </span>
          </div>
        )}
      </div>

      {/* Main Grid: Scatter Plot + Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: 2D Scatter Plot & Visual Controls (7 cols) */}
        <div className="lg:col-span-7 bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-xs flex flex-col justify-between space-y-4">
          {/* Scatter Plot Controls Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center space-x-2">
              <span className="font-bold text-[#302721]">Color by:</span>
              <div className="inline-flex rounded-lg border border-[#EFE7DB] bg-[#F8F5EF] p-0.5">
                <button
                  onClick={() => setColorMode('game')}
                  className={`px-2.5 py-1 rounded-md font-semibold text-xs transition-colors ${
                    colorMode === 'game'
                      ? 'bg-white text-[#C66A4E] shadow-xs'
                      : 'text-[#81786E] hover:text-[#302721]'
                  }`}
                >
                  Game Title
                </button>
                <button
                  onClick={() => setColorMode('action')}
                  className={`px-2.5 py-1 rounded-md font-semibold text-xs transition-colors ${
                    colorMode === 'action'
                      ? 'bg-white text-[#C66A4E] shadow-xs'
                      : 'text-[#81786E] hover:text-[#302721]'
                  }`}
                >
                  Dominant Action
                </button>
              </div>
            </div>

            {/* Filter by Game */}
            <div className="flex items-center space-x-2">
              <span className="font-bold text-[#302721]">Filter Game:</span>
              <select
                value={selectedGameFilter}
                onChange={(e) => setSelectedGameFilter(e.target.value)}
                className="text-xs bg-[#F8F5EF] border border-[#EFE7DB] text-[#302721] rounded-lg px-2 py-1 outline-none font-medium"
              >
                <option value="all">All Games ({data?.points?.length || 0})</option>
                {data?.legend?.games &&
                  Object.keys(data.legend.games).map((g) => (
                    <option key={g} value={g}>
                      {g.replace(/__/g, ': ').replace(/_/g, ' ').toUpperCase()}
                    </option>
                  ))}
              </select>
            </div>
          </div>

          {/* Interactive SVG Plot Container */}
          <div className="relative w-full aspect-[4/3] bg-[#F8F5EF] border border-[#EFE7DB] rounded-xl overflow-hidden flex items-center justify-center">
            {loading ? (
              <div className="flex flex-col items-center justify-center space-y-3 p-8">
                <div className="w-10 h-10 border-3 border-[#C66A4E] border-t-transparent rounded-full animate-spin" />
                <p className="text-xs font-bold text-[#302721]">Extracting 64-D Latent Embeddings...</p>
                <p className="text-[11px] text-[#81786E]">Running PyTorch encoder forward pass & PCA projection</p>
              </div>
            ) : error ? (
              <div className="p-6 text-center space-y-2 max-w-md">
                <ShieldAlert className="w-8 h-8 text-[#C66A4E] mx-auto" />
                <p className="text-xs font-bold text-[#302721]">Embedding Extraction Notice</p>
                <p className="text-[11px] text-[#81786E]">{error}</p>
                <button
                  onClick={() => fetchEmbeddings(true)}
                  className="mt-2 text-xs font-bold text-[#C66A4E] underline hover:text-[#B95D3B]"
                >
                  Retry with verified checkpoint
                </button>
              </div>
            ) : (
              <svg
                ref={svgRef}
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-full select-none cursor-crosshair"
              >
                {/* Background Grid Lines */}
                <line
                  x1={padding}
                  y1={height / 2}
                  x2={width - padding}
                  y2={height / 2}
                  stroke="#EFE7DB"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                />
                <line
                  x1={width / 2}
                  y1={padding}
                  x2={width / 2}
                  y2={height - padding}
                  stroke="#EFE7DB"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                />

                {/* Plot Border */}
                <rect
                  x={padding}
                  y={padding}
                  width={width - 2 * padding}
                  height={height - 2 * padding}
                  fill="none"
                  stroke="#EFE7DB"
                  strokeWidth="1.5"
                />

                {/* Axis Labels */}
                <text
                  x={width / 2}
                  y={height - 18}
                  textAnchor="middle"
                  className="text-[11px] fill-[#81786E] font-medium"
                >
                  Principal Component 1 (PC1) · {((data?.explained_variance_ratio[0] || 0) * 100).toFixed(1)}% variance
                </text>
                <text
                  x={18}
                  y={height / 2}
                  textAnchor="middle"
                  transform={`rotate(-90 18 ${height / 2})`}
                  className="text-[11px] fill-[#81786E] font-medium"
                >
                  Principal Component 2 (PC2) · {((data?.explained_variance_ratio[1] || 0) * 100).toFixed(1)}% variance
                </text>

                {/* Scatter Points */}
                {filteredPoints.map((pt) => {
                  const cx = scaleX(pt.x);
                  const cy = scaleY(pt.y);
                  const isSelected = selectedPoint?.sequence_index === pt.sequence_index;
                  const isHovered = hoveredPoint?.sequence_index === pt.sequence_index;
                  const color = getPointColor(pt);

                  return (
                    <g key={pt.id}>
                      {/* Selection Glow / Pulse */}
                      {isSelected && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={9}
                          fill="none"
                          stroke={color}
                          strokeWidth="2.5"
                          opacity="0.85"
                          className="animate-pulse"
                        />
                      )}
                      {/* Active Point Circle */}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isSelected ? 5.5 : isHovered ? 4.5 : 3.2}
                        fill={color}
                        stroke="#FFFFFF"
                        strokeWidth={isSelected ? 1.5 : 0.8}
                        opacity={isSelected ? 1.0 : isHovered ? 1.0 : 0.78}
                        className="transition-all duration-150 cursor-pointer"
                        onMouseEnter={() => setHoveredPoint(pt)}
                        onMouseLeave={() => setHoveredPoint(null)}
                        onClick={() => setSelectedPoint(pt)}
                      />
                    </g>
                  );
                })}
              </svg>
            )}

            {/* Hover Tooltip Overlay */}
            {hoveredPoint && (
              <div
                className="absolute pointer-events-none bg-[#302721] text-white text-[11px] p-2.5 rounded-lg shadow-lg space-y-1 z-10 max-w-xs border border-white/20"
                style={{
                  left: Math.min(Math.max(scaleX(hoveredPoint.x) - 80, 10), width - 180),
                  top: Math.max(scaleY(hoveredPoint.y) - 65, 10)
                }}
              >
                <div className="font-bold flex items-center justify-between">
                  <span>{hoveredPoint.id}</span>
                  <span className="text-[#EFE7DB] font-normal">{hoveredPoint.game_name}</span>
                </div>
                <div className="text-[10px] text-[#EFE7DB]/80">
                  Action: <strong className="text-white">{hoveredPoint.dominant_action}</strong>
                </div>
                <div className="text-[10px] text-white/60">
                  (PC1: {hoveredPoint.x.toFixed(2)}, PC2: {hoveredPoint.y.toFixed(2)}) · Click to inspect
                </div>
              </div>
            )}
          </div>

          {/* Dynamic Color Legend */}
          <div className="pt-2 border-t border-[#EFE7DB]">
            <span className="text-[11px] font-bold text-[#302721] uppercase tracking-wider block mb-1.5">
              {colorMode === 'game' ? 'Game Telemetry Provenance' : 'Observed Controller Action Clusters'}
            </span>
            <div className="flex flex-wrap gap-2 text-xs">
              {colorMode === 'game' && data?.legend?.games ? (
                Object.entries(data.legend.games).map(([gKey, hex]) => (
                  <div
                    key={gKey}
                    className="inline-flex items-center space-x-1.5 bg-[#F8F5EF] px-2.5 py-1 rounded-md border border-[#EFE7DB]"
                  >
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: hex }} />
                    <span className="text-[11px] font-semibold text-[#302721]">
                      {gKey.replace(/__/g, ': ').replace(/_/g, ' ').toUpperCase()}
                    </span>
                  </div>
                ))
              ) : colorMode === 'action' && data?.legend?.actions ? (
                Object.entries(data.legend.actions).map(([actName, hex]) => (
                  <div
                    key={actName}
                    className="inline-flex items-center space-x-1.5 bg-[#F8F5EF] px-2.5 py-1 rounded-md border border-[#EFE7DB]"
                  >
                    <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: hex }} />
                    <span className="text-[11px] font-semibold text-[#302721]">{actName}</span>
                  </div>
                ))
              ) : null}
            </div>
          </div>
        </div>

        {/* Right: Selected Sequence Inspector (5 cols) */}
        <div className="lg:col-span-5 bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-xs flex flex-col justify-between space-y-5">
          {selectedPoint ? (
            <div className="space-y-4">
              {/* Point Title & Actions */}
              <div className="flex items-start justify-between pb-3 border-b border-[#EFE7DB]">
                <div>
                  <div className="inline-flex items-center space-x-1.5 text-[11px] font-bold text-[#C66A4E]">
                    <Activity className="w-3.5 h-3.5" />
                    <span>Selected Sequence #{selectedPoint.sequence_index + 1}</span>
                  </div>
                  <h3 className="text-base font-bold text-[#302721] tracking-tight">
                    {selectedPoint.id}
                  </h3>
                  <p className="text-[11px] text-[#81786E] font-mono truncate max-w-[260px]">
                    Clip: {selectedPoint.clip_id}
                  </p>
                </div>

                <span
                  className="px-2.5 py-1 rounded-full text-xs font-bold text-white shadow-xs"
                  style={{ backgroundColor: getPointColor(selectedPoint) }}
                >
                  {selectedPoint.game_name}
                </span>
              </div>

              {/* Factual Behavior Summary */}
              <div className="p-3.5 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
                <div className="text-[11px] font-bold text-[#302721] uppercase tracking-wider flex items-center">
                  <Gamepad2 className="w-3.5 h-3.5 mr-1.5 text-[#C66A4E]" />
                  Observed Telemetry Summary
                </div>
                <p className="text-xs text-[#302721] leading-relaxed">
                  {selectedPoint.summary}
                </p>
                <div className="flex flex-wrap gap-1.5 pt-1 text-[11px]">
                  <span className="px-2 py-0.5 rounded bg-white border border-[#EFE7DB] text-[#302721] font-medium">
                    Dominant: <strong>{selectedPoint.dominant_action}</strong>
                  </span>
                  <span className="px-2 py-0.5 rounded bg-white border border-[#EFE7DB] text-[#81786E]">
                    Genre: {selectedPoint.genre}
                  </span>
                </div>
              </div>

              {/* 10-DoF Physical Controller Metrics */}
              <div className="space-y-2">
                <span className="text-[11px] font-bold text-[#302721] uppercase tracking-wider block">
                  Measured Physical Telemetry (16 Frames)
                </span>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  {/* Left Stick (Move / Strafe) */}
                  <div className="p-2.5 rounded-lg border border-[#EFE7DB] bg-white space-y-1">
                    <span className="text-[10px] text-[#81786E] font-bold uppercase tracking-wider">
                      Left Stick (Movement)
                    </span>
                    <div className="text-xs font-bold text-[#302721]">
                      {selectedPoint.controller_stats.movement_label}
                    </div>
                    <div className="text-[10px] text-[#81786E] flex justify-between">
                      <span>LX: {selectedPoint.controller_stats.mean_strafe_lx.toFixed(2)}</span>
                      <span>LY: {selectedPoint.controller_stats.mean_move_ly.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* Right Stick (Camera Look) */}
                  <div className="p-2.5 rounded-lg border border-[#EFE7DB] bg-white space-y-1">
                    <span className="text-[10px] text-[#81786E] font-bold uppercase tracking-wider">
                      Right Stick (Camera)
                    </span>
                    <div className="text-xs font-bold text-[#302721]">
                      {selectedPoint.controller_stats.camera_label}
                    </div>
                    <div className="text-[10px] text-[#81786E] flex justify-between">
                      <span>RX: {selectedPoint.controller_stats.mean_turn_rx.toFixed(2)}</span>
                      <span>RY: {selectedPoint.controller_stats.mean_pitch_ry.toFixed(2)}</span>
                    </div>
                  </div>

                  {/* Discrete Trigger Frequencies */}
                  <div className="p-2.5 rounded-lg border border-[#EFE7DB] bg-white space-y-1 col-span-2">
                    <span className="text-[10px] text-[#81786E] font-bold uppercase tracking-wider">
                      Button & Trigger Occurrence Rates
                    </span>
                    <div className="grid grid-cols-3 gap-2 pt-1 text-center">
                      <div className="p-1 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                        <span className="text-[10px] text-[#81786E] block">Fire (RT)</span>
                        <strong className="text-xs text-[#C66A4E]">
                          {Math.round(selectedPoint.controller_stats.fire_rate * 100)}%
                        </strong>
                      </div>
                      <div className="p-1 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                        <span className="text-[10px] text-[#81786E] block">Scope (LT)</span>
                        <strong className="text-xs text-[#4A6FA5]">
                          {Math.round(selectedPoint.controller_stats.ads_rate * 100)}%
                        </strong>
                      </div>
                      <div className="p-1 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                        <span className="text-[10px] text-[#81786E] block">Jump (A)</span>
                        <strong className="text-xs text-[#82917A]">
                          {Math.round(selectedPoint.controller_stats.jump_rate * 100)}%
                        </strong>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Latent Coordinate Positions */}
              <div className="p-2.5 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] flex items-center justify-between text-xs text-[#81786E]">
                <span>PCA Coordinates:</span>
                <span className="font-mono text-[#302721] font-semibold">
                  PC1 = {selectedPoint.x.toFixed(3)}, PC2 = {selectedPoint.y.toFixed(3)}
                </span>
              </div>

              {/* Seamless Bridge to Next Prediction Studio */}
              <div className="space-y-2 pt-2">
                <button
                  onClick={() => handleOpenInPrediction(selectedPoint)}
                  className="w-full inline-flex items-center justify-center space-x-2 py-3 px-4 rounded-xl text-xs font-bold bg-[#C66A4E] text-white hover:bg-[#B95D3B] transition-colors shadow-sm"
                >
                  <Crosshair className="w-4 h-4" />
                  <span>Test Next Action in Prediction Studio</span>
                  <ChevronRight className="w-4 h-4" />
                </button>
                <p className="text-[10px] text-center text-[#81786E]">
                  Opens sequence #{selectedPoint.sequence_index + 1} directly in the Next Prediction pipeline.
                </p>
              </div>

              {/* Optional 16-Frame Telemetry Drawer Toggle */}
              {seqDetail && (
                <div className="pt-2 border-t border-[#EFE7DB]">
                  <button
                    onClick={() => setShowFramesDrawer(!showFramesDrawer)}
                    className="text-xs font-semibold text-[#81786E] hover:text-[#C66A4E] inline-flex items-center space-x-1"
                  >
                    <span>{showFramesDrawer ? 'Hide' : 'Inspect'} 16-Frame Telemetry Track</span>
                    <ChevronRight className={`w-3.5 h-3.5 transition-transform ${showFramesDrawer ? 'rotate-90' : ''}`} />
                  </button>

                  {showFramesDrawer && (
                    <div className="mt-2.5 max-h-48 overflow-y-auto border border-[#EFE7DB] rounded-lg p-2 bg-[#F8F5EF] space-y-1 font-mono text-[10px]">
                      <div className="grid grid-cols-6 font-bold text-[#302721] border-b border-[#EFE7DB] pb-1">
                        <span>Frame</span>
                        <span>LX</span>
                        <span>LY</span>
                        <span>RX</span>
                        <span>RY</span>
                        <span>Triggers</span>
                      </div>
                      {seqDetail.frames.map((fr) => (
                        <div key={fr.frame} className="grid grid-cols-6 text-[#81786E]">
                          <span className="font-semibold text-[#302721]">T+{fr.frame}</span>
                          <span>{fr.lx.toFixed(2)}</span>
                          <span>{fr.ly.toFixed(2)}</span>
                          <span>{fr.rx.toFixed(2)}</span>
                          <span>{fr.ry.toFixed(2)}</span>
                          <span>
                            {fr.rt_fire > 0.5 ? 'RT ' : ''}
                            {fr.lt_ads > 0.5 ? 'LT ' : ''}
                            {fr.a_jump > 0.5 ? 'A ' : ''}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="p-8 text-center text-xs text-[#81786E] space-y-2">
              <Eye className="w-8 h-8 text-[#81786E]/40 mx-auto" />
              <p className="font-bold text-[#302721]">No Sequence Selected</p>
              <p className="text-[11px]">Click any point on the 2D scatter plot to inspect its telemetry.</p>
            </div>
          )}
        </div>
      </div>

      {/* Scientific Explanation & Mathematical Transparency Banner */}
      <div className="bg-[#F8F5EF] border border-[#EFE7DB] rounded-xl p-5 space-y-3">
        <div className="flex items-center space-x-2 text-xs font-bold text-[#302721]">
          <Info className="w-4 h-4 text-[#C66A4E]" />
          <span>Scientific Interpretation of Learned Gameplay Embeddings</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs leading-relaxed text-[#302721]">
          <div className="p-3.5 rounded-lg bg-white border border-[#EFE7DB] space-y-1.5">
            <span className="font-bold text-[#302721] block">
              1. What Latent Proximity Represents
            </span>
            <p className="text-[11px] text-[#81786E]">
              Points close together share similar pooled hidden-state activations in the Transformer encoder. Proximity reflects correlated temporal control inputs (analog stick velocities, sustained aim vectors, and button timing) across 16 frames of genuine CrossFPS controller telemetry.
            </p>
          </div>

          <div className="p-3.5 rounded-lg bg-white border border-[#EFE7DB] space-y-1.5">
            <span className="font-bold text-[#302721] block">
              2. Scientific Scope & Prohibited Inferences
            </span>
            <p className="text-[11px] text-[#81786E]">
              Proximity in latent space does <strong>not</strong> prove identical player personality, psychological intent, or tactical skill level. GAMER-TWIN strictly avoids assigning unsupported subjective labels such as “aggressive player” or “defensive player,” grounding all visual clusters in verified physical controller measurements.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
