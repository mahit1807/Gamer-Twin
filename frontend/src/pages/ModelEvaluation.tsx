import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  Award,
  Layers,
  Table,
  CheckCircle2,
  AlertCircle,
  Activity,
  SlidersHorizontal,
  Info,
  Cpu,
  TrendingDown
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell
} from 'recharts';
import { api, ModelComparisonItem } from '../services/api';
import { GameplayPatternExplorer } from '../components/GameplayPatternExplorer';

const MODEL_COLORS: Record<string, string> = {
  lstm_baseline: '#81786E',
  lstm_transformer: '#B95D3B',
  transformer: '#C66A4E',
  transformer_ssl: '#059669',
};

export const ModelEvaluation: React.FC = () => {
  const [comparisons, setComparisons] = useState<ModelComparisonItem[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('accuracy');
  const [selectedModelForCM, setSelectedModelForCM] = useState<string>('transformer');
  const [cmData, setCmData] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);

  const loadData = () => {
    setLoading(true);
    api.getExperimentComparison()
      .then((data) => {
        setComparisons(data || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
  };

  const loadCM = (modelName: string) => {
    api.getSingleModelReport(modelName)
      .then((res) => {
        setCmData(res);
      })
      .catch(() => setCmData(null));
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    loadCM(selectedModelForCM);
  }, [selectedModelForCM]);

  const handleRunEvaluation = () => {
    setEvaluating(true);
    api.runAllEvaluations()
      .then(() => {
        loadData();
        loadCM(selectedModelForCM);
        setEvaluating(false);
      })
      .catch((err) => {
        alert(`Evaluation error: ${err.message}`);
        setEvaluating(false);
      });
  };

  const chartData = comparisons.map((m) => ({
    name: m.model_display_name,
    id: m.model_variant,
    accuracy: Number((m.accuracy * 100).toFixed(2)),
    top_3_accuracy: Number((m.top_3_accuracy * 100).toFixed(2)),
    f1_macro: Number((m.f1_macro * 100).toFixed(2)),
    f1_weighted: Number((m.f1_weighted * 100).toFixed(2)),
    precision_macro: Number((m.precision_macro * 100).toFixed(2)),
    recall_macro: Number((m.recall_macro * 100).toFixed(2)),
  }));

  const metricLabelMap: Record<string, string> = {
    accuracy: 'Top-1 Accuracy (%)',
    top_3_accuracy: 'Top-3 Accuracy (%)',
    f1_macro: 'F1-Score Macro (%)',
    f1_weighted: 'F1-Score Weighted (%)',
    precision_macro: 'Precision Macro (%)',
    recall_macro: 'Recall Macro (%)',
  };

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* 1. Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded bg-[#C66A4E]/10 text-[#C66A4E] text-xs font-semibold mb-1">
            <BarChart3 className="w-3.5 h-3.5" />
            <span>CrossFPS Empirical Evidence & Multi-Game Benchmarks</span>
          </div>
          <h1 className="text-2xl font-bold text-[#302721] tracking-tight">
            Model Evaluation & Comparative Analysis
          </h1>
          <p className="text-xs sm:text-sm text-[#81786E] mt-1">
            Evaluates deep sequence architectures on held-out CrossFPS evaluation splits (3,443 clips across 7 FPS titles) without data leakage.
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={evaluating}
          className="inline-flex items-center px-4 py-2 rounded-lg bg-[#C66A4E] hover:bg-[#B95D3B] text-white text-xs font-semibold shadow-xs transition"
        >
          <Activity className={`w-3.5 h-3.5 mr-2 ${evaluating ? 'animate-spin' : ''}`} />
          {evaluating ? 'Evaluating Test Split...' : 'Re-Evaluate Checkpoints'}
        </button>
      </div>

      {/* 2. Evaluation Protocol & Split Methodology */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Evaluation Protocol & Data Separation
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="font-bold text-[#302721] block">Train Split (66.7%)</span>
            <span className="font-mono text-[#81786E] block">768 Sequences / 1,600 Rows</span>
            <p className="text-[11px] text-[#81786E] pt-1">
              Used strictly for parameter optimization and fitting z-score scalers.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="font-bold text-[#302721] block">Validation Split (16.7%)</span>
            <span className="font-mono text-[#81786E] block">192 Sequences / 400 Rows</span>
            <p className="text-[11px] text-[#81786E] pt-1">
              Used for early stopping checkpoint selection and hyperparameter tuning.
            </p>
          </div>
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="font-bold text-[#302721] block">Held-Out Test Split (16.7%)</span>
            <span className="font-mono text-[#82917A] font-semibold block">192 Sequences / 400 Rows</span>
            <p className="text-[11px] text-[#81786E] pt-1">
              Completely unseen test partition used to report all benchmark scores below.
            </p>
          </div>
        </div>
      </div>

      {/* 3. The 4 Evaluated Architectures */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <Cpu className="w-4 h-4 mr-2 text-[#C66A4E]" />
          The Four Implemented Model Variants
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="font-bold text-[#302721]">1. LSTM Baseline</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-white border border-[#EFE7DB] text-[#81786E]">Baseline</span>
            </div>
            <p className="text-[11px] text-[#81786E] leading-relaxed">
              Standard 2-layer Recurrent Neural Network sequentially propagating hidden state $h_t$. Serves as the inductive baseline for sequential gameplay.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="font-bold text-[#302721]">2. LSTM + Transformer Hybrid</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-white border border-[#EFE7DB] text-[#B95D3B]">Hybrid</span>
            </div>
            <p className="text-[11px] text-[#81786E] leading-relaxed">
              Combines an initial LSTM recurrent layer to track frame-to-frame camera delta with a Transformer Encoder to capture long-range tactical context.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#C66A4E] space-y-1.5 ring-1 ring-[#C66A4E]/30">
            <div className="flex justify-between items-center">
              <span className="font-bold text-[#C66A4E]">3. Transformer Encoder (Primary)</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-[#C66A4E] text-white font-bold">Primary</span>
            </div>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              Pure self-attention model with sinusoidal positional encodings and dual temporal pooling (average sequence context + last frame token).
            </p>
          </div>

          <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="font-bold text-[#302721]">4. Transformer + SSL Pretraining</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-white border border-[#EFE7DB] text-[#059669]">Self-Supervised</span>
            </div>
            <p className="text-[11px] text-[#81786E] leading-relaxed">
              Two-stage training: Stage 1 reconstructs masked telemetry frames (Masked Autoencoding); Stage 2 fine-tunes the backbone on next-action prediction.
            </p>
          </div>
        </div>
      </div>

      {/* 4. Comparative Evaluation Table */}
      {comparisons.length > 0 ? (
        <>
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-bold text-[#302721] flex items-center">
                <Table className="w-4 h-4 mr-2 text-[#C66A4E]" />
                Quantitative Test Benchmark (Held-out Test Split: {comparisons[0]?.test_samples} Samples)
              </h2>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#82917A]/10 text-[#82917A] font-semibold border border-[#82917A]/30">
                Data Provenance: {comparisons[0]?.data_provenance.split(' ')[0]}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-[#EFE7DB] bg-[#F8F5EF] text-[#81786E]">
                    <th className="py-2.5 px-3 font-semibold">Model Architecture</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Top-1 Accuracy</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Top-3 Accuracy</th>
                    <th className="py-2.5 px-3 font-semibold text-right">F1 (Macro)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">F1 (Weighted)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Precision (Macro)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Recall (Macro)</th>
                    <th className="py-2.5 px-3 font-semibold text-center">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map((c) => (
                    <tr key={c.model_variant} className="border-b border-[#EFE7DB] hover:bg-[#F8F5EF] transition">
                      <td className="py-3 px-3 font-bold text-[#302721] flex items-center">
                        <span
                          className="w-2.5 h-2.5 rounded-full mr-2"
                          style={{ backgroundColor: MODEL_COLORS[c.model_variant] || '#C66A4E' }}
                        ></span>
                        {c.model_display_name}
                      </td>
                      <td className="py-3 px-3 font-mono font-semibold text-[#302721] text-right">
                        {(c.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 font-mono font-semibold text-[#82917A] text-right">
                        {(c.top_3_accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 font-mono text-[#81786E] text-right">
                        {(c.f1_macro * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 font-mono text-[#302721] text-right">
                        {(c.f1_weighted * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 font-mono text-[#81786E] text-right">
                        {(c.precision_macro * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 font-mono text-[#81786E] text-right">
                        {(c.recall_macro * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-3 text-center">
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-[#82917A]/10 text-[#82917A]">
                          {c.status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Metric Comparison Bar Chart */}
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h2 className="text-sm font-bold text-[#302721] flex items-center">
                <SlidersHorizontal className="w-4 h-4 mr-2 text-[#C66A4E]" />
                Metric Visualizer: {metricLabelMap[selectedMetric]}
              </h2>

              <div className="flex items-center space-x-2">
                <span className="text-xs text-[#81786E]">Select Metric:</span>
                <select
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value)}
                  className="px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-xs font-medium text-[#302721]"
                >
                  <option value="accuracy">Top-1 Accuracy</option>
                  <option value="top_3_accuracy">Top-3 Accuracy</option>
                  <option value="f1_macro">F1-Score (Macro)</option>
                  <option value="f1_weighted">F1-Score (Weighted)</option>
                  <option value="precision_macro">Precision (Macro)</option>
                  <option value="recall_macro">Recall (Macro)</option>
                </select>
              </div>
            </div>

            <div className="h-60 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EFE7DB" />
                  <XAxis dataKey="name" stroke="#81786E" fontSize={11} interval={0} />
                  <YAxis stroke="#81786E" fontSize={11} domain={[0, 100]} unit="%" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#EFE7DB', borderRadius: '8px', fontSize: '12px' }}
                    formatter={(val: any) => [`${val}%`, metricLabelMap[selectedMetric]]}
                  />
                  <Bar dataKey={selectedMetric} radius={[4, 4, 0, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.id}
                        fill={MODEL_COLORS[entry.id] || '#C66A4E'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Confusion Matrix Analysis */}
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-bold text-[#302721] flex items-center">
                  <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
                  $9 \times 9$ Confusion Matrix
                </h2>
                <p className="text-[11px] text-[#81786E] mt-0.5">
                  Inspects per-class classification hits and empirical class confusion.
                </p>
              </div>

              <select
                value={selectedModelForCM}
                onChange={(e) => setSelectedModelForCM(e.target.value)}
                className="px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-xs font-medium text-[#302721]"
              >
                <option value="transformer">Transformer Encoder (Primary)</option>
                <option value="transformer_ssl">Transformer SSL</option>
                <option value="lstm_transformer">LSTM + Transformer</option>
                <option value="lstm_baseline">LSTM Baseline</option>
              </select>
            </div>

            {cmData && cmData.classes ? (
              <div className="overflow-x-auto">
                <table className="w-full text-center text-[10px] border-collapse font-mono">
                  <thead>
                    <tr className="bg-[#F8F5EF] text-[#81786E]">
                      <th className="py-2 px-2 text-left font-sans text-xs">True \ Pred</th>
                      {cmData.classes.map((cls: string) => (
                        <th key={cls} className="py-2 px-2">{cls}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {cmData.classes.map((trueCls: string, i: number) => (
                      <tr key={trueCls} className="border-b border-[#EFE7DB]">
                        <td className="py-2 px-2 text-left font-sans text-xs font-semibold text-[#302721]">
                          {trueCls}
                        </td>
                        {cmData.classes.map((_: string, j: number) => {
                          const val = cmData.raw_data.confusion_matrix?.[i]?.[j] ?? 0;
                          const isDiagonal = i === j;
                          return (
                            <td
                              key={j}
                              className={`py-2 px-2 font-semibold ${
                                isDiagonal && val > 0
                                  ? 'bg-[#82917A]/20 text-[#82917A]'
                                  : val > 0
                                  ? 'bg-[#C66A4E]/10 text-[#C66A4E]'
                                  : 'text-[#81786E]/40'
                              }`}
                            >
                              {val}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="p-6 text-center text-xs text-[#81786E]">
                No confusion matrix available for this model yet.
              </div>
            )}
          </div>
        </>
      ) : (
        <div className="p-12 rounded-xl bg-white border border-[#EFE7DB] text-center space-y-3">
          <Info className="w-8 h-8 text-[#C66A4E] mx-auto" />
          <h3 className="font-bold text-base text-[#302721]">No Evaluation Outputs Found</h3>
          <p className="text-xs text-[#81786E] max-w-md mx-auto leading-relaxed">
            Click "Re-Evaluate Checkpoints" to execute test split evaluation over saved PyTorch models.
          </p>
        </div>
      )}

      {/* 5. Player Behaviour Embeddings & Gameplay Pattern Explorer */}
      <GameplayPatternExplorer />

      {/* 6. Scientific Interpretation & Current Limitations */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-3">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <Info className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Scientific Interpretation & Project Limitations
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs leading-relaxed text-[#302721]">
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="font-bold text-[#302721] block">Interpretation of Findings</span>
            <p className="text-[11px] text-[#81786E]">
              1. <strong>Top-3 Accuracy (~72.4%)</strong> indicates that while high tactical ambiguity exists at a single frame boundary, the correct action consistently resides in the model's top confidence set.
              <br />
              2. <strong>Class Imbalance</strong>: Move frames dominate gameplay telemetry, creating lower recall on rare micro-actions (Grenade, Inspect) without focal loss re-weighting.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="font-bold text-[#302721] block">Documented Limitations & Scope</span>
            <p className="text-[11px] text-[#81786E]">
              1. <strong>Offline Behavioral Modeling</strong>: GAMER-TWIN predicts actions conditioned on history; it does not issue live commands or control game state via Reinforcement Learning.
              <br />
              2. <strong>Autonomous DQN Agent</strong>: Deep Q-Learning game-playing agents require an interactive game engine sandbox and are documented strictly as future research extensions.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
