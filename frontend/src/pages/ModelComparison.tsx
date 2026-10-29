import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  Award,
  Layers,
  Table,
  RefreshCw,
  Info,
  CheckCircle2,
  TrendingUp,
  Activity,
  SlidersHorizontal
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  Cell
} from 'recharts';
import { api, ModelComparisonItem } from '../services/api';

const MODEL_COLORS: Record<string, string> = {
  lstm_baseline: '#81786E',
  transformer: '#C66A4E',
  transformer_ssl: '#059669',
};

export const ModelComparison: React.FC = () => {
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
        const filtered = (data || []).filter((m) => m.model_variant !== 'lstm_transformer');
        setComparisons(filtered);
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
    top_3_accuracy: Number(((m.top_3_accuracy ?? 0) * 100).toFixed(2)),
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
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
            <BarChart3 className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
            Model Comparison & Evaluation Benchmarks
          </h1>
          <p className="text-sm text-[#81786E] mt-1">
            Quantitative analysis across LSTM Baseline, LSTM+Transformer Hybrid, Transformer Encoder, and Transformer SSL.
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={evaluating}
          className="inline-flex items-center px-4 py-2 rounded-lg bg-[#C66A4E] hover:bg-[#B95D3B] text-white text-xs font-semibold shadow-sm transition"
        >
          <Activity className={`w-3.5 h-3.5 mr-2 ${evaluating ? 'animate-spin' : ''}`} />
          {evaluating ? 'Evaluating Splits...' : 'Run Evaluation Suite'}
        </button>
      </div>

      {comparisons.length === 0 ? (
        <div className="p-12 rounded-xl bg-white border border-[#EFE7DB] text-center space-y-3">
          <Info className="w-8 h-8 text-[#C66A4E] mx-auto" />
          <h3 className="font-semibold text-base text-[#302721]">No Evaluation Records Found</h3>
          <p className="text-xs text-[#81786E] max-w-md mx-auto">
            To view comparative benchmark metrics, train model checkpoints in the Model Lab and click "Run Evaluation Suite". No metrics are ever fabricated.
          </p>
        </div>
      ) : (
        <>
          {/* Comparative Metrics Table */}
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold text-[#302721] flex items-center">
                <Table className="w-4 h-4 mr-2 text-[#C66A4E]" />
                Empirical Evaluation Benchmark (Test Set: {comparisons[0]?.test_samples} Samples)
              </h2>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#82917A]/10 text-[#82917A] font-semibold border border-[#82917A]/30">
                Data: {comparisons[0]?.data_provenance.split(' ')[0]}
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
                        {((c.top_3_accuracy ?? 0) * 100).toFixed(2)}%
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

          {/* Comparative Metric Bar Chart */}
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-[#302721] flex items-center">
                <SlidersHorizontal className="w-4 h-4 mr-2 text-[#C66A4E]" />
                Metric Visualizer: {metricLabelMap[selectedMetric]}
              </h2>

              <div className="flex items-center space-x-2">
                <span className="text-xs text-[#81786E]">Metric:</span>
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

            <div className="h-64 w-full">
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

          {/* Confusion Matrix Inspection */}
          <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-[#302721] flex items-center">
                <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
                Confusion Matrix Analysis
              </h2>
              <select
                value={selectedModelForCM}
                onChange={(e) => setSelectedModelForCM(e.target.value)}
                className="px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-xs font-medium text-[#302721]"
              >
                <option value="transformer">Transformer Encoder (Primary)</option>
                <option value="transformer_ssl">Transformer SSL</option>
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
      )}
    </div>
  );
};
