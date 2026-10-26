import React, { useEffect, useState } from 'react';
import {
  Database,
  Table,
  CheckCircle2,
  AlertTriangle,
  Layers,
  Search,
  RefreshCw,
  FolderOpen,
  PieChart as PieIcon,
  Clock,
  Info
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from 'recharts';
import { api, DatasetSummary } from '../services/api';

const WARM_COLORS = [
  '#C66A4E', '#B95D3B', '#82917A', '#302721', '#81786E',
  '#D97706', '#059669', '#4B5563', '#DC2626'
];

export const DatasetExplorer: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Preview state
  const [previewData, setPreviewData] = useState<{ columns: string[]; rows: any[]; filename: string } | null>(null);
  const [previewType, setPreviewType] = useState<'ticks' | 'events'>('ticks');
  const [previewLimit, setPreviewLimit] = useState<number>(10);
  const [searchTerm, setSearchTerm] = useState('');

  const loadData = () => {
    setLoading(true);
    setError(null);
    api.getDatasetSummary()
      .then((data) => {
        setSummary(data);
        return api.getDatasetPreview(previewLimit, previewType);
      })
      .then((prev) => {
        setPreviewData(prev);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load dataset details.');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadData();
  }, [previewType, previewLimit]);

  // Chart data from actual class distribution
  const chartData = summary
    ? Object.entries(summary.class_distribution).map(([name, count]) => ({
        name,
        count,
        percentage: ((count / Math.max(1, summary.total_records)) * 100).toFixed(1),
      }))
    : [];

  const filteredRows = previewData
    ? previewData.rows.filter((row) =>
        Object.values(row).some((val) =>
          String(val).toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    : [];

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* 1. Page Header & Source Link */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
              <Database className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
              EgoCS-400K Dataset Explorer
            </h1>
          </div>
          <p className="text-xs sm:text-sm text-[#81786E] mt-1">
            Grounded inspection of high-frequency Counter-Strike telemetry, temporal action chains, and sequence construction.
          </p>
        </div>

        <button
          onClick={loadData}
          className="inline-flex items-center px-3.5 py-2 rounded-lg bg-white border border-[#EFE7DB] text-xs font-semibold text-[#302721] hover:bg-[#F8F5EF] shadow-xs transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 mr-1.5 text-[#C66A4E] ${loading ? 'animate-spin' : ''}`} />
          Refresh Schema
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-center">
          <AlertTriangle className="w-4 h-4 mr-2 text-red-500 flex-shrink-0" />
          {error}
        </div>
      )}

      {/* 2. Dataset Availability & Provenance Audit */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <FolderOpen className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Dataset Availability & Provenance Tier
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="text-[#81786E] block font-medium">1. Official Cloned Repo</span>
            <div className="flex items-center text-[#82917A] font-bold">
              <CheckCircle2 className="w-4 h-4 mr-1.5" /> Cloned (/Dataset)
            </div>
            <span className="text-[11px] text-[#81786E] block pt-1">
              Contains parser pipelines (<code>parse_dem</code>, <code>overlay</code>, <code>action_timeline</code>).
            </span>
          </div>

          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="text-[#81786E] block font-medium">2. Active Telemetry Stream</span>
            <div className={`font-bold flex items-center ${summary?.is_synthetic ? 'text-[#C66A4E]' : 'text-[#82917A]'}`}>
              <span className="w-2 h-2 rounded-full mr-1.5 bg-current"></span>
              {summary?.dataset_source.split(' ')[0]}
            </div>
            <span className="text-[11px] text-[#81786E] block pt-1">
              {summary?.is_synthetic
                ? 'Validating schemas & sequence shapes via data/sample_data/'
                : 'Reading raw telemetry from data/raw/'}
            </span>
          </div>

          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
            <span className="text-[#81786E] block font-medium">3. Full 400K Raw Dataset</span>
            <div className="font-semibold text-[#81786E]">Hugging Face Archive</div>
            <span className="text-[11px] text-[#81786E] block pt-1">
              10,000+ hrs of raw demo videos distributed separately from code repository.
            </span>
          </div>
        </div>
      </div>

      {/* 3. Real Telemetry Record Preview */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-bold text-[#302721] flex items-center">
              <Table className="w-4 h-4 mr-2 text-[#C66A4E]" />
              Real Telemetry Record Preview ({previewData?.filename})
            </h2>
            <p className="text-[11px] text-[#81786E] mt-0.5">
              High-frequency tick coordinates, view pitch/yaw, and keyboard/mouse traces read directly from file.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <div className="relative">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-2.5 text-[#81786E]" />
              <input
                type="text"
                placeholder="Filter columns/values..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-[#F8F5EF] text-xs focus:outline-none focus:border-[#C66A4E]"
              />
            </div>

            <select
              value={previewType}
              onChange={(e) => setPreviewType(e.target.value as any)}
              className="px-2.5 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-xs text-[#302721]"
            >
              <option value="ticks">ticks.csv</option>
              <option value="events">events.csv</option>
            </select>
          </div>
        </div>

        {previewData && (
          <div className="overflow-x-auto border border-[#EFE7DB] rounded-lg">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-[#F8F5EF] text-[#81786E]">
                <tr>
                  {previewData.columns.slice(0, 12).map((col) => (
                    <th key={col} className="py-2.5 px-3 font-semibold border-b border-[#EFE7DB] whitespace-nowrap">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredRows.map((row, idx) => (
                  <tr key={idx} className="border-b border-[#EFE7DB] hover:bg-[#F8F5EF] transition font-mono text-[11px]">
                    {previewData.columns.slice(0, 12).map((col) => (
                      <td key={col} className="py-2 px-3 whitespace-nowrap truncate max-w-[120px]">
                        {String(row[col] ?? '')}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* 4. Preprocessing & Sequence Construction Methodology */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Preprocessing Pipeline: Raw Ticks to Model Inputs
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-2">
            <span className="font-bold text-[#C66A4E] block">1. Feature Normalization ($D=23$)</span>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              Continuous spatial attributes ($X, Y, Z$ coordinates, velocities, camera pitch/yaw, mouse delta angles) are scaled via zero-leakage standard z-score normalization computed strictly on the training partition.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-2">
            <span className="font-bold text-[#C66A4E] block">2. Temporal Windowing ($T=16$)</span>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              A temporal sliding window of length $T=16$ frames (stride $S=2$) is extracted chronologically. The input tensor to the Transformer is shaped as <code>(Batch, 16, 23)</code>.
            </p>
          </div>

          <div className="p-4 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-2">
            <span className="font-bold text-[#C66A4E] block">3. Next-Action Target ($t+1$)</span>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              The supervisory prediction target is the discrete action occurring at time step $t+1$ immediately following the 16th frame, mapped to one of the 9 project-defined classes.
            </p>
          </div>
        </div>
      </div>

      {/* 5. The 9 Project-Defined Target Classes Mapping */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-[#302721] flex items-center">
            <PieIcon className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Project Target Classes & Empirical Class Distribution
          </h2>
          <span className="text-xs text-[#81786E]">Imbalance Measured from Telemetry</span>
        </div>

        <p className="text-xs text-[#81786E]">
          *Important Taxonomy Note: These 9 target classes are project-defined tactical labels synthesized from raw keyboard/mouse inputs and game events.
        </p>

        {/* Bar Chart of Class Distribution */}
        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
              <XAxis dataKey="name" stroke="#81786E" fontSize={11} interval={0} angle={-25} textAnchor="end" />
              <YAxis stroke="#81786E" fontSize={11} />
              <Tooltip
                contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#EFE7DB', borderRadius: '8px', fontSize: '12px' }}
                formatter={(value: any, _, item: any) => [`${value} samples (${item.payload.percentage}%)`, 'Count']}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {chartData.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={WARM_COLORS[index % WARM_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Mapping & Exclusion Policy Table */}
        <div className="border-t border-[#EFE7DB] pt-4 text-xs space-y-2">
          <span className="font-bold text-[#302721] block">Documented Mapping & Exclusion Policy:</span>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px]">
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Fire:</strong> <code>in_attack == 1</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Reload:</strong> <code>in_reload == 1</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Switch Weapon:</strong> <code>slot1..3 change</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Inspect:</strong> <code>inspect_weapon</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Scope:</strong> <code>is_scoped == 1</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Grenade:</strong> <code>throw_grenade</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Jump:</strong> <code>in_jump == 1</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Crouch:</strong> <code>in_duck == 1</code>
            </div>
            <div className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
              <strong>Move:</strong> <code>W, A, S, D</code> or idle
            </div>
          </div>
          <p className="text-[11px] text-[#81786E] leading-relaxed pt-1">
            <strong>Deliberate Exclusions:</strong> Non-action game events (e.g., <code>player_death</code>, <code>bomb_planted</code>, <code>chat</code>) are filtered out so that only physical, intentional player inputs are forecasted.
          </p>
        </div>
      </div>
    </div>
  );
};
