import React, { useEffect, useState } from 'react';
import {
  Fingerprint,
  Activity,
  Layers,
  ArrowRight,
  Shield,
  Compass,
  Zap,
  Info,
  Clock,
  RefreshCw,
  GitCommit
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  PieChart,
  Pie,
  LineChart,
  Line,
  CartesianGrid
} from 'recharts';
import { api, PlayerSignature } from '../services/api';

const SIGNATURE_COLORS = ['#C66A4E', '#B95D3B', '#82917A', '#302721', '#81786E', '#D97706'];

export const PlayerSignaturePage: React.FC = () => {
  const [signature, setSignature] = useState<PlayerSignature | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadSignature = () => {
    setLoading(true);
    api.getPlayerSignature()
      .then((data) => {
        setSignature(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to compute player signature.');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadSignature();
  }, []);

  const actionDistData = signature
    ? Object.entries(signature.action_frequencies).map(([name, count]) => ({
        name,
        count,
        proportion: (signature.action_proportions[name] * 100).toFixed(1),
      }))
    : [];

  const ratioPieData = signature
    ? [
        { name: 'Movement Actions', value: signature.movement_vs_combat_ratio.movement_percentage, color: '#82917A' },
        { name: 'Combat / Weapons', value: signature.movement_vs_combat_ratio.combat_percentage, color: '#C66A4E' },
        { name: 'Utility / Other', value: signature.movement_vs_combat_ratio.utility_percentage, color: '#81786E' },
      ]
    : [];

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded bg-[#C66A4E]/10 text-[#C66A4E] text-xs font-semibold mb-1">
            <Fingerprint className="w-3.5 h-3.5" />
            <span>Novel Research Contribution</span>
          </div>
          <h1 className="text-2xl font-bold text-[#302721] tracking-tight">
            Player Behaviour Signature
          </h1>
          <p className="text-sm text-[#81786E] mt-1">
            Empirical behavioral pattern fingerprint computed from chronological gameplay action transitions and entropy shifts.
          </p>
        </div>

        <button
          onClick={loadSignature}
          className="inline-flex items-center px-3.5 py-1.5 rounded-lg bg-white border border-[#EFE7DB] text-xs font-semibold text-[#302721] hover:bg-[#F8F5EF] shadow-sm transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 mr-1.5 text-[#C66A4E] ${loading ? 'animate-spin' : ''}`} />
          Recalculate Signature
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700">
          {error}
        </div>
      )}

      {/* Signature Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-4 shadow-sm">
          <span className="text-[10px] uppercase font-bold text-[#81786E] tracking-wider block mb-1">
            Analyzed Player
          </span>
          <span className="text-base font-bold text-[#302721] font-mono">
            {signature?.player_id || 'Player_01'}
          </span>
          <span className="text-[10px] text-[#81786E] block mt-1">
            Source: {signature?.data_source}
          </span>
        </div>

        <div className="bg-white border border-[#EFE7DB] rounded-xl p-4 shadow-sm">
          <span className="text-[10px] uppercase font-bold text-[#81786E] tracking-wider block mb-1">
            Total Observed Events
          </span>
          <span className="text-base font-bold font-mono text-[#302721]">
            {signature ? signature.total_observed_events.toLocaleString() : '—'}
          </span>
          <span className="text-[10px] text-[#81786E] block mt-1">
            Synchronized tick events
          </span>
        </div>

        <div className="bg-white border border-[#EFE7DB] rounded-xl p-4 shadow-sm">
          <span className="text-[10px] uppercase font-bold text-[#81786E] tracking-wider block mb-1">
            Movement vs Combat
          </span>
          <span className="text-base font-bold font-mono text-[#82917A]">
            {signature?.movement_vs_combat_ratio.movement_percentage}% / {signature?.movement_vs_combat_ratio.combat_percentage}%
          </span>
          <span className="text-[10px] text-[#81786E] block mt-1">
            Macro tactical allocation
          </span>
        </div>

        <div className="bg-white border border-[#EFE7DB] rounded-xl p-4 shadow-sm">
          <span className="text-[10px] uppercase font-bold text-[#81786E] tracking-wider block mb-1">
            Top Action Transition
          </span>
          <span className="text-xs font-bold text-[#C66A4E] block truncate">
            {signature?.top_transitions[0]
              ? `${signature.top_transitions[0].from_action} → ${signature.top_transitions[0].to_action}`
              : 'Move → Move'}
          </span>
          <span className="text-[10px] text-[#81786E] block mt-1 font-mono">
            Prob: {signature?.top_transitions[0]?.probability}
          </span>
        </div>
      </div>

      {/* Macro Ratio & Action Frequencies */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Tactical Event Allocation Pie */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-[#302721] flex items-center">
            <Compass className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Tactical Action Allocation
          </h2>
          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={ratioPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {ratioPieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#EFE7DB', borderRadius: '8px', fontSize: '12px' }}
                  formatter={(val: any) => [`${val}%`, 'Proportion']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-1 text-xs">
            {ratioPieData.map((r) => (
              <div key={r.name} className="flex justify-between items-center">
                <span className="flex items-center text-[#81786E]">
                  <span className="w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: r.color }}></span>
                  {r.name}
                </span>
                <span className="font-mono font-semibold text-[#302721]">{r.value}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* Empirical Action Frequency Bar */}
        <div className="md:col-span-2 bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-[#302721] flex items-center">
            <Activity className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Action Distribution Across Replay
          </h2>
          <div className="h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={actionDistData} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <XAxis dataKey="name" stroke="#81786E" fontSize={10} interval={0} angle={-25} textAnchor="end" />
                <YAxis stroke="#81786E" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#EFE7DB', borderRadius: '8px', fontSize: '11px' }}
                  formatter={(val: any, _, item: any) => [`${val} samples (${item.payload.proportion}%)`, 'Count']}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {actionDistData.map((_, i) => (
                    <Cell key={i} fill={SIGNATURE_COLORS[i % SIGNATURE_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Transition Matrix & N-Grams */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Empirical 1st-Order Action Transitions */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-[#302721] flex items-center">
            <GitCommit className="w-4 h-4 mr-2 text-[#C66A4E]" />
            First-Order Action Transitions (Markov Conditional)
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-[#EFE7DB] bg-[#F8F5EF] text-[#81786E]">
                  <th className="py-2 px-3 font-semibold">From Action</th>
                  <th className="py-2 px-3 font-semibold">To Action</th>
                  <th className="py-2 px-3 font-semibold text-right">Count</th>
                  <th className="py-2 px-3 font-semibold text-right">Transition Prob</th>
                </tr>
              </thead>
              <tbody>
                {signature?.top_transitions.slice(0, 8).map((t, idx) => (
                  <tr key={idx} className="border-b border-[#EFE7DB] hover:bg-[#F8F5EF] transition font-mono">
                    <td className="py-2 px-3 font-semibold text-[#302721] font-sans">{t.from_action}</td>
                    <td className="py-2 px-3 font-semibold text-[#C66A4E] font-sans">→ {t.to_action}</td>
                    <td className="py-2 px-3 text-right">{t.count}</td>
                    <td className="py-2 px-3 text-right font-bold text-[#82917A]">
                      {(t.probability * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Common Micro-Action Sequences (N-Grams) */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h2 className="text-sm font-semibold text-[#302721] flex items-center">
            <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Common Action Sequences (Observed Micro-Tactics)
          </h2>
          <div className="space-y-2">
            {signature?.common_ngrams.map((ng, i) => (
              <div key={i} className="p-2.5 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] flex items-center justify-between text-xs">
                <div className="flex items-center space-x-2">
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-white border border-[#EFE7DB] text-[#81786E]">
                    {ng.ngram_type}
                  </span>
                  <span className="font-semibold text-[#302721]">{ng.sequence}</span>
                </div>
                <div className="text-right">
                  <span className="font-mono font-semibold text-[#C66A4E]">{ng.count}x</span>
                  <span className="text-[10px] text-[#81786E] ml-1">({ng.frequency_pct}%)</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Temporal Shifts Across Quarters */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-[#302721] flex items-center">
          <Clock className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Temporal Action Diversity & Combat Intensity (Replay Quarters)
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
          {signature?.window_entropy.map((w) => (
            <div key={w.window_quarter} className="p-3.5 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
              <span className="font-bold text-[#302721] block">{w.window_quarter}</span>
              <div className="flex justify-between text-[#81786E]">
                <span>Entropy:</span>
                <span className="font-mono font-semibold text-[#302721]">{w.entropy} bits</span>
              </div>
              <div className="flex justify-between text-[#81786E]">
                <span>Dominant:</span>
                <span className="font-medium text-[#C66A4E]">{w.dominant_action}</span>
              </div>
              <div className="flex justify-between text-[#81786E]">
                <span>Combat Rate:</span>
                <span className="font-mono font-semibold text-[#82917A]">{w.combat_intensity}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Academic Disclosure */}
      <div className="p-4 rounded-xl bg-white border border-[#EFE7DB] text-xs text-[#81786E] leading-relaxed flex items-start space-x-2">
        <Info className="w-4 h-4 text-[#C66A4E] flex-shrink-0 mt-0.5" />
        <div>
          <strong>Scientific Methodology Notice:</strong> The Player Behaviour Signature is a descriptive statistical summary of recorded telemetry frequencies and empirical Markov transition probabilities. It is strictly not a psychological profile, personality assessment, player skill rating, or prediction of mental intention.
        </div>
      </div>
    </div>
  );
};
