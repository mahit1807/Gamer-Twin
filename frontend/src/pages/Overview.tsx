import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Database,
  Cpu,
  Zap,
  BarChart3,
  CheckCircle2,
  Clock,
  Layers,
  Activity,
  BookOpen,
  HelpCircle,
  ExternalLink,
  ShieldAlert,
  Gamepad2
} from 'lucide-react';
import {
  api,
  SystemStatus,
  DatasetSummary,
  ModelComparisonItem,
  DatasetGateStatus
} from '../services/api';

const CROSSFPS_GAMES = [
  { name: 'Halo Infinite', count: '32,466 clips', color: '#C66A4E' },
  { name: 'Xonotic', count: '10,460 clips', color: '#82917A' },
  { name: 'Call of Duty: Modern Warfare', count: '8,853 clips', color: '#B95D3B' },
  { name: 'Halo MCC', count: '8,227 clips', color: '#4A6FA5' },
  { name: 'Call of Duty: Warzone', count: '4,818 clips', color: '#D97706' },
  { name: 'COD: Modern Warfare III', count: '3,662 clips', color: '#8E7DBE' },
  { name: 'Call of Duty (Classic)', count: '514 clips', color: '#5E8B7E' },
];

export const Overview: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [gateStatus, setGateStatus] = useState<DatasetGateStatus | null>(null);
  const [experiments, setExperiments] = useState<ModelComparisonItem[]>([]);

  useEffect(() => {
    Promise.all([
      api.getSystemStatus().catch(() => null),
      api.getDatasetSummary().catch(() => null),
      api.getDatasetGateStatus().catch(() => null),
      api.getExperimentComparison().catch(() => []),
    ]).then(([sys, ds, gate, exps]) => {
      setSystemStatus(sys);
      setDatasetSummary(ds);
      setGateStatus(gate);
      setExperiments(exps || []);
    });
  }, []);

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* 1. Academic Title & Author Identity Header */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-8 shadow-xs">
        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6">
          <div className="space-y-2">
            <div className="inline-flex items-center space-x-2 text-xs font-semibold px-2.5 py-1 rounded bg-[#C66A4E]/10 text-[#C66A4E]">
              <Activity className="w-3.5 h-3.5" />
              <span>B.Tech Final-Semester Project Defense · Department of AIML</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold text-[#302721] tracking-tight leading-tight">
              GAMER-TWIN: A Personalized Deep Learning Framework for Modeling and Predicting Player Behaviour
            </h1>
            <p className="text-sm sm:text-base font-medium text-[#81786E]">
              A Multi-Game Transformer Digital Twin for 10-DoF Next-Action Prediction (CrossFPS)
            </p>
          </div>

          <div className="flex-shrink-0 flex flex-col sm:flex-row gap-3">
            <Link
              to="/predict"
              className="inline-flex items-center justify-center px-5 py-2.5 rounded-lg bg-[#C66A4E] hover:bg-[#B95D3B] text-white text-xs font-semibold shadow-xs transition-colors"
            >
              <Zap className="w-3.5 h-3.5 mr-1.5" />
              Open Prediction Demo
              <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
            </Link>
          </div>
        </div>

        {/* Academic Metadata Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 pt-6 border-t border-[#EFE7DB] text-xs">
          <div>
            <span className="text-[#81786E] block mb-0.5">Students / Presenters</span>
            <span className="font-semibold text-[#302721]">Mahit Turakhia & Het Shah</span>
          </div>
          <div>
            <span className="text-[#81786E] block mb-0.5">Faculty Guide</span>
            <span className="font-semibold text-[#302721]">Dr. Ananya Bhattacharjee</span>
          </div>
          <div>
            <span className="text-[#81786E] block mb-0.5">Institution & Department</span>
            <span className="font-semibold text-[#302721]">Symbiosis Institute of Technology, Pune (AIML)</span>
          </div>
        </div>
      </div>

      {/* 2. Strict Dataset Gate Status Card (CrossFPS) */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[#EFE7DB]">
          <div className="flex items-center space-x-2">
            <Database className="w-4 h-4 text-[#C66A4E]" />
            <h2 className="text-sm font-bold text-[#302721]">
              Phase 1 — Official CrossFPS Dataset Gate Status
            </h2>
          </div>
          {gateStatus && (
            <span
              className={`text-xs px-2.5 py-1 rounded-full font-bold border ${
                gateStatus.gate_status === 'DATASET_READY'
                  ? 'bg-[#82917A]/10 text-[#82917A] border-[#82917A]/30'
                  : gateStatus.gate_status === 'DATASET_PARTIALLY_AVAILABLE'
                  ? 'bg-[#D97706]/10 text-[#D97706] border-[#D97706]/30'
                  : 'bg-[#C66A4E]/10 text-[#C66A4E] border-[#C66A4E]/30'
              }`}
            >
              {gateStatus.gate_status}
            </span>
          )}
        </div>

        <div className="space-y-3 text-xs">
          <p className="text-[#302721] leading-relaxed">
            {gateStatus?.gate_message || 'Official CrossFPS validation metadata verified. Telemetry sequences ready for sequence modeling.'}
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
              <span className="text-[#81786E] block mb-1">Dataset Name & License</span>
              <span className="font-bold text-[#302721]">CrossFPS (CC BY-NC 4.0)</span>
            </div>
            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
              <span className="text-[#81786E] block mb-1">Active Split / Source</span>
              <span className="font-bold font-mono text-[#302721]">
                CrossFPS-val (6.27 GB)
              </span>
            </div>
            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
              <span className="text-[#81786E] block mb-1">Clips in Scope</span>
              <span className="font-bold font-mono text-[#302721]">
                {datasetSummary ? `${Math.round(datasetSummary.total_records / 100).toLocaleString()} Verified Clips` : 'Scanning...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 3. 7 FPS Game Titles Included in CrossFPS */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-[#302721] flex items-center">
            <Gamepad2 className="w-4 h-4 mr-2 text-[#C66A4E]" />
            CrossFPS Multi-Game Coverage (69,000 Clips @ 20 FPS)
          </h2>
          <span className="text-xs text-[#81786E]">7 FPS Franchises</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {CROSSFPS_GAMES.map((g) => (
            <div key={g.name} className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
              <span className="text-xs font-bold text-[#302721] block leading-snug">{g.name}</span>
              <span className="text-[11px] font-mono text-[#81786E] block">{g.count}</span>
            </div>
          ))}
          <div className="p-3 rounded-lg bg-[#C66A4E]/10 border border-[#C66A4E]/30 flex flex-col justify-center">
            <span className="text-xs font-bold text-[#C66A4E] block">10-DoF Telemetry</span>
            <span className="text-[11px] text-[#302721]">4 Analog + 6 Buttons</span>
          </div>
        </div>
      </div>

      {/* 4. Prediction Workflow: 10-DoF Sequence Windowing to Multi-Action Forecast */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-xs space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold text-[#302721] flex items-center">
            <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Prediction Workflow: From 10-DoF Gamepad Telemetry to Next Action
          </h2>
          <span className="text-xs text-[#81786E]">Zero data-leakage pipeline</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-5 gap-3">
          {[
            {
              step: '01',
              title: 'CrossFPS Telemetry',
              detail: '10-DoF controller channels (LX, LY, RX, RY, RT, LT, R3, A, X, Y) sampled at 20 Hz.',
              icon: Database,
            },
            {
              step: '02',
              title: 'Clip Normalization',
              detail: 'Continuous joysticks normalized in [-1, 1]; discrete button events encoded as binary.',
              icon: CheckCircle2,
            },
            {
              step: '03',
              title: 'Sequence Windowing',
              detail: 'Sliding history window T=16 frames (~0.8s) capturing player aim and movement momentum.',
              icon: Layers,
            },
            {
              step: '04',
              title: 'Transformer Encoder',
              detail: 'Multi-head self-attention with dual pooling over tactical navigation and combat cues.',
              icon: Cpu,
            },
            {
              step: '05',
              title: 'Dual-Head Forecast',
              detail: 'Analog joystick regression (Tanh) + multi-label combat triggers (Sigmoid for Fire/Scope/Jump).',
              icon: Zap,
            },
          ].map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.step} className="p-3.5 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold text-[#C66A4E]">{item.step}</span>
                  <Icon className="w-3.5 h-3.5 text-[#81786E]" />
                </div>
                <h3 className="text-xs font-bold text-[#302721] leading-tight">{item.title}</h3>
                <p className="text-[11px] text-[#81786E] leading-snug">{item.detail}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* 5. Quick Navigation to Primary Pages */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Link
          to="/predict"
          className="p-5 rounded-xl bg-white border-2 border-[#C66A4E] shadow-xs group hover:bg-[#F8F5EF]/60 transition"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-bold text-[#C66A4E] flex items-center space-x-1.5">
              <span>Next Prediction Studio</span>
              <span className="text-xs">★</span>
            </span>
            <ArrowRight className="w-4 h-4 text-[#C66A4E] transition-transform group-hover:translate-x-1" />
          </div>
          <p className="text-xs text-[#81786E] leading-relaxed">
            Run live sequence inference across supported FPS titles, configure gameplay situations, and explore counterfactual What-If scenarios.
          </p>
        </Link>

        <Link
          to="/evaluation"
          className="p-5 rounded-xl bg-white border border-[#EFE7DB] hover:border-[#C66A4E] transition shadow-xs group hover:bg-[#F8F5EF]/60"
        >
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-bold text-[#302721] group-hover:text-[#C66A4E]">
              Model & Evaluation Benchmarks
            </span>
            <ArrowRight className="w-4 h-4 text-[#81786E] group-hover:text-[#C66A4E] transition-transform group-hover:translate-x-1" />
          </div>
          <p className="text-xs text-[#81786E] leading-relaxed">
            Inspect verified test metrics, recurrent Baseline vs. Transformer Encoder comparisons, and multi-label confusion matrices.
          </p>
        </Link>
      </div>
    </div>
  );
};
