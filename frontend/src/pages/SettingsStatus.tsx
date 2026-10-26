import React, { useEffect, useState } from 'react';
import {
  Settings,
  Server,
  Database,
  Cpu,
  CheckCircle2,
  XCircle,
  FolderOpen,
  Terminal,
  ShieldCheck,
  RefreshCw
} from 'lucide-react';
import { api, SystemStatus } from '../services/api';

export const SettingsStatus: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadStatus = () => {
    setLoading(true);
    api.getSystemStatus()
      .then((data) => {
        setStatus(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to reach backend.');
        setLoading(false);
      });
  };

  useEffect(() => {
    loadStatus();
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
            <Settings className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
            System & Environment Diagnostics
          </h1>
          <p className="text-sm text-[#81786E] mt-1">
            Real-time status of backend microservices, PyTorch tensor engine, dataset paths, and checkpoints.
          </p>
        </div>

        <button
          onClick={loadStatus}
          className="inline-flex items-center px-3 py-1.5 rounded-lg bg-white border border-[#EFE7DB] text-xs font-semibold text-[#302721] hover:bg-[#F8F5EF] shadow-sm transition"
        >
          <RefreshCw className={`w-3.5 h-3.5 mr-1.5 text-[#C66A4E] ${loading ? 'animate-spin' : ''}`} />
          Check Health
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700 flex items-center">
          <XCircle className="w-4 h-4 mr-2 text-red-500" />
          {error}
        </div>
      )}

      {/* System Status Table */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
        <h2 className="text-sm font-semibold text-[#302721] flex items-center">
          <Server className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Backend API & Compute Status
        </h2>

        <div className="space-y-2 text-xs">
          <div className="flex justify-between py-2 border-b border-[#EFE7DB]">
            <span className="text-[#81786E]">API Server Status</span>
            <span className="font-semibold text-[#82917A] flex items-center">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Online (FastAPI)
            </span>
          </div>

          <div className="flex justify-between py-2 border-b border-[#EFE7DB]">
            <span className="text-[#81786E]">PyTorch Device Engine</span>
            <span className="font-mono font-semibold text-[#302721]">
              {status?.pytorch_device.toUpperCase() || 'CPU'}
            </span>
          </div>

          <div className="flex justify-between py-2 border-b border-[#EFE7DB]">
            <span className="text-[#81786E]">Official EgoCS-400K Repo</span>
            <span className="font-semibold text-[#302721] flex items-center">
              {status?.dataset_repo_cloned ? (
                <span className="text-[#82917A] flex items-center">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Cloned (/Dataset)
                </span>
              ) : (
                <span className="text-[#C66A4E] flex items-center">
                  <XCircle className="w-3.5 h-3.5 mr-1" /> Not Found
                </span>
              )}
            </span>
          </div>

          <div className="flex justify-between py-2 border-b border-[#EFE7DB]">
            <span className="text-[#81786E]">Active Telemetry Source</span>
            <span className="font-medium text-[#302721]">
              {status?.active_dataset_source || 'Checking...'}
            </span>
          </div>

          <div className="flex justify-between py-2 border-b border-[#EFE7DB]">
            <span className="text-[#81786E]">Saved Model Checkpoints</span>
            <span className="font-mono font-semibold text-[#82917A]">
              {status?.trained_checkpoints.length} Checkpoints Available
            </span>
          </div>

          <div className="flex justify-between py-2">
            <span className="text-[#81786E]">Application Version</span>
            <span className="font-mono text-[#302721]">v{status?.api_version || '1.0.0'}</span>
          </div>
        </div>
      </div>

      {/* Checkpoints Registry */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-[#302721] flex items-center">
          <Cpu className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Model Checkpoints Directory (outputs/checkpoints)
        </h2>
        <div className="space-y-1.5">
          {status?.trained_checkpoints.map((ckpt) => (
            <div
              key={ckpt}
              className="px-3 py-2 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] flex items-center justify-between text-xs font-mono"
            >
              <span className="text-[#302721]">{ckpt}</span>
              <span className="text-[#82917A] font-semibold text-[10px] uppercase">Ready for Inference</span>
            </div>
          ))}
        </div>
      </div>

      {/* Quick Setup Commands Guide */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
        <h2 className="text-sm font-semibold text-[#302721] flex items-center">
          <Terminal className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Production Execution Commands
        </h2>
        <div className="space-y-2 text-xs font-mono text-[#302721]">
          <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
            <span className="text-[#81786E] block mb-1 font-sans text-[11px]"># Start FastAPI Backend</span>
            <code>uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload</code>
          </div>
          <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
            <span className="text-[#81786E] block mb-1 font-sans text-[11px]"># Start React Vite Frontend</span>
            <code>cd frontend && npm run dev</code>
          </div>
        </div>
      </div>
    </div>
  );
};
