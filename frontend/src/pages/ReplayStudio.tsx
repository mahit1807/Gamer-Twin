import React, { useEffect, useState } from 'react';
import {
  PlaySquare,
  ChevronLeft,
  ChevronRight,
  Play,
  Pause,
  RotateCcw,
  Crosshair,
  Shield,
  Activity,
  Layers,
  Zap,
  CheckCircle2,
  XCircle,
  Eye,
  Sliders,
  Compass
} from 'lucide-react';
import { api, ReplayTelemetryStep } from '../services/api';

const ACTION_COLORS: Record<string, string> = {
  Fire: '#C66A4E',
  Reload: '#B95D3B',
  'Switch Weapon': '#D97706',
  Inspect: '#81786E',
  Scope: '#059669',
  Grenade: '#DC2626',
  Jump: '#82917A',
  Crouch: '#4B5563',
  Move: '#302721',
};

export const ReplayStudio: React.FC = () => {
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [totalSteps, setTotalSteps] = useState<number>(100);
  const [telemetry, setTelemetry] = useState<ReplayTelemetryStep | null>(null);
  const [timelineEvents, setTimelineEvents] = useState<any[]>([]);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<string>('transformer');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Load replay info and timeline
  useEffect(() => {
    api.getReplayInfo()
      .then((info) => {
        setTotalSteps(info.total_steps || 100);
      })
      .catch((err) => console.error(err));

    api.getReplayTimeline(60)
      .then((res) => {
        setTimelineEvents(res.events || []);
      })
      .catch((err) => console.error(err));
  }, []);

  // Fetch telemetry whenever currentStep or selectedModel changes
  useEffect(() => {
    setLoading(true);
    api.getReplayStep(currentStep, selectedModel)
      .then((step) => {
        setTelemetry(step);
        setTotalSteps(step.total_steps);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message || 'Failed to load telemetry step.');
        setLoading(false);
      });
  }, [currentStep, selectedModel]);

  // Auto-play timer
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps - 1) {
            setIsPlaying(false);
            return 0;
          }
          return prev + 1;
        });
      }, 350);
    }
    return () => clearInterval(timer);
  }, [isPlaying, totalSteps]);

  const trace = telemetry?.prediction_trace;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
            <PlaySquare className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
            Replay Studio
          </h1>
          <p className="text-sm text-[#81786E] mt-1">
            Offline chronological telemetry replay engine synchronizing player controls with digital twin forecasts.
          </p>
        </div>

        {/* Playback Controls Bar */}
        <div className="flex items-center space-x-2 bg-white border border-[#EFE7DB] rounded-xl p-1.5 shadow-sm">
          <button
            onClick={() => setCurrentStep(0)}
            title="Reset to Start"
            className="p-2 rounded-lg hover:bg-[#F8F5EF] text-[#81786E] transition"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentStep((p) => Math.max(0, p - 1))}
            disabled={currentStep === 0}
            className="p-2 rounded-lg hover:bg-[#F8F5EF] text-[#81786E] disabled:opacity-40 transition"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-4 py-1.5 rounded-lg bg-[#C66A4E] hover:bg-[#B95D3B] text-white text-xs font-semibold flex items-center transition shadow-sm"
          >
            {isPlaying ? (
              <>
                <Pause className="w-3.5 h-3.5 mr-1.5 fill-current" /> Pause
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 mr-1.5 fill-current" /> Play
              </>
            )}
          </button>
          <button
            onClick={() => setCurrentStep((p) => Math.min(totalSteps - 1, p + 1))}
            disabled={currentStep >= totalSteps - 1}
            className="p-2 rounded-lg hover:bg-[#F8F5EF] text-[#81786E] disabled:opacity-40 transition"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <div className="border-l border-[#EFE7DB] pl-3 pr-2 font-mono text-xs text-[#302721]">
            Step {currentStep + 1} / {totalSteps}
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-xs text-red-700">
          {error}
        </div>
      )}

      {/* Scrubbing Slider & Timeline Strip */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-[#81786E] uppercase tracking-wider">
            Chronological Timeline Scrubber
          </span>
          <span className="font-mono text-[#81786E]">
            Tick: {telemetry?.tick ?? currentStep} | Time: {telemetry ? `${telemetry.seconds.toFixed(2)}s` : '0.00s'}
          </span>
        </div>

        {/* Step Slider */}
        <input
          type="range"
          min="0"
          max={totalSteps - 1}
          value={currentStep}
          onChange={(e) => setCurrentStep(Number(e.target.value))}
          className="w-full accent-[#C66A4E] cursor-pointer"
        />

        {/* Action Event Pills Timeline */}
        <div className="flex space-x-1.5 overflow-x-auto pb-2">
          {timelineEvents.map((ev, i) => {
            const isCurrent = Math.abs(ev.step_idx - currentStep) <= 1;
            return (
              <button
                key={i}
                onClick={() => setCurrentStep(ev.step_idx)}
                className={`px-2.5 py-1 rounded text-[11px] font-medium whitespace-nowrap transition flex items-center space-x-1 border ${
                  isCurrent
                    ? 'bg-[#C66A4E] text-white border-[#C66A4E] shadow-sm scale-105'
                    : ev.is_combat
                    ? 'bg-[#C66A4E]/10 text-[#C66A4E] border-[#C66A4E]/20 hover:bg-[#C66A4E]/20'
                    : 'bg-[#F8F5EF] text-[#81786E] border-[#EFE7DB] hover:bg-[#EFE7DB]'
                }`}
              >
                <span>{ev.action}</span>
                <span className="text-[9px] opacity-70">({ev.seconds}s)</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Main 2-Column Replay Inspection Layout */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left Column: Player Telemetry & Controller State */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-5">
          <div className="flex items-center justify-between border-b border-[#EFE7DB] pb-3">
            <h2 className="text-sm font-bold text-[#302721] flex items-center">
              <Compass className="w-4 h-4 mr-2 text-[#C66A4E]" />
              Physical Telemetry State
            </h2>
            <span className="text-xs px-2 py-0.5 rounded bg-[#F8F5EF] border border-[#EFE7DB] font-mono text-[#81786E]">
              {telemetry?.data_source}
            </span>
          </div>

          {/* Spatial Coordinates & Velocity */}
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
              <span className="text-[#81786E] font-medium block">Coordinates (X, Y, Z)</span>
              <div className="font-mono font-semibold text-[#302721]">
                X: {telemetry?.pos_x} <br />
                Y: {telemetry?.pos_y} <br />
                Z: {telemetry?.pos_z}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
              <span className="text-[#81786E] font-medium block">Velocity Vectors</span>
              <div className="font-mono font-semibold text-[#302721]">
                Vx: {telemetry?.vel_x} <br />
                Vy: {telemetry?.vel_y} <br />
                Vz: {telemetry?.vel_z}
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
              <span className="text-[#81786E] font-medium block">Camera Orientation</span>
              <div className="font-mono font-semibold text-[#302721]">
                Pitch: {telemetry?.view_pitch}° <br />
                Yaw: {telemetry?.view_yaw}°
              </div>
            </div>

            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] space-y-1">
              <span className="text-[#81786E] font-medium block">Vitals & Weapon</span>
              <div className="font-mono font-semibold text-[#302721]">
                Weapon: {telemetry?.active_weapon.toUpperCase()} <br />
                Ammo: {telemetry?.ammo} / 30 <br />
                Health: {telemetry?.health} HP
              </div>
            </div>
          </div>

          {/* Active Keyboard & Mouse Inputs */}
          <div>
            <span className="text-xs font-semibold text-[#81786E] uppercase tracking-wider block mb-2">
              Synchronized Input Traces
            </span>
            <div className="grid grid-cols-5 gap-2">
              {telemetry &&
                Object.entries(telemetry.inputs).map(([key, val]) => (
                  <div
                    key={key}
                    className={`py-2 px-1 rounded text-center text-xs font-bold font-mono border transition ${
                      val === 1
                        ? 'bg-[#C66A4E] text-white border-[#C66A4E] shadow-sm'
                        : 'bg-[#F8F5EF] text-[#81786E] border-[#EFE7DB]'
                    }`}
                  >
                    {key}
                  </div>
                ))}
            </div>
          </div>
        </div>

        {/* Right Column: Digital Twin Next-Action Prediction */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-5">
          <div className="flex items-center justify-between border-b border-[#EFE7DB] pb-3">
            <h2 className="text-sm font-bold text-[#302721] flex items-center">
              <Zap className="w-4 h-4 mr-2 text-[#C66A4E]" />
              Digital Twin Next-Action Forecast
            </h2>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="text-xs px-2 py-1 rounded border border-[#EFE7DB] bg-white text-[#302721]"
            >
              <option value="transformer">Transformer (Primary)</option>
              <option value="transformer_ssl">Transformer SSL</option>
              <option value="lstm_transformer">LSTM + Transformer</option>
              <option value="lstm_baseline">LSTM Baseline</option>
            </select>
          </div>

          {/* Comparison Result Box */}
          {trace ? (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] grid grid-cols-2 gap-4 text-center">
                <div>
                  <span className="text-[11px] text-[#81786E] block mb-1">Forecast Action</span>
                  <span
                    className="text-xl font-extrabold block"
                    style={{ color: ACTION_COLORS[trace.predicted_next_action] || '#C66A4E' }}
                  >
                    {trace.predicted_next_action}
                  </span>
                  <span className="text-xs font-mono text-[#81786E]">
                    {(trace.confidence * 100).toFixed(1)}% Conf
                  </span>
                </div>

                <div className="border-l border-[#EFE7DB]">
                  <span className="text-[11px] text-[#81786E] block mb-1">Actual Observed Action</span>
                  <span
                    className="text-xl font-extrabold block"
                    style={{ color: ACTION_COLORS[telemetry?.actual_next_action || ''] || '#302721' }}
                  >
                    {telemetry?.actual_next_action}
                  </span>
                  <span className="text-xs font-semibold flex items-center justify-center mt-1">
                    {trace.matches_ground_truth ? (
                      <span className="text-[#82917A] flex items-center">
                        <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Matched
                      </span>
                    ) : (
                      <span className="text-[#B95D3B] flex items-center">
                        <XCircle className="w-3.5 h-3.5 mr-1" /> Diverged
                      </span>
                    )}
                  </span>
                </div>
              </div>

              {/* Recent Action Sequence Context */}
              <div>
                <span className="text-xs font-semibold text-[#81786E] uppercase tracking-wider block mb-2">
                  Input History Sequence (Past 16 Frames)
                </span>
                <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] text-xs font-mono text-[#302721] flex flex-wrap gap-1.5">
                  {telemetry?.recent_actions.map((act, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded bg-white border border-[#EFE7DB] text-[10px]"
                    >
                      {act}
                    </span>
                  ))}
                </div>
              </div>

              {/* Ranked Probabilities Quick View */}
              <div>
                <span className="text-xs font-semibold text-[#81786E] uppercase tracking-wider block mb-2">
                  Top Class Probabilities
                </span>
                <div className="space-y-1.5">
                  {trace.ranked_probabilities.slice(0, 4).map((rp) => (
                    <div key={rp.class_name} className="flex items-center text-xs">
                      <span className="w-24 text-[#81786E] font-medium">{rp.class_name}</span>
                      <div className="flex-1 bg-[#F8F5EF] h-2 rounded-full overflow-hidden mr-3">
                        <div
                          className="bg-[#C66A4E] h-full rounded-full"
                          style={{ width: `${rp.percentage}%` }}
                        ></div>
                      </div>
                      <span className="font-mono text-[#302721] w-12 text-right">{rp.percentage}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="p-8 rounded-xl bg-[#F8F5EF] border border-dashed border-[#EFE7DB] text-center text-xs text-[#81786E]">
              Select a valid model checkpoint to enable live sequence inference during replay.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
