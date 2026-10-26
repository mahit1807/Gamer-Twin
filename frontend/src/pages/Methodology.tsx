import React, { useEffect, useState } from 'react';
import {
  BookOpen,
  Cpu,
  Database,
  Layers,
  ArrowRight,
  Shield,
  HelpCircle,
  ExternalLink,
  Code2,
  GitBranch,
  Sparkles
} from 'lucide-react';
import { api } from '../services/api';

export const Methodology: React.FC = () => {
  const [data, setData] = useState<any | null>(null);

  useEffect(() => {
    api.getMethodologyInfo()
      .then((res) => setData(res))
      .catch((err) => console.error(err));
  }, []);

  return (
    <div className="max-w-5xl mx-auto space-y-8 pb-12">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
          <BookOpen className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
          Research Methodology & Formal Architecture
        </h1>
        <p className="text-sm text-[#81786E] mt-1">
          Symbiosis Institute of Technology, Pune — Department of Artificial Intelligence and Machine Learning.
        </p>
      </div>

      {/* Abstract & Scientific Objective */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-sm space-y-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-[#C66A4E]">
          Project Objective & Problem Formulation
        </span>
        <h2 className="text-lg font-bold text-[#302721]">
          Sequential Next-Action Prediction in Competitive First-Person Gameplay
        </h2>
        <p className="text-xs text-[#302721] leading-relaxed">
          In competitive interactive games like Counter-Strike, player decisions unfold over micro-second horizons conditioned on continuous spatial telemetry (positions, velocities, view angles) and discrete input histories.
          <strong> GAMER-TWIN</strong> formulates player modeling as a self-attention sequential prediction task: given a chronological window of past telemetry vectors <span className="font-mono text-[#C66A4E]">{"X = (x_1, ..., x_T) ∈ R^(T×D)"}</span> (where T=16, D=23), forecast the categorical action distribution <span className="font-mono text-[#C66A4E]">{"ŷ_{T+1} ∈ Δ^9"}</span> across 9 project target classes.
        </p>
      </div>

      {/* Visual System Architecture Diagram */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-6 shadow-sm space-y-4">
        <h2 className="text-sm font-bold text-[#302721] flex items-center">
          <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Visual Architecture & Information Flow
        </h2>

        <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-3">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 text-center text-xs">
            <div className="p-3 bg-white rounded-lg border border-[#EFE7DB] shadow-sm">
              <span className="font-bold text-[#C66A4E] block mb-1">Input Stream</span>
              <span className="font-mono text-[11px] text-[#302721]">Raw EgoCS-400K</span>
              <span className="text-[10px] text-[#81786E] block mt-1">32Hz/64Hz Ticks + Keys</span>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#EFE7DB] shadow-sm">
              <span className="font-bold text-[#C66A4E] block mb-1">Feature Norm</span>
              <span className="font-mono text-[11px] text-[#302721]">D = 23 Features</span>
              <span className="text-[10px] text-[#81786E] block mt-1">Train-Split Z-Score</span>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#EFE7DB] shadow-sm">
              <span className="font-bold text-[#C66A4E] block mb-1">Windowing</span>
              <span className="font-mono text-[11px] text-[#302721]">T = 16 Frames</span>
              <span className="text-[10px] text-[#81786E] block mt-1">Stride = 2, Offset = 1</span>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#C66A4E] shadow-sm ring-1 ring-[#C66A4E]/30">
              <span className="font-bold text-[#C66A4E] block mb-1">Transformer</span>
              <span className="font-mono text-[11px] text-[#302721]">Encoder Stack</span>
              <span className="text-[10px] text-[#81786E] block mt-1">Multi-Head Attention</span>
            </div>

            <div className="p-3 bg-white rounded-lg border border-[#EFE7DB] shadow-sm">
              <span className="font-bold text-[#82917A] block mb-1">Inference Head</span>
              <span className="font-mono text-[11px] text-[#302721]">Softmax ŷ ∈ R^9</span>
              <span className="text-[10px] text-[#81786E] block mt-1">Ranked Probabilities</span>
            </div>
          </div>
        </div>
      </div>

      {/* 23 Feature Dimensions & 9 Target Classes */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Features */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h3 className="text-sm font-bold text-[#302721] flex items-center">
            <Database className="w-4 h-4 mr-2 text-[#C66A4E]" />
            23 Telemetry Feature Dimensions ($D=23$)
          </h3>
          <div className="space-y-2 text-xs max-h-72 overflow-y-auto pr-1">
            {data?.feature_dimensions.features.map((feat: any, idx: number) => (
              <div key={idx} className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                <div className="flex justify-between font-mono font-semibold text-[#302721]">
                  <span>{feat.name}</span>
                  <span className="text-[10px] text-[#81786E] font-sans font-normal">{feat.type}</span>
                </div>
                <p className="text-[11px] text-[#81786E] mt-0.5">{feat.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* 9 Target Classes Mapping Rules */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
          <h3 className="text-sm font-bold text-[#302721] flex items-center">
            <Cpu className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Project Target Classes & Mapping Rules
          </h3>
          <p className="text-xs text-[#81786E]">
            Derived directly from synchronized keyboard/mouse controls and telemetry state:
          </p>
          <div className="grid grid-cols-3 gap-2 text-xs">
            {[
              { cls: 'Fire', map: 'in_attack == 1 or primary_attack' },
              { cls: 'Reload', map: 'in_reload == 1' },
              { cls: 'Switch Weapon', map: 'slot1..3 or weapon change' },
              { cls: 'Inspect', map: 'inspect_weapon action' },
              { cls: 'Scope', map: 'is_scoped == 1 or in_attack2' },
              { cls: 'Grenade', map: 'throw_grenade event' },
              { cls: 'Jump', map: 'in_jump == 1' },
              { cls: 'Crouch', map: 'in_duck == 1' },
              { cls: 'Move', map: 'W, A, S, D or resting idle' },
            ].map((item) => (
              <div key={item.cls} className="p-2 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                <span className="font-bold text-[#302721] block">{item.cls}</span>
                <span className="text-[9px] text-[#81786E] font-mono block mt-0.5">{item.map}</span>
              </div>
            ))}
          </div>
          <div className="p-2.5 rounded bg-[#F8F5EF] text-[11px] text-[#81786E]">
            <strong>Exclusion Policy:</strong> Match-level events without direct physical input (such as <code>player_death</code>, <code>bomb_planted</code>, <code>chat</code>) are deliberately excluded to prevent non-tactical noise.
          </div>
        </div>
      </div>

      {/* The 4 Deep Learning Architectures */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-[#302721] flex items-center">
          <Cpu className="w-4 h-4 mr-2 text-[#C66A4E]" />
          Detailed Comparison of Evaluated Architectures
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          {data?.architectures.map((arch: any) => (
            <div key={arch.id} className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-2">
              <span className="font-bold text-sm text-[#302721] block">{arch.name}</span>
              <span className="text-[10px] font-semibold uppercase text-[#C66A4E] block">{arch.role}</span>
              <p className="text-xs text-[#81786E] leading-relaxed">{arch.description}</p>
              <div className="p-2 rounded bg-white border border-[#EFE7DB] font-mono text-[10px] text-[#302721] truncate">
                {arch.formulation}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Reinforcement Learning Distinction & DQN Future Work Scope */}
      <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-3">
        <div className="flex items-center space-x-2">
          <Shield className="w-4 h-4 text-[#C66A4E]" />
          <h3 className="text-sm font-bold text-[#302721]">
            Scope Boundary: Behavioral Prediction vs. Reinforcement Learning (DQN)
          </h3>
        </div>
        <p className="text-xs text-[#302721] leading-relaxed">
          A common faculty question is whether GAMER-TWIN plays the game or controls player movement via Reinforcement Learning:
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
          <div className="p-3.5 rounded-lg bg-[#82917A]/10 border border-[#82917A]/30">
            <span className="font-bold text-[#82917A] block mb-1">Current GAMER-TWIN Scope</span>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              Supervised & Self-Supervised Sequence Prediction. The model is an offline analytical Digital Twin that models player behavior and predicts probabilities of immediate actions based on historical observation. It does not issue live commands to Counter-Strike.
            </p>
          </div>
          <div className="p-3.5 rounded-lg bg-[#C66A4E]/10 border border-[#C66A4E]/30">
            <span className="font-bold text-[#C66A4E] block mb-1">Deep Q-Networks (DQN) Scope</span>
            <p className="text-[11px] text-[#302721] leading-relaxed">
              <strong>Future Work Only.</strong> Autonomous reward-maximizing game-playing agents via Deep Q-Learning or PPO are planned for subsequent phases when an active execution sandbox is connected. No DQN agent is required or asserted in the current implementation.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
