import React, { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Zap,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Clock,
  Layers,
  Cpu,
  Sliders,
  Sparkles,
  Gamepad2,
  Crosshair,
  ArrowRight,
  Shield,
  Eye,
  Flame,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Info,
  Check,
  Compass
} from 'lucide-react';
import {
  api,
  PredictionModelInfo,
  GameInfo,
  GameSequence,
  ConditionOverrides,
  PredictResponse,
  WhatIfDetailedResponse,
  RankedProbability
} from '../services/api';

const ACTION_COLORS: Record<string, string> = {
  'Fire (RT)': '#C66A4E',
  'Scope / Aim (LT)': '#4A6FA5',
  'Melee (R3)': '#8E7DBE',
  'Jump (A)': '#82917A',
  'Reload (X)': '#B95D3B',
  'Switch Weapon (Y)': '#E5A93C',
  'Neutral Steering / Movement': '#81786E',
};

const ACTION_ICONS: Record<string, string> = {
  'Fire (RT)': '🔥',
  'Scope / Aim (LT)': '🎯',
  'Melee (R3)': '👊',
  'Jump (A)': '⏫',
  'Reload (X)': '🔄',
  'Switch Weapon (Y)': '🔁',
  'Neutral Steering / Movement': '🕹️',
};

export const PredictionStudio: React.FC = () => {
  // Step 1: Model Selection
  const [models, setModels] = useState<PredictionModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('transformer');
  const [modelsLoading, setModelsLoading] = useState(true);

  // Step 2: Game Selection
  const [games, setGames] = useState<GameInfo[]>([]);
  const [selectedGame, setSelectedGame] = useState<string>('halo_infinite');
  const [gamesLoading, setGamesLoading] = useState(true);

  // Step 3: Situation & Sequences
  const [sequences, setSequences] = useState<GameSequence[]>([]);
  const [selectedSeqIndex, setSelectedSeqIndex] = useState<number>(0);
  const [sequencesLoading, setSequencesLoading] = useState(false);
  const [inputMode, setInputMode] = useState<'real_sequence' | 'custom_situation'>('real_sequence');

  // 10-DoF Supported Condition Overrides
  const [conditions, setConditions] = useState<ConditionOverrides>({
    movement: 'stationary',
    camera: 'steady',
    aim_down_sights: false,
    fire: false,
    jump: false,
    reload: false,
    switch_weapon: false,
    melee: false
  });

  // Step 4 & 5: Prediction Result & Explanation
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [predicting, setPredicting] = useState<boolean>(false);
  const [predictError, setPredictError] = useState<string | null>(null);

  // What-If Counterfactual Comparison
  const [showWhatIf, setShowWhatIf] = useState<boolean>(false);
  const [whatIfConditions, setWhatIfConditions] = useState<ConditionOverrides>({
    aim_down_sights: true,
    fire: true,
    movement: 'forward',
    camera: 'turn_right'
  });
  const [whatIfResult, setWhatIfResult] = useState<WhatIfDetailedResponse | null>(null);
  const [whatIfLoading, setWhatIfLoading] = useState<boolean>(false);
  const [whatIfError, setWhatIfError] = useState<string | null>(null);

  // Technical details toggle
  const [showTechDetails, setShowTechDetails] = useState<boolean>(false);

  const [searchParams] = useSearchParams();
  const paramSeq = searchParams.get('seq');
  const paramGame = searchParams.get('game');
  const paramModel = searchParams.get('model');

  // 1. Initial Load: Models & Games
  useEffect(() => {
    // Fetch verified prediction models
    api.getPredictionModels()
      .then((data) => {
        setModels(data);
        if (data.length > 0) {
          if (paramModel && data.some((m) => m.id === paramModel && m.is_available)) {
            setSelectedModel(paramModel);
          } else {
            const primary = data.find((m) => m.is_primary && m.is_available) || data[0];
            setSelectedModel(primary.id);
          }
        }
        setModelsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load prediction models:', err);
        setModelsLoading(false);
      });

    // Fetch supported games
    api.getSupportedGames()
      .then((res) => {
        const gList = res.games || [];
        setGames(gList);
        if (paramGame && gList.some((g) => g.id === paramGame)) {
          setSelectedGame(paramGame);
        } else {
          const firstAvail = gList.find((g) => g.is_available) || gList[0];
          if (firstAvail) {
            setSelectedGame(firstAvail.id);
          }
        }
        setGamesLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load games:', err);
        setGamesLoading(false);
      });
  }, [paramModel, paramGame]);

  // 2. When selected game changes, load real sequences for that game
  useEffect(() => {
    if (!selectedGame) return;
    setSequencesLoading(true);
    setPredictionResult(null);
    setWhatIfResult(null);
    setShowWhatIf(false);
    setPredictError(null);

    api.getPredictionSequences(50, selectedGame)
      .then((res) => {
        const seqs = res.sequences || [];
        setSequences(seqs);
        if (seqs.length > 0) {
          let targetSeq = seqs[0].sequence_index;
          if (paramSeq) {
            const parsed = parseInt(paramSeq, 10);
            if (!isNaN(parsed)) {
              targetSeq = parsed;
            }
          }
          setSelectedSeqIndex(targetSeq);
          executePrediction(targetSeq, selectedModel, undefined);
        }
        setSequencesLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load sequences:', err);
        setSequences([]);
        setSequencesLoading(false);
      });
  }, [selectedGame, paramSeq]);

  // Execute genuine PyTorch prediction
  const executePrediction = (
    seqIndex: number,
    modelName: string,
    overrides?: ConditionOverrides
  ) => {
    setPredicting(true);
    setPredictError(null);

    api.predictNextActionMode({
      model_name: modelName,
      sequence_index: seqIndex,
      condition_overrides: overrides
    })
      .then((res) => {
        setPredictionResult(res);
        setPredicting(false);
      })
      .catch((err) => {
        setPredictError(err.message || 'Prediction failed on neural network checkpoint.');
        setPredicting(false);
      });
  };

  // Handle Model Change: switches checkpoint and re-predicts
  const handleModelSelect = (modelId: string) => {
    const targetModel = models.find((m) => m.id === modelId);
    if (!targetModel || !targetModel.is_available) return;
    setSelectedModel(modelId);
    setWhatIfResult(null);
    if (sequences.length > 0) {
      executePrediction(
        selectedSeqIndex,
        modelId,
        inputMode === 'custom_situation' ? conditions : undefined
      );
    }
  };

  // Handle Game Change
  const handleGameSelect = (gameId: string) => {
    setSelectedGame(gameId);
  };

  // Trigger main prediction button
  const handleRunPrediction = () => {
    const overrides = inputMode === 'custom_situation' ? conditions : undefined;
    executePrediction(selectedSeqIndex, selectedModel, overrides);
  };

  // Run What-If Comparison
  const handleRunWhatIf = () => {
    setWhatIfLoading(true);
    setWhatIfError(null);

    api.predictWhatIf({
      model_name: selectedModel,
      sequence_index: selectedSeqIndex,
      condition_overrides: whatIfConditions
    })
      .then((res) => {
        setWhatIfResult(res);
        setWhatIfLoading(false);
      })
      .catch((err) => {
        setWhatIfError(err.message || 'Counterfactual inference failed.');
        setWhatIfLoading(false);
      });
  };

  // Find currently active sequence object
  const activeSequence = sequences.find((s) => s.sequence_index === selectedSeqIndex) || sequences[0];
  const activeGameInfo = games.find((g) => g.id === selectedGame);
  const activeModelInfo = models.find((m) => m.id === selectedModel);

  return (
    <div className="space-y-10 pb-16">
      {/* Hero Header & Flow Sequence Indicator */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-[#EFE7DB] shadow-xs">
        <div className="max-w-3xl">
          <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-[#C66A4E]/10 text-[#C66A4E] text-xs font-semibold mb-3">
            <Zap className="w-3.5 h-3.5" />
            <span>Interactive Gamer-Facing Digital Twin</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-[#302721] tracking-tight">
            Next Action Prediction
          </h1>
          <p className="mt-2 text-sm sm:text-base text-[#81786E] leading-relaxed">
            Follow our guided 4-step workflow to forecast gameplay actions. Choose an AI model, pick a verified FPS title, configure the recent gameplay situation, and observe the neural network's genuine next-frame prediction.
          </p>
        </div>

        {/* Step Flow Progress Breadcrumbs */}
        <div className="mt-6 pt-6 border-t border-[#EFE7DB]/80 grid grid-cols-2 md:grid-cols-4 gap-3">
          {[
            { num: '1', title: 'Choose Model', desc: activeModelInfo?.name || 'Transformer' },
            { num: '2', title: 'Choose Game', desc: activeGameInfo?.name || 'FPS Title' },
            { num: '3', title: 'Gameplay Situation', desc: inputMode === 'real_sequence' ? 'Real Clip Sequence' : 'Custom Situation' },
            { num: '4', title: 'Forecast & Result', desc: predictionResult ? 'Result Ready' : 'Awaiting Run' },
          ].map((step, idx) => (
            <div
              key={idx}
              className="flex items-center space-x-3 p-2.5 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB]"
            >
              <div className="w-7 h-7 rounded-lg bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold shrink-0">
                {step.num}
              </div>
              <div className="min-w-0">
                <div className="text-xs font-bold text-[#302721] truncate">{step.title}</div>
                <div className="text-[11px] text-[#81786E] truncate">{step.desc}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ============================================================== */}
      {/* STEP 1 — SELECT YOUR AI MODEL */}
      {/* ============================================================== */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="w-7 h-7 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold">
              1
            </span>
            <div>
              <h2 className="text-lg font-bold text-[#302721]">Select Your AI Model</h2>
              <p className="text-xs text-[#81786E]">
                Choose the neural network architecture to load from genuine PyTorch checkpoints.
              </p>
            </div>
          </div>
          {activeModelInfo && (
            <span className="text-xs font-medium text-[#82917A] bg-[#82917A]/10 px-2.5 py-1 rounded-full border border-[#82917A]/20">
              Active Checkpoint: {activeModelInfo.checkpoint}
            </span>
          )}
        </div>

        {modelsLoading ? (
          <div className="p-8 bg-white rounded-xl border border-[#EFE7DB] text-center text-sm text-[#81786E]">
            Loading verified model checkpoints...
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {models.map((m) => {
              const isSelected = selectedModel === m.id;
              const isAvailable = m.is_available;

              return (
                <div
                  key={m.id}
                  onClick={() => isAvailable && handleModelSelect(m.id)}
                  className={`relative p-5 rounded-xl border-2 transition-all flex flex-col justify-between ${
                    isSelected
                      ? 'border-[#C66A4E] bg-white shadow-sm ring-2 ring-[#C66A4E]/20'
                      : isAvailable
                      ? 'border-[#EFE7DB] bg-white hover:border-[#C66A4E]/50 cursor-pointer'
                      : 'border-gray-200 bg-[#F8F5EF]/60 opacity-75 cursor-not-allowed'
                  }`}
                >
                  <div>
                    <div className="flex items-start justify-between">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-bold text-base text-[#302721]">{m.name}</span>
                          {m.is_primary && (
                            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#C66A4E]/10 text-[#C66A4E] border border-[#C66A4E]/20">
                              Primary Twin
                            </span>
                          )}
                          {!m.is_primary && m.id === 'transformer_ssl' && (
                            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#059669]/10 text-[#059669] border border-[#059669]/20">
                              Self-Supervised
                            </span>
                          )}
                          {!m.is_primary && m.id !== 'transformer_ssl' && (
                            <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#81786E]/10 text-[#81786E] border border-[#81786E]/20">
                              Baseline
                            </span>
                          )}
                        </div>
                        <p className="mt-2 text-xs text-[#81786E] leading-relaxed">
                          {m.description}
                        </p>
                      </div>

                      {/* Selected Radio Indicator */}
                      <div
                        className={`w-5 h-5 rounded-full border-2 flex items-center justify-center shrink-0 ml-3 ${
                          isSelected
                            ? 'border-[#C66A4E] bg-[#C66A4E] text-white'
                            : isAvailable
                            ? 'border-[#81786E]/40'
                            : 'border-gray-300 bg-gray-100'
                        }`}
                      >
                        {isSelected && <Check className="w-3 h-3 stroke-[3]" />}
                      </div>
                    </div>
                  </div>

                  {/* Verified Evaluation Metrics & Availability */}
                  <div className="mt-4 pt-3 border-t border-[#EFE7DB] flex flex-col space-y-2 text-xs">
                    {m.evaluation ? (
                      <div className="flex items-center space-x-2 text-[#302721]">
                        <span className="font-bold text-[#C66A4E]">
                          {m.evaluation.multi_label_accuracy_pct}% Accuracy
                        </span>
                        <span className="text-[#81786E]">·</span>
                        <span className="text-[#81786E]">
                          Macro F1: <strong className="text-[#302721]">{m.evaluation.macro_f1}</strong>
                        </span>
                      </div>
                    ) : (
                      <span className="text-[#81786E] text-[11px] italic">
                        Evaluation unavailable (requires trained checkpoint)
                      </span>
                    )}

                    <div>
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold ${
                          isAvailable
                            ? 'bg-[#82917A]/15 text-[#82917A]'
                            : 'bg-amber-100 text-amber-800'
                        }`}
                      >
                        {isAvailable ? '✓ Ready for Inference' : m.status}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Empirical Data & Partition Disclosure */}
        <div className="p-3.5 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex items-start space-x-2.5 text-xs text-[#81786E]">
          <Info className="w-4 h-4 text-[#82917A] shrink-0 mt-0.5" />
          <div className="leading-relaxed">
            <strong className="text-[#302721]">Empirical Evaluation Disclosure: </strong>
            All three architectures (LSTM Baseline, Transformer Encoder, and Self-Supervised Transformer) are trained and evaluated on held-out clip partitions of the available CrossFPS validation partition (<code className="text-[#C66A4E] font-mono text-[11px]">data/CrossFPS-val/</code>, 10,859 test sequences across 517 clips). Because the unextracted 10,000-hour archive is hosted externally, results reflect offline held-out validation performance and are not claimed as independent production test metrics.
          </div>
        </div>
      </section>

      {/* ============================================================== */}
      {/* STEP 2 — SELECT YOUR GAME */}
      {/* ============================================================== */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="w-7 h-7 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold">
              2
            </span>
            <div>
              <h2 className="text-lg font-bold text-[#302721]">Select Your Game</h2>
              <p className="text-xs text-[#81786E]">
                Populated dynamically from verified CrossFPS data. Only titles with genuine playable telemetry sequences are listed.
              </p>
            </div>
          </div>
          {activeGameInfo && (
            <span className="text-xs text-[#81786E]">
              Selected: <strong className="text-[#302721]">{activeGameInfo.name}</strong> ({activeGameInfo.sequence_count.toLocaleString()} sequences)
            </span>
          )}
        </div>

        {gamesLoading ? (
          <div className="p-8 bg-white rounded-xl border border-[#EFE7DB] text-center text-sm text-[#81786E]">
            Loading verified game distribution...
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {games
              .filter((g) => g.is_available && g.sequence_count > 0)
              .map((g) => {
                const isSelected = selectedGame === g.id;

                return (
                  <div
                    key={g.id}
                    onClick={() => handleGameSelect(g.id)}
                    className={`relative p-4 rounded-xl border-2 transition-all cursor-pointer flex flex-col justify-between ${
                      isSelected
                        ? 'border-[#C66A4E] bg-white shadow-sm ring-2 ring-[#C66A4E]/20'
                        : 'border-[#EFE7DB] bg-white hover:border-[#C66A4E]/40'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] font-bold uppercase tracking-wider text-[#C66A4E]">
                          {g.genre}
                        </span>
                        {isSelected && (
                          <span className="w-4 h-4 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-[10px]">
                            <Check className="w-2.5 h-2.5 stroke-[3]" />
                          </span>
                        )}
                      </div>
                      <div className="font-bold text-sm text-[#302721] mt-1">{g.name}</div>
                      <div className="text-[11px] text-[#81786E]">{g.tagline}</div>
                    </div>

                    <div className="mt-3 pt-2.5 border-t border-[#EFE7DB] flex items-center justify-between text-xs">
                      <span className="text-[11px] font-medium text-[#302721]">
                        {g.sequence_count.toLocaleString()} seqs
                      </span>
                      <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-[#82917A]/15 text-[#82917A]">
                        Ready
                      </span>
                    </div>
                  </div>
                );
              })}
          </div>
        )}
      </section>

      {/* ============================================================== */}
      {/* STEP 3 — CONFIGURE THE GAMEPLAY SITUATION */}
      {/* ============================================================== */}
      <section className="space-y-5 bg-white p-6 rounded-2xl border border-[#EFE7DB] shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center space-x-3">
            <span className="w-7 h-7 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold">
              3
            </span>
            <div>
              <h2 className="text-lg font-bold text-[#302721]">
                What is happening in your game right now?
              </h2>
              <p className="text-xs text-[#81786E]">
                Define the player's recent controller inputs before requesting a prediction.
              </p>
            </div>
          </div>

          {/* Mode Switcher: Real Sequence vs Custom Situation */}
          <div className="flex items-center p-1 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] self-start sm:self-auto">
            <button
              onClick={() => setInputMode('real_sequence')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                inputMode === 'real_sequence'
                  ? 'bg-white text-[#302721] shadow-xs border border-[#EFE7DB]'
                  : 'text-[#81786E] hover:text-[#302721]'
              }`}
            >
              Use Real Gameplay Sequence
            </button>
            <button
              onClick={() => setInputMode('custom_situation')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                inputMode === 'custom_situation'
                  ? 'bg-white text-[#302721] shadow-xs border border-[#EFE7DB]'
                  : 'text-[#81786E] hover:text-[#302721]'
              }`}
            >
              Customize Situation (What-If)
            </button>
          </div>
        </div>

        {/* Input Source A: Real Gameplay Sequence Picker */}
        {inputMode === 'real_sequence' && (
          <div className="space-y-4 pt-2">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <label className="text-xs font-bold text-[#302721]">
                Select a genuine sequence from {activeGameInfo?.name}:
              </label>
              <span className="text-[11px] text-[#81786E]">
                Showing {sequences.length} available test sequences
              </span>
            </div>

            {sequencesLoading ? (
              <div className="p-6 text-center text-xs text-[#81786E] bg-[#F8F5EF] rounded-xl border border-[#EFE7DB]">
                Loading verified sequences...
              </div>
            ) : sequences.length === 0 ? (
              <div className="p-6 text-center text-xs text-[#81786E] bg-[#F8F5EF] rounded-xl border border-[#EFE7DB]">
                No sequences available for this title in the current validation split.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <select
                  value={selectedSeqIndex}
                  onChange={(e) => {
                    const idx = Number(e.target.value);
                    setSelectedSeqIndex(idx);
                    executePrediction(idx, selectedModel, undefined);
                  }}
                  className="w-full text-xs font-medium text-[#302721] bg-[#F8F5EF] border border-[#EFE7DB] rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-[#C66A4E]"
                >
                  {sequences.map((s) => (
                    <option key={s.sequence_index} value={s.sequence_index}>
                      {s.sequence_id} · Sequence #{s.sequence_index + 1}
                    </option>
                  ))}
                </select>

                {/* Quick Info Box for Selected Sequence */}
                {activeSequence && (
                  <div className="p-3 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex items-center justify-between text-xs">
                    <div>
                      <span className="text-[#81786E]">Clip: </span>
                      <strong className="text-[#302721] font-mono text-[11px]">{activeSequence.clip_id.slice(0, 26)}...</strong>
                    </div>
                    <div className="text-right">
                      <span className="text-[#81786E]">Telemetry Status: </span>
                      <strong className="text-[#82917A]">10-DoF Frame Telemetry Verified</strong>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Sequence Input Timeline Visualization */}
            {activeSequence && (
              <div className="mt-3 p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB]">
                <div className="text-xs font-bold text-[#302721] mb-2 flex items-center justify-between">
                  <span>Recent Action Timeline (Last 4 Time Frames):</span>
                  <span className="text-[11px] text-[#81786E] font-normal">16-frame window</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                  {activeSequence.recent_actions.map((act, idx) => (
                    <div
                      key={idx}
                      className="p-2.5 rounded-lg bg-white border border-[#EFE7DB] text-center"
                    >
                      <div className="text-[10px] text-[#81786E] uppercase font-bold">
                        Frame t - {4 - idx}
                      </div>
                      <div className="mt-1 font-semibold text-xs text-[#302721] truncate">
                        {act}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Input Source B: Customize Situation Controls */}
        {inputMode === 'custom_situation' && (
          <div className="space-y-5 pt-2">
            <div className="p-3.5 rounded-xl bg-[#C66A4E]/10 border border-[#C66A4E]/20 text-xs text-[#302721] leading-relaxed">
              <strong>Custom Situation Mode:</strong> Modifying supported controller conditions on top of baseline sequence{' '}
              <span className="font-mono">{activeSequence?.sequence_id || 'SEQ-CFPS-00001'}</span>. These controls directly modulate the 10-DoF gamepad channels in the final frames before the predicted step.
            </div>

            {/* 10-DoF Action Buttons (Discrete) */}
            <div className="space-y-2">
              <label className="text-xs font-bold text-[#302721]">
                A. Action Buttons & Triggers (Supported 10-DoF Controller Channels):
              </label>
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
                {[
                  { key: 'aim_down_sights', label: 'Aim / Scope', code: 'LT', icon: '🎯' },
                  { key: 'fire', label: 'Fire Trigger', code: 'RT', icon: '🔥' },
                  { key: 'jump', label: 'Jump Impulse', code: 'A', icon: '⏫' },
                  { key: 'reload', label: 'Reload Action', code: 'X', icon: '🔄' },
                  { key: 'switch_weapon', label: 'Switch Weapon', code: 'Y', icon: '🔁' },
                  { key: 'melee', label: 'Melee Strike', code: 'R3', icon: '👊' },
                ].map((item) => {
                  const isActive = Boolean(conditions[item.key as keyof ConditionOverrides]);
                  return (
                    <button
                      key={item.key}
                      onClick={() =>
                        setConditions((prev) => ({
                          ...prev,
                          [item.key]: !isActive,
                        }))
                      }
                      className={`p-3 rounded-xl border text-left transition-all flex flex-col justify-between ${
                        isActive
                          ? 'border-[#C66A4E] bg-[#C66A4E]/10 text-[#C66A4E] font-bold shadow-xs'
                          : 'border-[#EFE7DB] bg-white text-[#81786E] hover:border-[#81786E]'
                      }`}
                    >
                      <div className="flex items-center justify-between text-base">
                        <span>{item.icon}</span>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-[#F8F5EF] border border-[#EFE7DB]">
                          {item.code}
                        </span>
                      </div>
                      <div className="mt-2 text-xs font-semibold">{item.label}</div>
                      <div className="text-[10px] text-[#81786E]">
                        {isActive ? 'Pressed (1.0)' : 'Released (0.0)'}
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Continuous Movement & Camera Controls */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
              {/* Analog Movement */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-[#302721]">
                  B. Analog Movement (Left Stick LX, LY):
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {[
                    { id: 'forward', label: 'Move Forward', val: 'LY = +0.85' },
                    { id: 'backward', label: 'Move Backward', val: 'LY = -0.85' },
                    { id: 'strafe_left', label: 'Strafe Left', val: 'LX = -0.85' },
                    { id: 'strafe_right', label: 'Strafe Right', val: 'LX = +0.85' },
                    { id: 'stationary', label: 'Stationary', val: 'LX=0, LY=0' },
                  ].map((m) => (
                    <button
                      key={m.id}
                      onClick={() => setConditions((prev) => ({ ...prev, movement: m.id as any }))}
                      className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                        conditions.movement === m.id
                          ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                          : 'border-[#EFE7DB] bg-white text-[#81786E] hover:border-[#81786E]'
                      }`}
                    >
                      <div>{m.label}</div>
                      <div className="text-[10px] text-[#81786E] font-mono">{m.val}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Analog Camera */}
              <div className="space-y-2">
                <label className="text-xs font-bold text-[#302721]">
                  C. Analog Camera / Look (Right Stick RX, RY):
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                  {[
                    { id: 'turn_left', label: 'Turn Left', val: 'RX = -0.75' },
                    { id: 'turn_right', label: 'Turn Right', val: 'RX = +0.75' },
                    { id: 'look_up', label: 'Look Up', val: 'RY = +0.65' },
                    { id: 'look_down', label: 'Look Down', val: 'RY = -0.65' },
                    { id: 'steady', label: 'Steady Camera', val: 'RX=0, RY=0' },
                  ].map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setConditions((prev) => ({ ...prev, camera: c.id as any }))}
                      className={`p-2.5 rounded-lg border text-left text-xs transition-all ${
                        conditions.camera === c.id
                          ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                          : 'border-[#EFE7DB] bg-white text-[#81786E] hover:border-[#81786E]'
                      }`}
                    >
                      <div>{c.label}</div>
                      <div className="text-[10px] text-[#81786E] font-mono">{c.val}</div>
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Reset custom situation button */}
            <div className="pt-2 flex justify-end">
              <button
                onClick={() =>
                  setConditions({
                    movement: 'stationary',
                    camera: 'steady',
                    aim_down_sights: false,
                    fire: false,
                    jump: false,
                    reload: false,
                    switch_weapon: false,
                    melee: false
                  })
                }
                className="inline-flex items-center space-x-1.5 text-xs text-[#81786E] hover:text-[#302721] transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset Conditions to Neutral</span>
              </button>
            </div>
          </div>
        )}

        {/* Dataset Limitations: Weapon & Environment Notes */}
        <div className="pt-4 border-t border-[#EFE7DB] grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-[#81786E]">
          <div className="p-3 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex items-start space-x-2.5">
            <Info className="w-4 h-4 text-[#C66A4E] shrink-0 mt-0.5" />
            <div>
              <strong className="text-[#302721]">Weapon Selection: </strong>
              Weapon identity is not available in the current dataset, so it cannot directly condition this prediction. CrossFPS captures controller button telemetry; weapon loadout types are not recorded.
            </div>
          </div>
          <div className="p-3 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex items-start space-x-2.5">
            <Info className="w-4 h-4 text-[#82917A] shrink-0 mt-0.5" />
            <div>
              <strong className="text-[#302721]">Environment & Maps: </strong>
              Spatial map identity is not present in controller telemetry streams. The model forecasts next actions purely from temporal input patterns and analog velocities.
            </div>
          </div>
        </div>
      </section>

      {/* ============================================================== */}
      {/* STEP 4 — PREDICT THE NEXT ACTION BUTTON */}
      {/* ============================================================== */}
      <section className="text-center">
        <button
          onClick={handleRunPrediction}
          disabled={predicting || !activeGameInfo?.is_available || sequences.length === 0}
          className={`w-full sm:w-auto min-w-[280px] px-8 py-4 rounded-xl text-white font-bold text-base shadow-md transition-all flex items-center justify-center space-x-3 mx-auto ${
            predicting || !activeGameInfo?.is_available || sequences.length === 0
              ? 'bg-[#81786E]/50 cursor-not-allowed'
              : 'bg-[#C66A4E] hover:bg-[#B95D3B] hover:shadow-lg active:scale-95'
          }`}
        >
          <Zap className={`w-5 h-5 ${predicting ? 'animate-bounce' : ''}`} />
          <span>
            {predicting
              ? 'Running PyTorch Inference...'
              : `Predict My Next Action (${activeModelInfo?.name || 'Model'})`}
          </span>
        </button>

        {predictError && (
          <div className="mt-3 text-xs text-red-600 max-w-md mx-auto">
            {predictError}
          </div>
        )}
      </section>

      {/* ============================================================== */}
      {/* STEP 4 & 5 — PREDICTION RESULT & UNDERSTAND THE RESULT */}
      {/* ============================================================== */}
      {predictionResult && (
        <section className="space-y-6">
          <div className="bg-white rounded-2xl p-6 sm:p-8 border-2 border-[#C66A4E] shadow-sm space-y-6">
            {/* Header: Result pill & Model info */}
            <div className="flex flex-wrap items-center justify-between border-b border-[#EFE7DB] pb-4 gap-2">
              <div className="flex items-center space-x-2">
                <span className="w-7 h-7 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold">
                  4
                </span>
                <span className="text-xs font-bold uppercase tracking-wider text-[#C66A4E]">
                  Verified Prediction Result
                </span>
              </div>
              <div className="flex items-center space-x-2 text-xs text-[#81786E]">
                <span>Model: <strong className="text-[#302721]">{predictionResult.trace.model_name}</strong></span>
                <span>·</span>
                <span>Game: <strong className="text-[#302721]">{predictionResult.game_name}</strong></span>
              </div>
            </div>

            {/* Main Result Display - Balanced 2-Card Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Card 1: Discrete Predicted Next Action */}
              <div className="p-6 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex flex-col justify-between space-y-4">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#81786E] uppercase tracking-wider">
                      Forecasted Action Output
                    </span>
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#C66A4E]/10 text-[#C66A4E] border border-[#C66A4E]/20">
                      Multi-Label Discrete
                    </span>
                  </div>
                  <div className="mt-3 text-2xl sm:text-3xl font-extrabold text-[#302721] tracking-tight flex items-center space-x-3">
                    <span className="text-3xl">
                      {ACTION_ICONS[predictionResult.trace.predicted_next_action] || '⚡'}
                    </span>
                    <span>{predictionResult.trace.predicted_next_action}</span>
                  </div>
                  <p className="mt-2 text-xs text-[#81786E] leading-relaxed">
                    Estimated physical button activation for the transition frame based on temporal input patterns across the preceding 16 frames.
                  </p>
                </div>

                <div className="pt-3 border-t border-[#EFE7DB] flex items-center justify-between text-xs">
                  <span className="text-[#81786E]">Lead Action Confidence:</span>
                  <span className="font-mono font-bold text-[#302721] text-sm">
                    {(predictionResult.trace.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Card 2: Continuous Analog Steering & Orientation (4-DoF) */}
              <div className="p-6 rounded-xl bg-white border border-[#EFE7DB] flex flex-col justify-between space-y-4 shadow-xs">
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-[#81786E] uppercase tracking-wider">
                      Continuous Trajectory Head
                    </span>
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full bg-[#82917A]/15 text-[#82917A] border border-[#82917A]/30">
                      4-DoF Analog Channels
                    </span>
                  </div>

                  <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                    <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
                      <span className="text-[#81786E] block font-medium">Movement Vector:</span>
                      <strong className="text-[#302721] block mt-0.5 font-bold">
                        {predictionResult.continuous_prediction.movement_summary}
                      </strong>
                      <div className="text-[10px] text-[#81786E] font-mono mt-1 pt-1 border-t border-[#EFE7DB]/60">
                        LX={predictionResult.continuous_prediction.strafe_lx}, LY={predictionResult.continuous_prediction.move_ly}
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB]">
                      <span className="text-[#81786E] block font-medium">Camera Adjustment:</span>
                      <strong className="text-[#302721] block mt-0.5 font-bold">
                        {predictionResult.continuous_prediction.camera_summary}
                      </strong>
                      <div className="text-[10px] text-[#81786E] font-mono mt-1 pt-1 border-t border-[#EFE7DB]/60">
                        RX={predictionResult.continuous_prediction.turn_yaw_rx}, RY={predictionResult.continuous_prediction.look_pitch_ry}
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#EFE7DB] flex items-center justify-between text-xs">
                  <span className="text-[#81786E]">Analog Head Range:</span>
                  <span className="font-mono text-[11px] text-[#81786E]">Tanh normalized [-1.0, +1.0]</span>
                </div>
              </div>
            </div>

            {/* Action Activation Probabilities Breakdown */}
            <div className="space-y-3 pt-2">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-[#302721]">
                  Neural Network Output Activation Scores:
                </span>
                <span className="text-[11px] text-[#81786E]">
                  Sigmoid multi-label probabilities across 6 discrete action channels
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                {predictionResult.trace.ranked_probabilities.map((item) => {
                  const color = ACTION_COLORS[item.class_name] || '#C66A4E';
                  return (
                    <div
                      key={item.class_name}
                      className="p-3 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] flex flex-col justify-between"
                    >
                      <div className="flex items-center justify-between text-xs mb-2">
                        <span className="font-semibold text-[#302721] truncate">
                          {item.class_name}
                        </span>
                        <span className="font-mono text-[11px] font-bold text-[#302721]">
                          {item.percentage}%
                        </span>
                      </div>
                      <div className="w-full bg-[#EFE7DB] h-2 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.min(100, Math.max(4, item.percentage))}%`,
                            backgroundColor: color
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="text-[11px] text-[#81786E] italic">
                Note: Activation scores reflect the model's confidence in individual gamepad buttons being engaged simultaneously. They do not represent human certainty.
              </p>
            </div>

            {/* ============================================================== */}
            {/* STEP 5 — WHY THIS PREDICTION? (Observable Explanation) */}
            {/* ============================================================== */}
            <div className="mt-6 pt-6 border-t border-[#EFE7DB] space-y-3">
              <div className="flex items-center space-x-2">
                <span className="w-7 h-7 rounded-full bg-[#C66A4E] text-white flex items-center justify-center text-xs font-bold">
                  5
                </span>
                <h3 className="text-base font-bold text-[#302721]">Why this prediction?</h3>
              </div>

              <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] text-xs text-[#302721] leading-relaxed">
                {predictionResult.explanation}
              </div>

              {/* Context Summary Table */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
                <div className="p-3 rounded-lg bg-white border border-[#EFE7DB]">
                  <span className="text-[10px] uppercase font-bold text-[#81786E]">Input History</span>
                  <div className="mt-1 font-semibold text-[#302721] truncate">
                    {predictionResult.trace.previous_observed_actions.join(', ')}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-white border border-[#EFE7DB]">
                  <span className="text-[10px] uppercase font-bold text-[#81786E]">Condition Overrides</span>
                  <div className="mt-1 font-semibold text-[#302721] truncate">
                    {predictionResult.applied_conditions.length > 0
                      ? predictionResult.applied_conditions.join(', ')
                      : 'Baseline Sequence'}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-white border border-[#EFE7DB]">
                  <span className="text-[10px] uppercase font-bold text-[#81786E]">Architecture</span>
                  <div className="mt-1 font-semibold text-[#302721] truncate">
                    {predictionResult.trace.model_name}
                  </div>
                </div>
                <div className="p-3 rounded-lg bg-white border border-[#EFE7DB]">
                  <span className="text-[10px] uppercase font-bold text-[#81786E]">Forecasted Action</span>
                  <div className="mt-1 font-semibold text-[#C66A4E] truncate">
                    {predictionResult.trace.predicted_next_action}
                  </div>
                </div>
              </div>
            </div>

            {/* Optional What-If Expansion Button */}
            <div className="pt-4 border-t border-[#EFE7DB] flex flex-wrap items-center justify-between gap-3">
              <button
                onClick={() => setShowWhatIf(!showWhatIf)}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold text-[#302721] bg-[#F8F5EF] hover:bg-[#EFE7DB] border border-[#EFE7DB] transition-all"
              >
                <Sliders className="w-3.5 h-3.5 text-[#C66A4E]" />
                <span>{showWhatIf ? 'Hide Counterfactual Comparison' : 'Compare Another Situation (What-If)'}</span>
                {showWhatIf ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              <button
                onClick={() => setShowTechDetails(!showTechDetails)}
                className="text-xs text-[#81786E] hover:text-[#302721] flex items-center space-x-1"
              >
                <span>{showTechDetails ? 'Hide' : 'Show'} Technical Details</span>
                {showTechDetails ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>
            </div>

            {/* Collapsible Technical Details */}
            {showTechDetails && (
              <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] text-xs font-mono space-y-1 text-[#81786E]">
                <div>Sequence ID: <span className="text-[#302721]">{predictionResult.trace.sequence_id}</span></div>
                <div>Clip Source: <span className="text-[#302721]">{predictionResult.trace.player_id}</span></div>
                <div>Checkpoint: <span className="text-[#302721]">{predictionResult.trace.checkpoint_file}</span></div>
                <div>Input Tensor Dimensions: <span className="text-[#302721]">[1, 16, 10] (T=16 frames, 10-DoF)</span></div>
                <div>Inference Time: <span className="text-[#302721]">{predictionResult.trace.timestamp}</span></div>
              </div>
            )}
          </div>

          {/* ============================================================== */}
          {/* OPTIONAL — WHAT-IF COUNTERFACTUAL COMPARISON */}
          {/* ============================================================== */}
          {showWhatIf && (
            <div className="bg-white rounded-2xl p-6 sm:p-8 border border-[#EFE7DB] shadow-xs space-y-6">
              <div className="flex items-center justify-between border-b border-[#EFE7DB] pb-4">
                <div>
                  <h3 className="text-base font-bold text-[#302721] flex items-center space-x-2">
                    <Sparkles className="w-4 h-4 text-[#C66A4E]" />
                    <span>Hypothetical Counterfactual Comparison</span>
                  </h3>
                  <p className="text-xs text-[#81786E] mt-0.5">
                    Alter supported inputs to simulate an alternate gameplay branch and measure genuine probability shifts through the neural network.
                  </p>
                </div>
              </div>

              {/* What-If Input Toggles */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <button
                  onClick={() =>
                    setWhatIfConditions((prev) => ({
                      ...prev,
                      aim_down_sights: !prev.aim_down_sights
                    }))
                  }
                  className={`p-3 rounded-xl border text-left text-xs transition-all ${
                    whatIfConditions.aim_down_sights
                      ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                      : 'border-[#EFE7DB] bg-white text-[#81786E]'
                  }`}
                >
                  <div className="font-bold">Aim Down Sights (LT)</div>
                  <div className="text-[10px] text-[#81786E] mt-0.5">
                    {whatIfConditions.aim_down_sights ? 'Scoped (1.0)' : 'Unscoped (0.0)'}
                  </div>
                </button>

                <button
                  onClick={() =>
                    setWhatIfConditions((prev) => ({
                      ...prev,
                      fire: !prev.fire
                    }))
                  }
                  className={`p-3 rounded-xl border text-left text-xs transition-all ${
                    whatIfConditions.fire
                      ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                      : 'border-[#EFE7DB] bg-white text-[#81786E]'
                  }`}
                >
                  <div className="font-bold">Fire Trigger (RT)</div>
                  <div className="text-[10px] text-[#81786E] mt-0.5">
                    {whatIfConditions.fire ? 'Firing (1.0)' : 'Released (0.0)'}
                  </div>
                </button>

                <button
                  onClick={() =>
                    setWhatIfConditions((prev) => ({
                      ...prev,
                      jump: !prev.jump
                    }))
                  }
                  className={`p-3 rounded-xl border text-left text-xs transition-all ${
                    whatIfConditions.jump
                      ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                      : 'border-[#EFE7DB] bg-white text-[#81786E]'
                  }`}
                >
                  <div className="font-bold">Jump Impulse (A)</div>
                  <div className="text-[10px] text-[#81786E] mt-0.5">
                    {whatIfConditions.jump ? 'Airborne (1.0)' : 'Grounded (0.0)'}
                  </div>
                </button>

                <button
                  onClick={() =>
                    setWhatIfConditions((prev) => ({
                      ...prev,
                      camera: prev.camera === 'turn_right' ? 'steady' : 'turn_right'
                    }))
                  }
                  className={`p-3 rounded-xl border text-left text-xs transition-all ${
                    whatIfConditions.camera === 'turn_right'
                      ? 'border-[#C66A4E] bg-[#C66A4E]/10 font-bold text-[#C66A4E]'
                      : 'border-[#EFE7DB] bg-white text-[#81786E]'
                  }`}
                >
                  <div className="font-bold">Camera Turn (RX)</div>
                  <div className="text-[10px] text-[#81786E] mt-0.5">
                    {whatIfConditions.camera === 'turn_right' ? 'Turn Right (+0.75)' : 'Steady (0.0)'}
                  </div>
                </button>
              </div>

              {/* Run What-If Inference Button */}
              <div>
                <button
                  onClick={handleRunWhatIf}
                  disabled={whatIfLoading}
                  className="px-6 py-2.5 rounded-xl bg-[#302721] hover:bg-[#1a1410] text-white text-xs font-bold transition-all flex items-center space-x-2"
                >
                  <Sparkles className="w-3.5 h-3.5 text-[#C66A4E]" />
                  <span>{whatIfLoading ? 'Running Counterfactual Forward Pass...' : 'Evaluate What-If Scenario'}</span>
                </button>
                {whatIfError && (
                  <div className="mt-2 text-xs text-red-600">{whatIfError}</div>
                )}
              </div>

              {/* Side-by-Side Comparison */}
              {whatIfResult && (
                <div className="pt-4 border-t border-[#EFE7DB] space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Baseline Column */}
                    <div className="p-5 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-3">
                      <div className="text-[11px] uppercase font-bold text-[#81786E]">
                        Baseline Scenario (Original Sequence)
                      </div>
                      <div className="text-xl font-bold text-[#302721]">
                        {whatIfResult.baseline_prediction.predicted_action}
                      </div>
                      <div className="text-xs text-[#81786E]">
                        Top Activation: <strong className="text-[#302721]">{whatIfResult.baseline_prediction.confidence_pct}%</strong>
                      </div>
                      <div className="text-[11px] text-[#81786E]">
                        Movement: Forward={whatIfResult.baseline_prediction.continuous.move_ly}, Turn={whatIfResult.baseline_prediction.continuous.turn_yaw_rx}
                      </div>
                    </div>

                    {/* What-If Column */}
                    <div className="p-5 rounded-xl bg-white border-2 border-[#C66A4E] space-y-3">
                      <div className="text-[11px] uppercase font-bold text-[#C66A4E] flex items-center space-x-1">
                        <span>Hypothetical Scenario</span>
                        {whatIfResult.action_changed && (
                          <span className="px-2 py-0.5 rounded-full bg-[#C66A4E]/10 text-[10px] font-bold">
                            Action Shifted!
                          </span>
                        )}
                      </div>
                      <div className="text-xl font-bold text-[#302721]">
                        {whatIfResult.what_if_prediction.predicted_action}
                      </div>
                      <div className="text-xs text-[#81786E]">
                        Top Activation: <strong className="text-[#C66A4E]">{whatIfResult.what_if_prediction.confidence_pct}%</strong>
                      </div>
                      <div className="text-[11px] text-[#81786E]">
                        Movement: Forward={whatIfResult.what_if_prediction.continuous.move_ly}, Turn={whatIfResult.what_if_prediction.continuous.turn_yaw_rx}
                      </div>
                    </div>
                  </div>

                  {/* Probability Shifts Table */}
                  <div className="p-4 rounded-xl bg-[#F8F5EF] border border-[#EFE7DB] space-y-2">
                    <div className="text-xs font-bold text-[#302721]">
                      Measured Probability Shifts (Genuine Neural Network Deltas):
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2 text-xs">
                      {Object.entries(whatIfResult.probability_shifts_pct).map(([action, shift]) => (
                        <div key={action} className="p-2.5 rounded-lg bg-white border border-[#EFE7DB] text-center">
                          <div className="text-[10px] text-[#81786E] truncate">{action}</div>
                          <div
                            className={`font-mono font-bold text-xs mt-1 ${
                              shift > 0
                                ? 'text-emerald-700'
                                : shift < 0
                                ? 'text-amber-700'
                                : 'text-[#81786E]'
                            }`}
                          >
                            {shift > 0 ? `+${shift}%` : `${shift}%`}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <p className="text-[11px] text-[#81786E] italic">
                    Disclaimer: This comparison measures the neural network's internal sensitivity to modulated input channels. It represents a model response and does not claim to prove real-world human causal effect.
                  </p>
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
};
