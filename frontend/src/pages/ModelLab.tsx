import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Activity,
  Layers,
  Sparkles,
  TrendingDown,
  Clock
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';
import { api, ModelInfo, TrainingProgress } from '../services/api';

export const ModelLab: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('transformer');
  const [epochs, setEpochs] = useState<number>(5);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<number>(0.001);
  const [sequenceLength, setSequenceLength] = useState<number>(16);

  const [trainingStatus, setTrainingStatus] = useState<TrainingProgress | null>(null);
  const [historyData, setHistoryData] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const loadModels = () => {
    api.getModels()
      .then((data) => setModels(data))
      .catch((err) => console.error(err));
  };

  const loadHistory = (modelId: string) => {
    api.getModelHistory(modelId)
      .then((hist) => {
        if (hist && hist.epochs) {
          const chartRows = hist.epochs.map((ep: number, i: number) => ({
            epoch: `Ep ${ep}`,
            train_loss: hist.train_loss[i],
            val_loss: hist.val_loss[i],
            train_acc: hist.train_acc[i] ? (hist.train_acc[i] * 100).toFixed(1) : undefined,
            val_acc: hist.val_acc[i] ? (hist.val_acc[i] * 100).toFixed(1) : undefined,
          }));
          setHistoryData(chartRows);
        }
      })
      .catch(() => {
        setHistoryData([]);
      });
  };

  useEffect(() => {
    loadModels();
    loadHistory(selectedModel);
  }, [selectedModel]);

  // Poll training status if training is active
  useEffect(() => {
    const poll = setInterval(() => {
      api.getTrainingStatus()
        .then((status) => {
          setTrainingStatus(status);
          if (status.is_training && status.history && status.history.epochs.length > 0) {
            const chartRows = status.history.epochs.map((ep: number, i: number) => ({
              epoch: `Ep ${ep}`,
              train_loss: status.history!.train_loss[i],
              val_loss: status.history!.val_loss[i],
            }));
            setHistoryData(chartRows);
          } else if (!status.is_training && status.stage === 'completed') {
            loadModels();
            loadHistory(selectedModel);
          }
        })
        .catch(() => {});
    }, 2000);

    return () => clearInterval(poll);
  }, [selectedModel]);

  const handleStartTraining = () => {
    setLoading(true);
    setStatusMessage(null);
    api.startTraining({
      model_type: selectedModel,
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      sequence_length: sequenceLength,
    })
      .then((res) => {
        setStatusMessage(`Training started for ${res.model_type} (${res.epochs} epochs). Tracking real progress...`);
        setLoading(false);
      })
      .catch((err) => {
        setStatusMessage(`Error: ${err.message}`);
        setLoading(false);
      });
  };

  const currentModelMeta = models.find((m) => m.id === selectedModel);

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-[#302721] tracking-tight flex items-center">
          <Cpu className="w-6 h-6 mr-2.5 text-[#C66A4E]" />
          Model Lab
        </h1>
        <p className="text-sm text-[#81786E] mt-1">
          Configure architectures, run real PyTorch training jobs, track convergence loss curves, and manage checkpoints.
        </p>
      </div>

      {statusMessage && (
        <div className={`p-4 rounded-xl text-xs flex items-center ${
          statusMessage.startsWith('Error')
            ? 'bg-red-50 text-red-700 border border-red-200'
            : 'bg-[#82917A]/10 text-[#82917A] border border-[#82917A]/30 font-medium'
        }`}>
          {statusMessage.startsWith('Error') ? (
            <AlertCircle className="w-4 h-4 mr-2 text-red-500" />
          ) : (
            <CheckCircle2 className="w-4 h-4 mr-2 text-[#82917A]" />
          )}
          {statusMessage}
        </div>
      )}

      {/* Model Architecture Selection Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {models.map((m) => {
          const isSelected = selectedModel === m.id;
          return (
            <div
              key={m.id}
              onClick={() => setSelectedModel(m.id)}
              className={`p-4 rounded-xl border cursor-pointer transition relative flex flex-col justify-between ${
                isSelected
                  ? 'bg-white border-[#C66A4E] shadow-sm ring-1 ring-[#C66A4E]'
                  : 'bg-white border-[#EFE7DB] hover:border-[#81786E]'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] uppercase font-bold tracking-wider text-[#81786E]">
                    {m.category}
                  </span>
                  {m.is_primary && (
                    <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#C66A4E] text-white">
                      Primary
                    </span>
                  )}
                </div>
                <h3 className="font-bold text-sm text-[#302721]">{m.name}</h3>
                <p className="text-xs text-[#81786E] mt-1 leading-relaxed line-clamp-3">
                  {m.description}
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-[#EFE7DB] flex items-center justify-between text-[11px]">
                <span className="text-[#81786E]">Checkpoint:</span>
                {m.checkpoint_available ? (
                  <span className="text-[#82917A] font-semibold flex items-center">
                    <CheckCircle2 className="w-3 h-3 mr-1" /> Ready (.pt)
                  </span>
                ) : (
                  <span className="text-[#81786E]">Not Trained</span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Training Controls and Status Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Hyperparameter Settings Panel */}
        <div className="bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
          <h2 className="text-sm font-semibold text-[#302721] flex items-center">
            <Layers className="w-4 h-4 mr-2 text-[#C66A4E]" />
            Training Hyperparameters
          </h2>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-[#81786E] mb-1 font-medium">Selected Model</label>
              <input
                type="text"
                disabled
                value={currentModelMeta?.name || selectedModel}
                className="w-full px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-[#F8F5EF] text-[#302721] font-medium"
              />
            </div>

            <div>
              <label className="block text-[#81786E] mb-1 font-medium">
                Epochs ({epochs})
              </label>
              <input
                type="range"
                min="1"
                max="25"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
                className="w-full accent-[#C66A4E]"
              />
            </div>

            <div>
              <label className="block text-[#81786E] mb-1 font-medium">
                Batch Size ({batchSize})
              </label>
              <select
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-[#302721]"
              >
                <option value={16}>16</option>
                <option value={32}>32</option>
                <option value={64}>64</option>
              </select>
            </div>

            <div>
              <label className="block text-[#81786E] mb-1 font-medium">
                Learning Rate ({learningRate})
              </label>
              <select
                value={learningRate}
                onChange={(e) => setLearningRate(Number(e.target.value))}
                className="w-full px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-[#302721]"
              >
                <option value={0.005}>0.005</option>
                <option value={0.001}>0.001 (Recommended)</option>
                <option value={0.0005}>0.0005</option>
              </select>
            </div>

            <div>
              <label className="block text-[#81786E] mb-1 font-medium">
                Sequence Window Size ({sequenceLength} frames)
              </label>
              <input
                type="number"
                min="8"
                max="32"
                value={sequenceLength}
                onChange={(e) => setSequenceLength(Number(e.target.value))}
                className="w-full px-3 py-1.5 rounded-lg border border-[#EFE7DB] bg-white text-[#302721]"
              />
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={handleStartTraining}
              disabled={loading || trainingStatus?.is_training}
              className={`w-full py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center transition shadow-sm ${
                trainingStatus?.is_training
                  ? 'bg-amber-100 text-amber-800 border border-amber-300 cursor-not-allowed'
                  : 'bg-[#C66A4E] hover:bg-[#B95D3B] text-white'
              }`}
            >
              {trainingStatus?.is_training ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-spin" /> Training in Progress...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2 fill-current" /> Start Training Job
                </>
              )}
            </button>
          </div>
        </div>

        {/* Real Training Convergence Curves */}
        <div className="md:col-span-2 bg-white border border-[#EFE7DB] rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-[#302721] flex items-center">
              <TrendingDown className="w-4 h-4 mr-2 text-[#C66A4E]" />
              Convergence Loss Curves ({currentModelMeta?.name})
            </h2>
            {trainingStatus?.is_training && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-800 font-medium animate-pulse">
                Epoch {trainingStatus.current_epoch}/{trainingStatus.total_epochs}
              </span>
            )}
          </div>

          {historyData.length > 0 ? (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#EFE7DB" />
                  <XAxis dataKey="epoch" stroke="#81786E" fontSize={11} />
                  <YAxis stroke="#81786E" fontSize={11} domain={['auto', 'auto']} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#FFFFFF', borderColor: '#EFE7DB', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    name="Train Loss"
                    stroke="#C66A4E"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    name="Validation Loss"
                    stroke="#82917A"
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-64 flex flex-col items-center justify-center border border-dashed border-[#EFE7DB] rounded-lg text-xs text-[#81786E] space-y-2">
              <Clock className="w-6 h-6 text-[#81786E]" />
              <span>No training history found for this model.</span>
              <span>Click "Start Training Job" to execute real gradient descent optimization.</span>
            </div>
          )}

          {/* Real Live Metrics Status */}
          {trainingStatus && (
            <div className="p-3 rounded-lg bg-[#F8F5EF] border border-[#EFE7DB] text-xs flex flex-wrap gap-4 justify-between">
              <div>
                <span className="text-[#81786E]">Status:</span>{' '}
                <span className="font-semibold text-[#302721] capitalize">{trainingStatus.stage}</span>
              </div>
              <div>
                <span className="text-[#81786E]">Train Loss:</span>{' '}
                <span className="font-mono font-semibold text-[#302721]">
                  {trainingStatus.train_loss !== null ? trainingStatus.train_loss : '—'}
                </span>
              </div>
              <div>
                <span className="text-[#81786E]">Val Loss:</span>{' '}
                <span className="font-mono font-semibold text-[#302721]">
                  {trainingStatus.val_loss !== null ? trainingStatus.val_loss : '—'}
                </span>
              </div>
              <div>
                <span className="text-[#81786E]">Val Accuracy:</span>{' '}
                <span className="font-mono font-semibold text-[#82917A]">
                  {trainingStatus.val_acc !== null ? `${trainingStatus.val_acc}%` : '—'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
