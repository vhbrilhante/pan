import React from 'react';
import { Activity, Cpu, Zap, Eye, Gauge, Server } from 'lucide-react';

export default function Telemetry({ telemetry, streamFps, classCounts, totalDetections }) {
  const inferFps = telemetry?.inference_fps || 0;
  const captureFps = telemetry?.capture_fps || 0;
  const totalLatency = telemetry?.total_latency_ms || 0;
  const prepMs = telemetry?.preprocess_ms || 0;
  const inferMs = telemetry?.inference_ms || 0;
  const postMs = telemetry?.postprocess_ms || 0;
  const backend = telemetry?.backend || "ultralytics";
  const device = telemetry?.device || "cpu";

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {/* Card 1: FPS Metrics */}
      <div className="p-4 rounded-xl glass-panel border border-cyber-border bg-cyber-card flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <Gauge className="w-4 h-4 text-cyber-accent" /> Throughput
          </span>
          <span className="text-cyber-green font-bold">{inferFps.toFixed(1)} FPS</span>
        </div>
        <div className="mt-3">
          <div className="text-2xl font-bold font-mono text-white flex items-baseline gap-2">
            {inferFps.toFixed(1)}
            <span className="text-xs font-normal text-slate-400 font-sans">Infer FPS</span>
          </div>
          <div className="flex items-center justify-between text-xs font-mono text-slate-400 mt-2 pt-2 border-t border-cyber-border/40">
            <span>Stream: {streamFps} FPS</span>
            <span>Cap: {captureFps.toFixed(0)} FPS</span>
          </div>
        </div>
      </div>

      {/* Card 2: Total Latency & Breakdown */}
      <div className="p-4 rounded-xl glass-panel border border-cyber-border bg-cyber-card flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <Zap className="w-4 h-4 text-cyber-warning" /> Latency
          </span>
          <span className="text-cyber-warning font-bold">{totalLatency.toFixed(1)} ms</span>
        </div>
        <div className="mt-2">
          <div className="text-2xl font-bold font-mono text-white">
            {totalLatency.toFixed(1)} <span className="text-xs font-normal text-slate-400 font-sans">ms</span>
          </div>
          {/* Latency Visual Bar */}
          <div className="w-full bg-cyber-bg h-2 rounded-full overflow-hidden flex mt-2">
            <div 
              style={{ width: `${Math.min(100, (prepMs / Math.max(1, totalLatency)) * 100)}%` }} 
              className="bg-blue-500 h-full" 
              title={`Prep: ${prepMs}ms`}
            />
            <div 
              style={{ width: `${Math.min(100, (inferMs / Math.max(1, totalLatency)) * 100)}%` }} 
              className="bg-cyber-accent h-full" 
              title={`Infer: ${inferMs}ms`}
            />
            <div 
              style={{ width: `${Math.min(100, (postMs / Math.max(1, totalLatency)) * 100)}%` }} 
              className="bg-purple-500 h-full" 
              title={`Post: ${postMs}ms`}
            />
          </div>
          <div className="flex justify-between text-[10px] font-mono text-slate-400 mt-1">
            <span className="text-blue-400">P:{prepMs}ms</span>
            <span className="text-cyber-accent">I:{inferMs}ms</span>
            <span className="text-purple-400">Post:{postMs}ms</span>
          </div>
        </div>
      </div>

      {/* Card 3: Backend & Hardware Engine */}
      <div className="p-4 rounded-xl glass-panel border border-cyber-border bg-cyber-card flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <Server className="w-4 h-4 text-purple-400" /> Runtime Engine
          </span>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-purple-500/20 text-purple-300 border border-purple-500/40">
            {backend}
          </span>
        </div>
        <div className="mt-3">
          <div className="text-lg font-bold font-mono text-slate-200 capitalize">
            {backend === 'onnxruntime' ? 'ONNX Runtime' : 'Ultralytics PyTorch'}
          </div>
          <div className="flex items-center gap-2 mt-2 pt-2 border-t border-cyber-border/40 text-xs font-mono text-slate-400">
            <Cpu className="w-3.5 h-3.5 text-cyber-accent" />
            <span>Target: <strong className="text-white uppercase">{device}</strong></span>
          </div>
        </div>
      </div>

      {/* Card 4: Detections Counter */}
      <div className="p-4 rounded-xl glass-panel border border-cyber-border bg-cyber-card flex flex-col justify-between">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="flex items-center gap-1.5">
            <Eye className="w-4 h-4 text-cyber-green" /> Detected Targets
          </span>
          <span className="text-cyber-green font-bold">{totalDetections} Total</span>
        </div>
        <div className="mt-2">
          <div className="text-2xl font-bold font-mono text-white">
            {totalDetections}
            <span className="text-xs font-normal text-slate-400 font-sans ml-2">Objects</span>
          </div>
          <div className="flex flex-wrap gap-1 mt-2 max-h-8 overflow-hidden">
            {Object.entries(classCounts).map(([cls, count]) => (
              <span key={cls} className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-cyber-bg text-cyber-accent border border-cyber-border">
                {cls}: {count}
              </span>
            ))}
            {Object.keys(classCounts).length === 0 && (
              <span className="text-[11px] font-mono text-slate-500">No objects in view</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
