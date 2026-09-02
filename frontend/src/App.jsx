import React, { useState, useEffect } from 'react';
import { useWebSocketStream } from './hooks/useWebSocketStream';
import VideoPlayer from './components/VideoPlayer';
import Telemetry from './components/Telemetry';
import ControlPanel from './components/ControlPanel';
import DetectionTable from './components/DetectionTable';
import { ShieldCheck, Cpu, HardDrive, Terminal, Radio } from 'lucide-react';

export default function App() {
  const {
    frameImage,
    metadata,
    telemetry,
    detections,
    classCounts,
    status,
    streamFps,
    error,
    reconnect,
  } = useWebSocketStream();

  const [systemMetrics, setSystemMetrics] = useState(null);

  // Poll hardware metrics from REST /api/v1/metrics
  useEffect(() => {
    const fetchMetrics = () => {
      fetch('/api/v1/metrics')
        .then(res => res.json())
        .then(data => setSystemMetrics(data))
        .catch(() => {});
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-cyber-bg text-slate-100 flex flex-col">
      {/* Top Navbar */}
      <header className="border-b border-cyber-border/70 bg-cyber-surface/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-cyber-accent/10 border border-cyber-accent/30 text-cyber-accent glow-accent">
              <Radio className="w-5 h-5 animate-pulse" />
            </div>
            <div>
              <h1 className="text-base font-bold font-mono tracking-tight text-white flex items-center gap-2">
                PANGIZ <span className="text-xs px-2 py-0.5 rounded bg-cyber-border text-cyber-accent">EDGE AI</span>
              </h1>
              <p className="text-[11px] text-slate-400 font-mono">YOLO Computer Vision Inference Engine</p>
            </div>
          </div>

          {/* Hardware & System Status Header Widgets */}
          <div className="hidden sm:flex items-center space-x-4 text-xs font-mono">
            {systemMetrics && (
              <>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyber-card border border-cyber-border text-slate-300">
                  <Cpu className="w-3.5 h-3.5 text-cyber-accent" />
                  <span>CPU: <strong className="text-white">{systemMetrics.cpu_percent}%</strong></span>
                </div>
                <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyber-card border border-cyber-border text-slate-300">
                  <HardDrive className="w-3.5 h-3.5 text-purple-400" />
                  <span>RAM: <strong className="text-white">{systemMetrics.memory_percent}%</strong></span>
                </div>
              </>
            )}

            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyber-card border border-cyber-border">
              <span className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-cyber-green animate-pulse' : 'bg-cyber-danger'}`} />
              <span className="uppercase text-slate-200 font-semibold">{status}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col gap-6">
        {/* Telemetry Metrics Bar */}
        <Telemetry 
          telemetry={telemetry}
          streamFps={streamFps}
          classCounts={classCounts}
          totalDetections={detections.length}
        />

        {/* Video Canvas & Live Stream Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Video Viewport (2 Cols on Large Screens) */}
          <div className="lg:col-span-2 flex flex-col">
            <VideoPlayer 
              frameImage={frameImage}
              status={status}
              metadata={metadata}
              error={error}
              onReconnect={reconnect}
            />
          </div>

          {/* Detections List & Objects Feed (1 Col) */}
          <div className="lg:col-span-1 flex flex-col">
            <DetectionTable detections={detections} />
          </div>
        </div>

        {/* Bottom Control & Dynamic Configuration */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-3">
            <ControlPanel />
          </div>
        </div>
      </main>

      {/* System Footer */}
      <footer className="border-t border-cyber-border/60 bg-cyber-surface/60 py-4 mt-auto text-center text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Edge Vision Engine Boilerplate &bull; FastAPI + Ultralytics YOLO &bull; ONNX Runtime</span>
          <div className="flex items-center gap-3 text-slate-400">
            <span className="flex items-center gap-1"><ShieldCheck className="w-3.5 h-3.5 text-cyber-green" /> Production Ready</span>
            <span className="flex items-center gap-1"><Terminal className="w-3.5 h-3.5 text-cyber-accent" /> WebSocket /ws/stream</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
