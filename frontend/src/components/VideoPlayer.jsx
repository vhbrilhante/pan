import React, { useRef, useState } from 'react';
import { Maximize2, Minimize2, Video, AlertCircle, RefreshCw } from 'lucide-react';

export default function VideoPlayer({ frameImage, status, metadata, error, onReconnect }) {
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(err => console.error(err));
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(err => console.error(err));
      setIsFullscreen(false);
    }
  };

  return (
    <div 
      ref={containerRef}
      className="relative rounded-2xl overflow-hidden glass-panel border border-cyber-border bg-cyber-surface flex flex-col justify-center items-center aspect-video w-full shadow-2xl group"
    >
      {/* Live Video Feed */}
      {frameImage && status === 'connected' ? (
        <img 
          src={frameImage} 
          alt="Live YOLO Edge Stream" 
          className="w-full h-full object-contain"
        />
      ) : (
        <div className="flex flex-col items-center justify-center p-8 text-center">
          {status === 'connecting' ? (
            <div className="flex flex-col items-center space-y-4">
              <RefreshCw className="w-12 h-12 text-cyber-accent animate-spin" />
              <p className="text-sm font-mono text-slate-300">Connecting to Edge Vision WebSocket stream...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center space-y-4">
              <div className="p-4 rounded-full bg-red-500/10 border border-red-500/30">
                <AlertCircle className="w-10 h-10 text-cyber-danger" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-slate-200">Stream Disconnected</h3>
                <p className="text-xs text-slate-400 max-w-sm mt-1">
                  {error || "Waiting for backend server at /ws/stream. Check if FastAPI backend is running."}
                </p>
              </div>
              <button
                onClick={onReconnect}
                className="px-4 py-2 text-xs font-mono bg-cyber-accent/20 hover:bg-cyber-accent/30 text-cyber-accent border border-cyber-accent/50 rounded-lg transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> Reconnect
              </button>
            </div>
          )}
        </div>
      )}

      {/* Top Left Overlay: Stream Status & Resolution */}
      <div className="absolute top-4 left-4 flex items-center gap-2 pointer-events-none z-10">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-xs font-mono">
          <span className={`w-2.5 h-2.5 rounded-full ${status === 'connected' ? 'bg-cyber-green animate-pulse' : 'bg-cyber-danger'}`} />
          <span className="uppercase text-slate-200 font-semibold">{status}</span>
        </div>

        {metadata && (
          <div className="px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-md border border-white/10 text-xs font-mono text-slate-300">
            {metadata.width}x{metadata.height} px
          </div>
        )}
      </div>

      {/* Top Right Overlay: Fullscreen Toggle */}
      <div className="absolute top-4 right-4 z-10 opacity-80 group-hover:opacity-100 transition-opacity">
        <button
          onClick={toggleFullscreen}
          className="p-2 rounded-lg bg-black/60 hover:bg-black/80 backdrop-blur-md border border-white/10 text-slate-300 hover:text-white transition-all"
          title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
        >
          {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
        </button>
      </div>

      {/* Bottom Overlay: Frame Counter Bar */}
      {metadata && (
        <div className="absolute bottom-3 left-4 right-4 flex items-center justify-between pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity text-xs font-mono text-slate-400 bg-black/60 backdrop-blur-md px-4 py-1.5 rounded-xl border border-white/10">
          <span>Frame #{metadata.frame_id}</span>
          <span>Detections: {metadata.detections?.length || 0}</span>
        </div>
      )}
    </div>
  );
}
