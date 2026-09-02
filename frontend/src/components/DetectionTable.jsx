import React from 'react';
import { Target, ListFilter } from 'lucide-react';

export default function DetectionTable({ detections }) {
  return (
    <div className="p-5 rounded-2xl glass-panel border border-cyber-border bg-cyber-surface flex flex-col h-full">
      <div className="flex items-center justify-between border-b border-cyber-border/60 pb-3 mb-3">
        <h3 className="text-sm font-semibold font-mono text-white flex items-center gap-2">
          <Target className="w-4 h-4 text-cyber-accent" /> Live Detection Stream
        </h3>
        <span className="text-xs font-mono text-slate-400">
          {detections.length} {detections.length === 1 ? 'target' : 'targets'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto max-h-[360px] pr-1 space-y-2">
        {detections.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-12 text-center text-slate-500 font-mono text-xs">
            <ListFilter className="w-8 h-8 text-slate-600 mb-2 opacity-50" />
            No targets detected in current frame
          </div>
        ) : (
          detections.map((det, idx) => {
            const confPct = Math.round(det.confidence * 100);
            return (
              <div 
                key={`${det.class_name}-${idx}`} 
                className="p-2.5 rounded-xl bg-cyber-card border border-cyber-border/80 flex items-center justify-between text-xs font-mono hover:border-cyber-accent/40 transition-all"
              >
                <div className="flex items-center gap-2.5">
                  <span className="w-2 h-2 rounded-full bg-cyber-accent" />
                  <div>
                    <div className="font-semibold text-slate-100 capitalize">{det.class_name}</div>
                    <div className="text-[10px] text-slate-400">
                      [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}] - [{det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="w-16 bg-cyber-bg h-1.5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full ${confPct > 70 ? 'bg-cyber-green' : 'bg-cyber-warning'}`}
                      style={{ width: `${confPct}%` }}
                    />
                  </div>
                  <span className="font-bold text-slate-200 min-w-[32px] text-right">{confPct}%</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
