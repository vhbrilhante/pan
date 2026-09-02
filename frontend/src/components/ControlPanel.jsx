import React, { useState, useEffect } from 'react';
import { Sliders, CheckCircle2, RotateCcw, Filter, Settings2, Save } from 'lucide-react';

export default function ControlPanel({ onConfigUpdated }) {
  const [config, setConfig] = useState({
    confidence_threshold: 0.45,
    iou_threshold: 0.45,
    jpeg_quality: 75,
    target_fps: 30,
    active_classes: null,
    available_classes: [],
    model_path: '',
    backend: '',
    camera_source: '',
    is_mock_camera: false,
  });

  const [saving, setSaving] = useState(false);
  const [selectedClasses, setSelectedClasses] = useState([]);
  const [classFilterActive, setClassFilterActive] = useState(false);

  // Fetch initial config from backend
  useEffect(() => {
    fetch('/api/v1/config')
      .then(res => res.json())
      .then(data => {
        setConfig(data);
        if (data.active_classes && data.active_classes.length > 0) {
          setSelectedClasses(data.active_classes);
          setClassFilterActive(true);
        } else {
          setSelectedClasses(data.available_classes || []);
          setClassFilterActive(false);
        }
      })
      .catch(err => console.error("Failed to load engine config:", err));
  }, []);

  const handleUpdate = async (overrides = {}) => {
    setSaving(true);
    const payload = {
      confidence_threshold: overrides.confidence_threshold !== undefined ? overrides.confidence_threshold : config.confidence_threshold,
      iou_threshold: overrides.iou_threshold !== undefined ? overrides.iou_threshold : config.iou_threshold,
      jpeg_quality: overrides.jpeg_quality !== undefined ? overrides.jpeg_quality : config.jpeg_quality,
      target_fps: overrides.target_fps !== undefined ? overrides.target_fps : config.target_fps,
      active_classes: overrides.active_classes !== undefined ? overrides.active_classes : (classFilterActive ? selectedClasses : null),
    };

    try {
      const res = await fetch('/api/v1/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.status === 'success') {
        setConfig(prev => ({ ...prev, ...payload }));
        if (onConfigUpdated) onConfigUpdated(payload);
      }
    } catch (e) {
      console.error("Failed to update config:", e);
    } finally {
      setSaving(false);
    }
  };

  const toggleClass = (cls) => {
    let next;
    if (selectedClasses.includes(cls)) {
      next = selectedClasses.filter(c => c !== cls);
    } else {
      next = [...selectedClasses, cls];
    }
    setSelectedClasses(next);
    if (classFilterActive) {
      handleUpdate({ active_classes: next });
    }
  };

  return (
    <div className="p-5 rounded-2xl glass-panel border border-cyber-border bg-cyber-surface flex flex-col gap-5">
      <div className="flex items-center justify-between border-b border-cyber-border/60 pb-3">
        <h3 className="text-sm font-semibold font-mono text-white flex items-center gap-2">
          <Sliders className="w-4 h-4 text-cyber-accent" /> Edge Engine Controls
        </h3>
        {config.is_mock_camera && (
          <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-amber-500/20 text-amber-300 border border-amber-500/40">
            Mock Camera Active
          </span>
        )}
      </div>

      {/* Slider: Confidence Threshold */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs font-mono text-slate-300">
          <span>Confidence Threshold</span>
          <span className="text-cyber-accent font-bold">{(config.confidence_threshold * 100).toFixed(0)}%</span>
        </div>
        <input 
          type="range" 
          min="0.05" 
          max="0.95" 
          step="0.05"
          value={config.confidence_threshold}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setConfig(prev => ({ ...prev, confidence_threshold: val }));
          }}
          onMouseUp={() => handleUpdate()}
          onTouchEnd={() => handleUpdate()}
          className="w-full h-1.5 bg-cyber-card rounded-lg appearance-none cursor-pointer accent-cyber-accent"
        />
      </div>

      {/* Slider: IOU / NMS Threshold */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs font-mono text-slate-300">
          <span>NMS / IOU Threshold</span>
          <span className="text-cyber-accent font-bold">{(config.iou_threshold * 100).toFixed(0)}%</span>
        </div>
        <input 
          type="range" 
          min="0.10" 
          max="0.90" 
          step="0.05"
          value={config.iou_threshold}
          onChange={(e) => {
            const val = parseFloat(e.target.value);
            setConfig(prev => ({ ...prev, iou_threshold: val }));
          }}
          onMouseUp={() => handleUpdate()}
          onTouchEnd={() => handleUpdate()}
          className="w-full h-1.5 bg-cyber-card rounded-lg appearance-none cursor-pointer accent-cyber-accent"
        />
      </div>

      {/* Slider: Streaming JPEG Quality */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs font-mono text-slate-300">
          <span>JPEG Stream Quality</span>
          <span className="text-slate-400 font-bold">{config.jpeg_quality}%</span>
        </div>
        <input 
          type="range" 
          min="20" 
          max="95" 
          step="5"
          value={config.jpeg_quality}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            setConfig(prev => ({ ...prev, jpeg_quality: val }));
          }}
          onMouseUp={() => handleUpdate()}
          onTouchEnd={() => handleUpdate()}
          className="w-full h-1.5 bg-cyber-card rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
      </div>

      {/* Class Filter Toggle & Selector */}
      <div className="space-y-2 pt-2 border-t border-cyber-border/40">
        <div className="flex items-center justify-between">
          <span className="text-xs font-mono text-slate-300 flex items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-cyber-accent" /> Class Filtering
          </span>
          <button
            onClick={() => {
              const nextState = !classFilterActive;
              setClassFilterActive(nextState);
              handleUpdate({ active_classes: nextState ? selectedClasses : null });
            }}
            className={`px-2 py-0.5 text-[10px] font-mono rounded transition-all ${
              classFilterActive 
                ? 'bg-cyber-accent text-black font-bold' 
                : 'bg-cyber-card text-slate-400 border border-cyber-border'
            }`}
          >
            {classFilterActive ? 'Active' : 'All Enabled'}
          </button>
        </div>

        {classFilterActive && (
          <div className="flex flex-wrap gap-1 max-h-36 overflow-y-auto p-2 bg-cyber-bg/70 rounded-xl border border-cyber-border">
            {config.available_classes?.slice(0, 30).map((cls) => {
              const active = selectedClasses.includes(cls);
              return (
                <button
                  key={cls}
                  onClick={() => toggleClass(cls)}
                  className={`px-2 py-0.5 text-[10px] font-mono rounded transition-all ${
                    active
                      ? 'bg-cyber-accent/20 text-cyber-accent border border-cyber-accent/50 font-medium'
                      : 'bg-cyber-card text-slate-500 border border-transparent'
                  }`}
                >
                  {cls}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Read-Only System Specs */}
      <div className="pt-2 border-t border-cyber-border/40 text-[11px] font-mono text-slate-400 space-y-1">
        <div className="flex justify-between">
          <span>Model File:</span>
          <span className="text-slate-200">{config.model_path}</span>
        </div>
        <div className="flex justify-between">
          <span>Video Source:</span>
          <span className="text-slate-200 truncate max-w-[180px]">{config.camera_source}</span>
        </div>
      </div>
    </div>
  );
}
