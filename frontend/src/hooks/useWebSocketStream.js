import { useState, useEffect, useRef, useCallback } from 'react';

/**
 * Custom React Hook for resilient, low-latency WebSocket video streaming.
 * Handles auto-reconnect, client-side streaming FPS, and frame deserialization.
 */
export function useWebSocketStream(url) {
  const [frameImage, setFrameImage] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [status, setStatus] = useState('disconnected'); // 'connecting' | 'connected' | 'disconnected'
  const [streamFps, setStreamFps] = useState(0);
  const [error, setError] = useState(null);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const shouldReconnectRef = useRef(true);

  // Compute WebSocket URL based on current host if relative
  const resolveWsUrl = useCallback(() => {
    if (url) return url;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    // When running with Vite dev server (port 3000) or proxy
    const port = window.location.port === '3000' ? '8000' : window.location.port || '8000';
    return `${protocol}//${host}:${port}/ws/stream`;
  }, [url]);

  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsUrl = resolveWsUrl();
    setStatus('connecting');
    setError(null);

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        setError(null);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = performance.now();
      };

      ws.onmessage = (event) => {
        try {
          const packet = JSON.parse(event.data);
          if (packet.type === 'frame') {
            setFrameImage(packet.image);
            setMetadata(packet.metadata);

            // Compute client-side rendering FPS
            frameCountRef.current += 1;
            const now = performance.now();
            const elapsed = now - lastFpsTimeRef.current;
            if (elapsed >= 1000) {
              setStreamFps(Math.round((frameCountRef.current * 1000) / elapsed));
              frameCountRef.current = 0;
              lastFpsTimeRef.current = now;
            }
          }
        } catch (err) {
          console.error("Failed to parse WebSocket packet:", err);
        }
      };

      ws.onclose = () => {
        setStatus('disconnected');
        if (shouldReconnectRef.current) {
          reconnectTimeoutRef.current = setTimeout(connect, 2000);
        }
      };

      ws.onerror = (err) => {
        setError("WebSocket connection failed. Ensure backend is running.");
        ws.close();
      };
    } catch (e) {
      setStatus('disconnected');
      setError(e.message);
      if (shouldReconnectRef.current) {
        reconnectTimeoutRef.current = setTimeout(connect, 3000);
      }
    }
  }, [resolveWsUrl]);

  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    if (wsRef.current) wsRef.current.close();
    setStatus('disconnected');
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    shouldReconnectRef.current = true;
    connect();
  }, [disconnect, connect]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return {
    frameImage,
    metadata,
    telemetry: metadata?.telemetry || null,
    detections: metadata?.detections || [],
    classCounts: metadata?.class_counts || {},
    status,
    streamFps,
    error,
    reconnect,
    disconnect,
  };
}
