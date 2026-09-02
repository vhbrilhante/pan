# ⚡ Edge AI Computer Vision Engine with YOLO & WebSockets

Boilerplate de arquitetura de alta performance para **Visão Computacional na Borda (Edge AI)** utilizando **Python 3.11**, **FastAPI**, **Ultralytics YOLO (v8/v11)**, **ONNX Runtime**, **WebSockets** e um **Dashboard React (Vite) + TailwindCSS**.

---

## 🏛️ Visão Geral da Arquitetura

```
+-------------------------------------------------------------------------------+
|                             Edge Device / Host                                |
|                                                                               |
|  +-------------------------------------------------------------------------+  |
|  |                     YOLOInferenceEngine (Thread-Safe)                   |  |
|  |                                                                         |  |
|  |  [Camera/RTSP] -> [Capture Thread (Buffer LIFO/Drop)]                   |  |
|  |                        |                                                |  |
|  |                        v                                                |  |
|  |        [Inference Pipeline (PyTorch / ONNX Runtime)]                    |  |
|  |             - Pre-process -> Infer -> Post-process (NMS)               |  |
|  |             - Performance Telemetry (FPS, Latency, Stats)               |  |
|  +-----------------------------------|-------------------------------------+  |
|                                      |                                        |
|                                      v                                        |
|  +-------------------------------------------------------------------------+  |
|  |                        FastAPI Backend Server                           |  |
|  |                                                                         |  |
|  |  - REST API: /api/v1/health, /api/v1/config, /api/v1/metrics            |  |
|  |  - WebSocket: /ws/stream (Real-time Video Feed + JSON Detections)       |  |
|  +-----------------------------------|-------------------------------------+  |
+--------------------------------------|----------------------------------------+
                                       | WebSocket / HTTP
                                       v
+-------------------------------------------------------------------------------+
|                 Frontend Dashboard (React + Vite + TailwindCSS)                |
|                                                                               |
|  - Real-time Video Stream / Canvas Overlay                                    |
|  - Live Detections & Bounding Box Telemetry                                   |
|  - Hardware & Latency Telemetry (FPS, Inference ms, CPU/RAM)                  |
|  - Interactive Controls (Confidence/IOU Sliders, Class Filters)               |
+-------------------------------------------------------------------------------+
```

---

## 📂 Estrutura de Diretórios

```
pangiz/
├── api/                        # Camada de Apresentação e Comunicação
│   ├── __init__.py
│   ├── app.py                  # Ponto de entrada FastAPI com Lifespan Context Manager
│   ├── routes/
│   │   ├── health.py           # Endpoints de status e telemetria de hardware (CPU/RAM/GPU)
│   │   └── control.py          # Endpoints de configuração dinâmica (confiança, IOU, classes)
│   └── websocket/
│       └── stream_handler.py   # Gerenciador de conexões WebSocket e streaming de vídeo/telemetria
├── engine/                     # Núcleo de Visão Computacional e Inferência
│   ├── __init__.py
│   ├── capture.py              # Thread de captura de vídeo com buffer de descarte (Zero-latency)
│   ├── inference.py            # YOLOInferenceEngine (Thread-safe, PyTorch & ONNX Runtime)
│   ├── models.py               # Schemas Pydantic para detecções, bbox e telemetria
│   └── utils.py                # Anotações visuais OpenCV e compressão JPEG
├── frontend/                   # Dashboard Web React + Vite + TailwindCSS
│   ├── src/
│   │   ├── components/         # VideoPlayer, Telemetry, ControlPanel, DetectionTable
│   │   ├── hooks/              # useWebSocketStream (auto-reconnect e cálculo de FPS)
│   │   ├── App.jsx             # Layout principal Cyber-Edge Dark
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── scripts/
│   ├── export.py               # Exportador PyTorch (.pt) para ONNX otimizado (FP16/Dynamic)
│   └── benchmark.py            # Benchmark de inferência (PyTorch vs ONNX Runtime)
├── config.py                   # Configurações com Pydantic Settings
├── requirements.txt            # Dependências Python travadas
├── Dockerfile                  # Multi-stage build otimizado para Edge
├── docker-compose.yml          # Orquestração do Backend + Frontend
└── .env.example                # Template de variáveis de ambiente
```

---

## 🚀 Como Executar Localmente

### 1. Pré-requisitos
- Python 3.11+
- Node.js 18+ e npm

### 2. Backend (FastAPI + YOLO)

```bash
# 1. Crie e ative o ambiente virtual
python -m venv venv
# No Windows:
.\venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure as variáveis de ambiente
copy .env.example .env

# 4. Inicie o servidor FastAPI
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

> **Nota:** Se nenhuma câmera física estiver conectada, o sistema aciona automaticamente o **Mock Camera Generator** para geração de frames sintéticos dinâmicos.

### 3. Frontend Dashboard (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

Acesse o dashboard em: [http://localhost:3000](http://localhost:3000)  
Documentação da API Swagger: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Executando com Docker Compose

```bash
docker-compose up --build
```

---

## 🔄 Exportação e Otimização para ONNX

Para obter a máxima taxa de FPS em dispositivos embarcados (Jetson, Intel NUC, Raspberry Pi):

```bash
# Exportação básica para ONNX
python scripts/export.py --model yolov8n.pt --imgsz 640

# Exportação com precisão FP16 (Recomendado para GPU/Jetson)
python scripts/export.py --model yolov8n.pt --half --device 0

# Executar benchmark de desempenho
python scripts/benchmark.py --model yolov8n.onnx --iterations 100
```

Após exportar, atualize seu arquivo `.env`:
```env
MODEL_PATH=yolov8n.onnx
INFERENCE_BACKEND=onnxruntime
```

---

## 📡 Endpoints da API

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/api/v1/health` | Status de saúde do serviço e backend ativo |
| `GET` | `/api/v1/metrics` | Telemetria de hardware (CPU %, RAM %, GPU %, Uptime) |
| `GET` | `/api/v1/config` | Configurações atuais do motor de inferência |
| `POST` | `/api/v1/config` | Ajuste dinâmico de limiares (confiança, IOU, classes) |
| `WS` | `/ws/stream` | Stream WebSocket de vídeo (JPEG) + Metadados JSON de detecção |

---

## 🛡️ Técnicas de Otimização na Borda (Edge Tuning)

1. **Zero-Latency Frame Dropping:** A classe `VideoCaptureThread` mantém um buffer atômico de tamanho 1 (`maxlen=1`). Quando a inferência leva mais tempo que a taxa de captura da câmera, frames desatualizados são descartados automaticamente, eliminando lag acumulado.
2. **Thread Safety:** `YOLOInferenceEngine` encapsula a sessão do modelo com travas de concorrência (`threading.Lock`), permitindo chamadas seguras a partir de loops assíncronos FastAPI.
3. **JPEG Compression Control:** O parâmetro `WS_JPEG_QUALITY` permite calibrar a largura de banda de rede vs qualidade visual diretamente pelo painel ou via API REST.
