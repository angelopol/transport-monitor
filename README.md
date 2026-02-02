# Transport Monitor

Sistema inteligente de monitoreo para transporte urbano que corre en dispositivos edge (Raspberry Pi). Captura video en tiempo real, detecta movimiento para filtrar frames irrelevantes, y cuenta rostros usando AWS Rekognition con resiliencia a fallas de red mediante Store-and-Forward con SQLite.

## Estructura del Proyecto

```
transport-monitor/
├── transport_monitor.py     # Script principal de orquestación
├── config.yaml              # Configuración centralizada
├── requirements.txt         # Dependencias Python
├── stream_count_faces/      # Paquete modular
│   ├── __init__.py
│   ├── camera.py            # VideoStream (captura threaded)
│   ├── motion.py            # MotionDetector (diferenciación frames)
│   ├── storage.py           # LocalBuffer (SQLite Store-and-Forward)
│   └── detector.py          # FaceCounter (AWS Rekognition wrapper)
├── data/                    # Base de datos SQLite (generado)
└── logs/                    # Archivos de log (generado)
```

## Instalación

```bash
# Clonar/navegar al repositorio
cd transport-monitor

# Crear entorno virtual (recomendado)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## Configuración

Editar `config.yaml` para ajustar:
- **camera**: Fuente de video, resolución, FPS
- **motion**: Umbrales de detección de movimiento
- **detector**: Umbrales de confianza para rostros
- **storage**: Ruta de base de datos, retención
- **aws**: Región de AWS Rekognition

### Credenciales AWS

Configurar credenciales de AWS de una de estas formas:

#### Opción 1: Archivo .env (recomendado)

```bash
cp .env.example .env
```

Editar `.env`:
```
AWS_ACCESS_KEY_ID=tu_access_key_aqui
AWS_SECRET_ACCESS_KEY=tu_secret_key_aqui
AWS_REGION=us-east-1
```

#### Opción 2: Variables de entorno del sistema

```bash
export AWS_ACCESS_KEY_ID=tu_access_key
export AWS_SECRET_ACCESS_KEY=tu_secret_key
```

#### Opción 3: AWS CLI

```bash
aws configure
```

#### Opción 4: Modo desarrollo (sin AWS)

```bash
python transport_monitor.py --dry-run
```

## Uso

```bash
# Modo normal (requiere AWS configurado)
python transport_monitor.py

# Modo desarrollo (sin llamadas a AWS)
python transport_monitor.py --dry-run --verbose

# Usar cámara específica
python transport_monitor.py --source 0

# Usar archivo de video
python transport_monitor.py --source video.mp4

# Configuración personalizada
python transport_monitor.py --config custom_config.yaml
```

### Argumentos de Línea de Comandos

| Argumento | Descripción |
|-----------|-------------|
| `--config, -c` | Ruta al archivo de configuración YAML |
| `--source, -s` | Fuente de video (índice o archivo) |
| `--dry-run` | Modo simulación sin AWS |
| `--verbose, -v` | Logging DEBUG |
| `--log-file` | Ruta al archivo de log |

## Componentes

### VideoStream
Captura de video no bloqueante usando threading. Incluye reconexión automática.

### MotionDetector
Filtra frames sin actividad usando diferenciación de frames y detección de contornos.

### LocalBuffer
Almacenamiento SQLite con patrón Store-and-Forward para resiliencia offline.

### FaceCounter
Wrapper para AWS Rekognition con filtrado de calidad (frontalidad, oclusión, confianza).

## Licencia

MIT License
