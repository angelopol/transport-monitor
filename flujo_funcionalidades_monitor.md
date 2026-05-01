# Flujo Funcional: Transport Monitor (Edge Orchestrator)

El paquete `transport-monitor` es el eslabón final y ejecutable principal que consolida todas las lógicas y se despliega como un DEMONIO (Servicio) directamente sobre la unidad de hardware (Raspberry Pi / Jetson) empotrada en el autobús.

Mientras que otros paquetes como `stream-count-faces` proveen las librerías matemáticas y lógicas puras, **`transport-monitor` orquesta secuencialmente estos recursos** de cara a una ejecución estabilizada, en tiempo real y conectada con la nube administrativa.

A continuación, se detalla el ciclo de vida y flujo funcional dictado por su script organizador `transport_monitor.py`.

---

## 1. Bootstrapping y Configuración (Arranque)
El ciclo inicia al prenderse la placa de hardware empotrada en la buseta, activándose el servicio del orquestador:
1. **Argumentos y Configuración Inicial:** El orquestador extrae los parámetros base dictados en `config.yaml` o mediante un archivo `.env`, configurando el puerto del hardware de la cámara (Ej. `/dev/video0`), los márgenes geométricos de la detección, credenciales encriptadas y URLs del servidor administrativo (`transport-admin`).
2. **Setup de Logs:** Instancia y rotula el registro de actividades locales (`setup_logging()`) dejando trazas explícitas e imprimibles bajo el directorio `logs/` para auditorias técnicas o fallos lógicos abordo.
3. **Instanciación de Módulos Core (`_init_components`):** Se instancian en RAM todas las sub-librerías (Cámara asíncrona, Detector de movimiento perimetral, Reconocimiento AWS, SQLite Local y Geolocalizador).

## 2. Autenticación y Sincronización Biométrica (SyncClient)
Antes de iniciar la captura, el dispositivo debe ser validado por el servidor:
1. **Autenticación Obligatoria:** El monitor intenta autenticarse usando su MAC Address. Si el dispositivo no está vinculado a un autobús activo en `transport-admin`, el script entra en un bucle de espera (Retry cada 30s) hasta que el administrador complete el emparejamiento.
2. **Descarga de Rostros Excluidos:** Una vez autenticado, descarga las fotos de **Conductores y Colectores** asignados a la empresa (`_sync_excluded_faces()`).
3. **Persistencia de Exclusión:** Estos rostros alimentan al `FaceTracker`, permitiendo que el personal autorizado se mueva por la cabina sin generar falsos positivos en el conteo de pasajeros.

## 3. El Bucle de Supervivencia (Event Loop de Monitoreo)
El orquestador invoca la función `run()`, iniciando el ciclo perpetuo infinito del viaje:
1. **Captura Visual Asíncrona:** Lee pasivamente un fotograma de RAM (gracias al sub-proceso de `VideoStream`).
2. **Descarte por Movimiento:** Interroga al `MotionDetector` para evitar procesar frames innecesarios en unidades estacionarias.
3. **Validación de Calidad Facial:** Al detectar un rostro, se evalúan umbrales de **Nitidez (Sharpness)** y **Tamaño Mínimo** para descartar detecciones accidentales o borrosas por vibración.
4. **Telemetría e Inferencia AWS:** Se somete el frame a la validación de `FaceCounter` y se compara con la base de datos de exclusión.
5. **Generador del Payload:** Si es un "Pasajero Nuevo", se crea un JSON con el `event_timestamp`, el conteo y las geo-coordenadas (GPS o Fallback por IP).

## 4. Persistencia Subterránea
Todas las validaciones en carretera son propensas a pérdida de señal. Para resolver esto:
* El script empuja el evento consolidado y sus variables al almacenamiento estático local (SQLite) y lo cataloga como `'Pendiente'`.

## 5. El Hilo del Despachador (Sync Loop)
De forma paralela (y controlada por un Thread de Python `_sync_loop()`), el sistema revisa a un ritmo distinto todo lo que se ha ido guardando estáticamente en la cabina.
1. Examina la calidad de la conexión con el servidor maestro.
2. Sí existe cobertura celular o WiFi, empaqueta los eventos `'Pendiente'` desde la base de datos local y los inyecta al endpoint de la API `/telemetry`.
3. Al recibir un acuse de recibo positivo `(HTTP 200 OK)`, limpia su propia base de datos.
4. **Heartbeat Proactivo:** Si no hay eventos que sincronizar, el hilo envía un "Latido" periódico al servidor incluyendo la ubicación GPS actual para el rastreo en tiempo real del bus en el mapa administrativo.

## 6. Hibernación Segura y Salida
El orquestador está diseñado para resistir re-arranques u órdenes directas del Kernel de Linux.
Si el bus se apaga o la placa recibe detención, el manejador de señales (`_signal_handler`) atiende los comandos SIGINT/SIGTERM orquestando un **"Apagado Limpio"** (`_shutdown()`):
* Apaga el lector de la lente previniendo corrupciones de video.
* Descarga y cierra las conexiones SQLite para no corromper la cola local de pasajeros.
* Escribe una última línea en su archivo local log alertando de que el hardware se desconectará del mundo real con gracia.
