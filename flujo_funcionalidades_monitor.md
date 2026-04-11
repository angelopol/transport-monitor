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

## 2. Inyección Biométrica de Nube (SyncClient - Pre-Ruta)
Antes de siquiera grabar el primer rostro, `transport-monitor` debe resolver quién está manejando hoy:
* El módulo nativo `CloudSync` realiza una consulta segura al backend ('transport-admin') informando la Identidad (Token MAC) de esta unidad física.
* El servidor responde descargándole silenciosamente al script (`_sync_excluded_faces()`) las identidades y rostros registrados de los **Dueños, Conductores y Colectores** que la base de datos de la empresa tiene como habilitados.
* Estos rostros descargados alimentan al `FaceTracker` para blindar al sistema de no contabilizar falsamente al propio conductor mientras conduce la unidad.

## 3. El Bucle de Supervivencia (Event Loop de Monitoreo)
El orquestador invoca la función `run()`, iniciando el ciclo perpetuo infinito del viaje:
1. **Captura Visual Asíncrona:** Lee pasivamente un fotograma de RAM (gracias al sub-proceso de `VideoStream`).
2. **Descarte por Movimiento:** Interroga al `MotionDetector` para evitar disparar lógicas en una unidad estacionaria o vacía.
3. **Inferencia Local:** Extrae las siluetas de cabezas si detectó el paso de peatones mediante su detector local para ahorrar peticiones a la nube.
4. **Telemetría e Inferencia AWS (La decantación):** Se somete el rostro o frames seleccionados a la validación estricta de `FaceCounter` y se compara a la bio-restricción de `FaceTracker` evitando que el conductor genere +1 en los indicadores económicos.
5. **Generador del Payload:** Al declararse contundentemente cómo un **"Pasajero Nuevo"**, el `transport-monitor` empaqueta una entidad relacional construyendo un JSON con la hora exacta y las geo-coordenadas brindadas por su antena.

## 4. Persistencia Subterránea
Todas las validaciones en carretera son propensas a pérdida de señal. Para resolver esto:
* El script empuja el evento consolidado y sus variables al almacenamiento estático local (SQLite) y lo cataloga como `'Pendiente'`.

## 5. El Hilo del Despachador (Sync Loop)
De forma paralela (y controlada por un Thread de Python `_sync_loop()`), el sistema revisa a un ritmo distinto todo lo que se ha ido guardando estáticamente en la cabina.
1. Examina la calidad de la conexión con el servidor maestro.
2. Sí existe cobertura celular o WiFi, empaqueta los eventos `'Pendiente'` desde la base de datos local y los inyecta al endpoint de la API `/telemetry`.
3. Al recibir un acuse de recibo positivo `(HTTP 200 OK)`, limpia su propia base de datos, garantizando que el dueño visualice al instante la métrica en su "Portal Admin" sin gastar memoria o red repetida de la placa.

## 6. Hibernación Segura y Salida
El orquestador está diseñado para resistir re-arranques u órdenes directas del Kernel de Linux.
Si el bus se apaga o la placa recibe detención, el manejador de señales (`_signal_handler`) atiende los comandos SIGINT/SIGTERM orquestando un **"Apagado Limpio"** (`_shutdown()`):
* Apaga el lector de la lente previniendo corrupciones de video.
* Descarga y cierra las conexiones SQLite para no corromper la cola local de pasajeros.
* Escribe una última línea en su archivo local log alertando de que el hardware se desconectará del mundo real con gracia.
