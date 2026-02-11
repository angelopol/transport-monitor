#!/usr/bin/env python3
"""
transport_monitor.py - Script principal de monitoreo de transporte

Este script es el orquestador principal que corre en el dispositivo edge
(Raspberry Pi) dentro del autobús. Integra todos los componentes para:
1. Capturar video en tiempo real
2. Detectar movimiento para filtrar frames irrelevantes
3. Contar rostros usando AWS Rekognition
4. Almacenar eventos localmente con resiliencia a fallas de red

Uso:
    python transport_monitor.py
    python transport_monitor.py --config custom_config.yaml
    python transport_monitor.py --source 0 --dry-run --verbose
"""

import argparse
import logging
import os
import cv2
import requests

import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


from sync_client import CloudSync
from stream_count_faces import (
    VideoStream,
    MotionDetector,
    FaceCounter,
    FaceTracker,
    LocalBuffer,
    LocationProvider,
    PassengerEventStore,
    extract_face_image,
    get_device_mac
)

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configura el sistema de logging.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Ruta al archivo de log (opcional)
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Force clear existing handlers to ensure our config applies
    logging.root.handlers = []
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def load_config(config_path: str) -> dict:
    """
    Carga configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logging.info(f"Configuración cargada desde: {config_path}")
            return config
    except FileNotFoundError:
        logging.warning(f"Archivo de configuración no encontrado: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error al parsear configuración YAML: {e}")
        return {}


def get_default_config() -> dict:
    """
    Retorna configuración por defecto.
    
    Returns:
        Diccionario con configuración por defecto
    """
    return {
        "camera": {
            "source": 0,
            "width": 1280,
            "height": 720,
            "target_fps": 5
        },
        "motion": {
            "min_area": 15000,
            "threshold": 50,
            "blur_kernel": 21,
            "cooldown_frames": 5
        },
        "detector": {
            "face_confidence_threshold": 90,
            "face_occluded_threshold": 80,
            "frontal_threshold": 35,
            "dry_run": False,
            "min_face_size": 50,
            "blur_threshold": 80.0
        },
        "storage": {
            "database_path": "data/transport_events.db",
            "retention_days": 30
        },
        "aws": {
            "region": "us-east-1"
        },
        "tracking": {
            "enabled": True,
            "ttl_minutes": 180,  # 3 horas
            "similarity_threshold": 80.0,
            "max_tracked_faces": 500
        },
        "system": {
            "loop_delay": 0.1,
            "verbose": False,
            "log_level": "INFO",
            "log_file": None
        }
    }


class TransportMonitor:
    """
    Orquestador principal del sistema de monitoreo.
    
    Integra todos los componentes y gestiona el ciclo principal
    de captura, detección y almacenamiento de eventos.
    """
    
    def __init__(self, config: dict):
        """
        Inicializa el monitor de transporte.
        
        Args:
            config: Diccionario con la configuración del sistema
        """
        self.config = config
        self.running = False
        self.logger = logging.getLogger(__name__)
        
        # Estadísticas
        self.stats = {
            "start_time": None,
            "frames_processed": 0,
            "motion_detected_count": 0,
            "faces_detected_total": 0,
            "new_passengers": 0,
            "duplicate_passengers": 0,
            "excluded_detected": 0,
            "events_saved": 0
        }
        
        # Inicializar componentes
        self._init_components()
        
        # Configurar manejadores de señales
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_components(self) -> None:
        """Inicializa todos los componentes del sistema."""
        cam_config = self.config.get("camera", {})
        motion_config = self.config.get("motion", {})
        detector_config = self.config.get("detector", {})
        storage_config = self.config.get("storage", {})
        aws_config = self.config.get("aws", {})
        tracking_config = self.config.get("tracking", {})
        
        # Geolocation configuration from environment
        self.location_enabled = os.getenv("ENABLE_LOCATION_TRACKING", "true").lower() == "true"
        self.ip_fallback_enabled = os.getenv("ENABLE_IP_FALLBACK", "true").lower() == "true"
        gps_serial_port = os.getenv("GPS_SERIAL_PORT", None)
        
        # VideoStream
        print(f"[3/8] Inicializando VideoStream (source={cam_config.get('source', 0)})...")
        self.video_stream = VideoStream(
            source=cam_config.get("source", 0),
            width=cam_config.get("width", 1280),
            height=cam_config.get("height", 720)
        )
        print("[3/8] VideoStream OK")
        
        # MotionDetector
        print("[4/8] Inicializando MotionDetector...")
        self.motion_detector = MotionDetector(
            min_area=motion_config.get("min_area", 5000),
            threshold=motion_config.get("threshold", 25),
            blur_kernel=motion_config.get("blur_kernel", 21),
            cooldown_frames=motion_config.get("cooldown_frames", 5)
        )
        print("[4/8] MotionDetector OK")
        
        # FaceCounter
        print(f"[5/8] Inicializando FaceCounter (dry_run={detector_config.get('dry_run', False)})...")
        self.face_counter = FaceCounter(
            face_confidence_threshold=detector_config.get("face_confidence_threshold", 90),
            face_occluded_threshold=detector_config.get("face_occluded_threshold", 80),
            frontal_threshold=detector_config.get("frontal_threshold", 35),
            dry_run=detector_config.get("dry_run", False),
            region=aws_config.get("region", "us-east-1")
        )
        print("[5/8] FaceCounter OK")
        
        # LocalBuffer
        print("[6/8] Inicializando LocalBuffer...")
        self.local_buffer = LocalBuffer(
            db_path=storage_config.get("database_path", "data/transport_events.db")
        )
        print("[6/8] LocalBuffer OK")
        
        # FaceTracker (deduplicación de pasajeros)
        print("[7/8] Inicializando FaceTracker...")
        self.tracking_enabled = tracking_config.get("enabled", True)
        if self.tracking_enabled:
            excluded_paths = tracking_config.get("excluded_faces_paths", [])
            offline_cache = tracking_config.get("offline_cache_path", None)
            self.face_tracker = FaceTracker(
                ttl_minutes=tracking_config.get("ttl_minutes", 180),
                similarity_threshold=tracking_config.get("similarity_threshold", 80.0),
                max_faces=tracking_config.get("max_tracked_faces", 500),
                excluded_faces=excluded_paths if excluded_paths else None,
                offline_cache_path=offline_cache,
                dry_run=detector_config.get("dry_run", False),
                region=aws_config.get("region", "us-east-1")
            )
            self.logger.info(
                f"Tracking habilitado: TTL={tracking_config.get('ttl_minutes', 180)} min, "
                f"Similitud={tracking_config.get('similarity_threshold', 80.0)}%, "
                f"Excluidos={len(excluded_paths)} rostros, "
                f"Offline cache={'sí' if offline_cache else 'no'}"
            )
        else:
            self.face_tracker = None
            self.logger.info("Tracking de pasajeros deshabilitado")
        print("[7/8] FaceTracker OK")
        
        # LocationProvider y PassengerEventStore (geolocalización)
        if self.location_enabled:
            self.location_provider = LocationProvider(
                serial_port=gps_serial_port,
                use_ip_fallback=self.ip_fallback_enabled
            )
            self.passenger_store = PassengerEventStore(
                db_path=storage_config.get("passenger_events_path", "data/passenger_events.db")
            )
            self.logger.info(
                f"Geolocalización habilitada: GPS={gps_serial_port or 'auto'}, "
                f"IP fallback={'sí' if self.ip_fallback_enabled else 'no'}"
            )
        else:
            self.location_provider = None
            self.passenger_store = None
            self.logger.info("Geolocalización deshabilitada")
            
        # Cloud Sync initialization
        print(f"[7.5/8] Inicializando Cloud Sync (SYNC_ENABLED={os.getenv('SYNC_ENABLED')})...")
        sync_config = self.config.get("sync", {})
        
        # Override with environment variables
        if os.getenv("SYNC_ENABLED", "").lower() == "true":
            sync_config["enabled"] = True
        if os.getenv("SYNC_API_URL"):
            sync_config["api_url"] = os.getenv("SYNC_API_URL")
        if os.getenv("SYNC_API_TOKEN"):
            sync_config["api_token"] = os.getenv("SYNC_API_TOKEN")
            
        # Device MAC: Try to get from hardware first, fallback to env/config
        try:
            device_mac = get_device_mac()
            if not device_mac or "unknown" in device_mac.lower():
                raise ValueError("Invalid hardware MAC")
        except Exception as e:
            self.logger.warning(f"Could not get hardware MAC: {e}. Using fallback.")
            device_mac = os.getenv("DEVICE_MAC", self.config.get("device", {}).get("mac", "unknown"))
        print(f"[7.5/8] Device MAC: {device_mac}")

        if sync_config.get("enabled", False):
            self.cloud_sync = CloudSync(
                api_url=sync_config.get("api_url", "http://localhost:8000/api/v1"),
                api_token="",  # Always start with empty token, let auth flow set it
                device_mac=device_mac
            )
            
            # Always authenticate via device/auth endpoint
            self.logger.info(f"Intentando autenticación para MAC: {device_mac}")
            print(f"[7.6/8] Autenticando dispositivo {device_mac}...", flush=True)
            retry_interval = 30  # seconds
            
            while True:
                auth_result = self.cloud_sync.authenticate_device()
                status = auth_result.get('status', 'error')
                token = auth_result.get('token')
                
                if token:
                    print(f"[7.7/8] Token del dispositivo obtenido.", flush=True)
                
                if status == 'authenticated':
                    self.logger.info(f"Dispositivo autenticado y vinculado a bus")
                    device_info = auth_result.get('device', {})
                    print(f"✅ Dispositivo autenticado. Vinculado a bus: {device_info.get('plate', '?')}", flush=True)
                    break
                
                elif status in ('registered', 'pending'):
                    print(f"⏳ {auth_result.get('message')}", flush=True)
                    print(f"   Esperando vinculación con un autobús... (reintento en {retry_interval}s)", flush=True)
                    self.logger.info(f"Device {status}: waiting for bus link. Retry in {retry_interval}s")
                    time.sleep(retry_interval)
                    continue
                
                elif status == 'inactive':
                    print(f"❌ Dispositivo inactivo: {auth_result.get('message')}", flush=True)
                    self.logger.error("Device is inactive. Cannot proceed.")
                    raise SystemExit(1)
                
                else:  # error
                    self.logger.warning(f"Auth error: {auth_result.get('message')}. Retry in {retry_interval}s")
                    print(f"⚠️  Error: {auth_result.get('message')}", flush=True)
                    print(f"   Reintentando en {retry_interval} segundos...", flush=True)
                    time.sleep(retry_interval)
                    continue
            
            self.sync_interval = sync_config.get("interval", 60)
            self.stop_sync = threading.Event()
            self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
            self.sync_thread.start()
            self.logger.info(f"Cloud Sync habilitado. Intervalo: {self.sync_interval}s")
        else:
            self.cloud_sync = None
            self.logger.info("Cloud Sync deshabilitado")
        
        self.logger.info("Componentes inicializados correctamente")
    
    def _signal_handler(self, signum, frame) -> None:
        """Manejador de señales para graceful shutdown."""
        self.logger.info(f"Señal recibida ({signum}), iniciando apagado...")
        self.running = False
    
    def _create_face_event(self, face_count: int) -> dict:
        """
        Crea un evento de detección de rostros.
        
        Args:
            face_count: Número de rostros detectados
            
        Returns:
            Diccionario con datos del evento
        """
        return {
            "count": face_count,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "device_id": "transport_monitor_001",  # TODO: Hacer configurable
            "location": {
                "lat": None,  # TODO: Integrar GPS
                "lon": None,
                "route": "default_route"  # TODO: Hacer configurable
            }
        }
    
    def run(self) -> None:
        """
        Ejecuta el ciclo principal de monitoreo.
        
        El bucle principal:
        1. Lee frame del stream
        2. Verifica movimiento
        3. Si hay movimiento, detecta rostros
        4. Si hay rostros, guarda evento
        5. Duerme brevemente para controlar CPU
        """
        self.running = True
        self.stats["start_time"] = datetime.now().isoformat()
        loop_delay = self.config.get("system", {}).get("loop_delay", 0.1)
        
        self.logger.info("=" * 50)
        self.logger.info("TRANSPORT MONITOR - INICIANDO")
        self.logger.info("=" * 50)
        self.logger.info(f"Configuración: {self.config.get('system', {})}")
        
        # Iniciar stream de video
        self.video_stream.start()
        
        # Esperar a que el stream se estabilice
        time.sleep(1.0)
        
        if not self.video_stream.is_running():
            self.logger.error("No se pudo iniciar el stream de video")
            return
        
        self.logger.info("Stream de video activo, iniciando bucle principal...")
        self.logger.info("Entrando al bucle while...")
        
        # Initialize debug window if needed
        if self.logger.isEnabledFor(logging.DEBUG):
            cv2.namedWindow("Transport Monitor Debug", cv2.WINDOW_NORMAL)
            self.logger.debug("Visual Debug Window Initialized")

        try:
            while self.running:
                # 1. Leer frame
                frame = self.video_stream.read()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.stats["frames_processed"] += 1
                


                # Heartbeat log every 300 frames (approx 10-30s depending on fps)
                if self.stats["frames_processed"] % 300 == 0:
                    fc_stats = self.face_counter.get_stats()
                    last_err = fc_stats.get("last_error")
                    err_msg = f" | Last Error: {last_err}" if last_err else ""
                    self.logger.info(f"[HEARTBEAT] Frames: {self.stats['frames_processed']} | Motion: {self.stats['motion_detected_count']} | Faces: {self.stats['faces_detected_total']}{err_msg}")
                
                # 2. Verificar movimiento
                motion_detected = self.motion_detector.detect(frame)
                
                faces = []
                face_count = 0

                if motion_detected:
                    self.stats["motion_detected_count"] += 1
                    self.logger.debug("Movimiento detectado, analizando rostros...")
                    
                    # 3. Detectar rostros
                    faces = self.face_counter.count_faces(frame)
                    face_count = len(faces)

                # Visual Debug
                if self.logger.isEnabledFor(logging.DEBUG):
                    debug_frame = frame.copy()
                    
                    if face_count > 0:
                        debug_frame = self.face_counter.draw_faces(debug_frame, faces)
                    
                    color = (0, 255, 0) if motion_detected else (0, 0, 255)
                    cv2.putText(debug_frame, f"Motion: {motion_detected}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    cv2.imshow("Transport Monitor Debug", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                if not motion_detected:
                    time.sleep(loop_delay)
                    continue
                
                if face_count == 0:
                    time.sleep(loop_delay)
                    continue
                
                self.stats["faces_detected_total"] += face_count
                self.logger.debug(f"Rostros detectados en frame: {face_count}")
                
                # 4. Filtrar pasajeros duplicados (si tracking está habilitado)
                new_passengers = []
                if self.tracking_enabled and self.face_tracker:
                    frame_height, frame_width = frame.shape[:2]
                    min_face_size = self.config["detector"].get("min_face_size", 50)
                    blur_threshold = self.config["detector"].get("blur_threshold", 80.0)

                    for face in faces:
                        # 1. Check face size
                        face_w = face.bounding_box['Width'] * frame_width
                        face_h = face.bounding_box['Height'] * frame_height
                        if face_w < min_face_size or face_h < min_face_size:
                            self.logger.debug(f"Face too small ({int(face_w)}x{int(face_h)} < {min_face_size}), skipping")
                            continue

                        # 2. Check sharpness (blur)
                        sharpness = face.quality.get('sharpness', 0)
                        if sharpness < blur_threshold:
                            self.logger.debug(f"Face too blurry (sharpness {sharpness:.1f} < {blur_threshold}), skipping")
                            continue

                        try:
                            # Extraer imagen del rostro
                            face_image = extract_face_image(frame, face.bounding_box)
                            
                            # Verificar si es nuevo pasajero
                            # Retorna: (is_new, face_id, is_excluded)
                            is_new, face_id, is_excluded = self.face_tracker.is_new_passenger(face_image)
                            
                            if is_excluded:
                                # Personal autorizado (operador, conductor)
                                self.stats["excluded_detected"] += 1
                                self.logger.debug("Personal autorizado detectado, ignorando")
                            elif is_new:
                                new_passengers.append(face)
                                self.stats["new_passengers"] += 1
                                
                                # DEBUG: Save face image for inspection
                                if self.logger.isEnabledFor(logging.DEBUG):
                                    try:
                                        debug_dir = Path("data/debug_faces")
                                        debug_dir.mkdir(parents=True, exist_ok=True)
                                        ts = datetime.now().strftime('%H%M%S_%f')
                                        filename = debug_dir / f"face_{ts}_{face_id}.jpg"
                                        with open(filename, "wb") as f:
                                            f.write(face_image)
                                        self.logger.debug(f"Saved debug face: {filename}")
                                    except Exception as e:
                                        self.logger.warning(f"Failed to save debug face: {e}")
                            else:
                                self.stats["duplicate_passengers"] += 1
                                self.logger.debug(f"Pasajero duplicado detectado: {face_id}")
                        except Exception as e:
                            self.logger.warning(f"Error procesando rostro para tracking: {e}")
                            # Si hay error, considerar como nuevo pasajero
                            new_passengers.append(face)
                else:
                    # Sin tracking, todos los rostros son nuevos pasajeros
                    new_passengers = faces
                    self.stats["new_passengers"] += len(faces)
                    
                    # DEBUG: Save all faces when tracking is disabled
                    if self.logger.isEnabledFor(logging.DEBUG):
                        for i, face in enumerate(faces):
                            try:
                                face_image = extract_face_image(frame, face.bounding_box)
                                debug_dir = Path("data/debug_faces")
                                debug_dir.mkdir(parents=True, exist_ok=True)
                                ts = datetime.now().strftime('%H%M%S_%f')
                                filename = debug_dir / f"face_{ts}_no_track_{i}.jpg"
                                with open(filename, "wb") as f:
                                    f.write(face_image)
                            except Exception as e:
                                self.logger.warning(f"Failed to save debug face: {e}")
                
                if len(new_passengers) == 0:
                    self.logger.debug("Sin nuevos pasajeros en este frame")
                    time.sleep(loop_delay)
                    continue
                
                self.logger.info(f"Nuevos pasajeros: {len(new_passengers)} (de {face_count} rostros)")
                
                # 5. Registrar eventos de abordaje con geolocalización
                if self.location_enabled and hasattr(self, 'passenger_store'):
                    location = self.location_provider.get_location()
                    for face in new_passengers:
                        # Get face_id if available from tracking
                        passenger_face_id = None
                        if self.tracking_enabled and self.face_tracker:
                            try:
                                face_img = extract_face_image(frame, face.bounding_box)
                                _, passenger_face_id, _ = self.face_tracker.is_new_passenger(face_img)
                            except:
                                pass
                        
                        self.passenger_store.record_boarding(
                            face_id=passenger_face_id,
                            latitude=location.latitude,
                            longitude=location.longitude,
                            location_source=location.source,
                            location_accuracy=location.accuracy
                        )
                
                # 6. Guardar evento solo para nuevos pasajeros
                event_data = self._create_face_event(len(new_passengers))
                event_id = self.local_buffer.save_event("face_count", event_data)
                self.stats["events_saved"] += 1
                
                self.logger.debug(f"Evento guardado: id={event_id}, nuevos_pasajeros={len(new_passengers)}")
                
                # 5. Dormir para controlar CPU
                time.sleep(loop_delay)
                
                # Log confirmation periodically
                if self.stats["frames_processed"] % 100 == 0:
                    self.logger.debug(f"Updating visual window... (Motion: {motion_detected})")
                    
        except Exception as e:
            self.logger.error(f"Error en bucle principal: {e}", exc_info=True)
        finally:
            self._shutdown()
    
    def _log_stats(self) -> None:
        """Registra estadísticas actuales."""
        buffer_stats = self.local_buffer.get_stats()
        tracking_info = ""
        if self.tracking_enabled and self.face_tracker:
            tracker_stats = self.face_tracker.get_stats()
            tracking_info = f", tracked={tracker_stats['tracked_faces']}, dup_rate={tracker_stats['duplicate_rate']:.1f}%"
        
        self.logger.info(
            f"Stats: frames={self.stats['frames_processed']}, "
            f"motion={self.stats['motion_detected_count']}, "
            f"faces={self.stats['faces_detected_total']}, "
            f"new={self.stats['new_passengers']}, "
            f"dup={self.stats['duplicate_passengers']}, "
            f"events={self.stats['events_saved']}, "
            f"pending={buffer_stats['pending_events']}{tracking_info}"
        )
    
    def _sync_excluded_faces(self) -> None:
        """
        Sincroniza y descarga las fotos de personal para exclusión (conductores y colectores).
        """
        if not self.cloud_sync or not self.face_tracker:
            return

        try:
            self.logger.info("Iniciando sincronización de rostros excluidos...")
            
            # 1. Obtener conductores
            drivers = self.cloud_sync.get_excluded_faces()
            # 2. Obtener colectores
            collectors = self.cloud_sync.get_excluded_collectors()
            
            all_excluded = []
            for d in drivers:
                d['type'] = 'driver'
                all_excluded.append(d)
            for c in collectors:
                c['type'] = 'collector'
                all_excluded.append(c)
            
            if not all_excluded:
                self.logger.info("No hay personal para excluir.")
                return

            # Directorio para guardar fotos
            excluded_dir = Path("data/excluded_faces")
            excluded_dir.mkdir(parents=True, exist_ok=True)
            
            # Limpiar collage actual
            self.logger.info("Limpiando rostros excluidos anteriores...")
            self.face_tracker.clear_excluded()
            
            count = 0
            for person in all_excluded:
                try:
                    photo_url = person.get("photo_url")
                    person_id = person.get("id")
                    person_type = person.get("type", "unknown")
                    
                    if not photo_url:
                        continue
                        
                    # Determinar extension
                    path_tokens = os.path.splitext(photo_url)
                    ext = path_tokens[1] if len(path_tokens) > 1 else ".jpg"
                    if not ext: ext = ".jpg"
                    
                    # Prefix 'driver_' or 'collector_'
                    local_path = excluded_dir / f"{person_type}_{person_id}{ext}"
                    local_path_str = str(local_path)
                    
                    # Descargar foto
                    self.logger.debug(f"Descargando foto ({person_type} {person_id}): {photo_url}")
                    response = requests.get(photo_url, timeout=10)
                    
                    if response.status_code == 200:
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        
                        # Agregar al tracker
                        if self.face_tracker.add_excluded_face(local_path_str):
                            count += 1
                    else:
                        self.logger.warning(f"Error descargando foto {photo_url}: {response.status_code}")
                        
                except Exception as e:
                    self.logger.error(f"Error procesando persona {person.get('id')}: {e}")
            
            self.logger.info(f"Sincronización de excluidos completada: {count} rostros cargados.")
            
        except Exception as e:
            self.logger.error(f"Error general en _sync_excluded_faces: {e}")

    def _sync_loop(self):
        """Background loop for syncing events."""
        # Initial sync of excluded faces
        if self.tracking_enabled:
             self._sync_excluded_faces()
             self._last_excluded_sync = time.time()

        while not self.stop_sync.is_set():
            try:
                # 1. Get pending events from local buffer
                if hasattr(self.local_buffer, 'get_pending_events'):
                    pending_events = self.local_buffer.get_pending_events(limit=50)
                else:
                    # Fallback if method doesn't exist yet in LocalBuffer
                    pending_events = []
                
                if pending_events:
                    self.logger.debug(f"Intentando sincronizar {len(pending_events)} eventos...")
                    
                    # Convert to API format
                    api_events = []
                    for evt in pending_events:
                        # Assuming evt is (id, event_type, data_json, created_at, synced)
                        try:
                            # data_json comes from db, might require parsing if it's string
                            data = json.loads(evt['data']) if isinstance(evt['data'], str) else evt['data']
                            api_events.append({
                                'timestamp': data.get('timestamp'),
                                'count': data.get('count', 0),
                                'location': data.get('location'),
                                'type': 'boarding' 
                            })
                        except Exception as e:
                            self.logger.error(f"Error parsing event {evt['id']}: {e}")

                    # 2. Sync
                    if api_events and self.cloud_sync:
                         synced_count = self.cloud_sync.sync_events(api_events)
                    
                         if synced_count > 0:
                            # 3. Mark as synced in local db
                            event_ids = [evt['id'] for evt in pending_events[:synced_count]]
                            if hasattr(self.local_buffer, 'mark_synced'):
                                self.local_buffer.mark_synced(event_ids)
                else:
                    if self.cloud_sync:
                        self.logger.debug("No hay eventos pendientes, enviando heartbeat...")
                        self.cloud_sync.send_heartbeat()
            
            except Exception as e:
                self.logger.error(f"Error en bucle de sincronización: {e}")
            
            # Save tracking state periodically (every 5 mins) to prevent data loss on power failure
            if self.tracking_enabled and self.face_tracker:
                current_time = time.time()
                
                # Persistence check
                if not hasattr(self, '_last_tracking_save'):
                    self._last_tracking_save = current_time
                
                if current_time - self._last_tracking_save > 300:  # 5 minutes
                    self.face_tracker.save_state()
                    self._last_tracking_save = current_time
                
                # Excluded faces sync check (every 15 mins)
                if not hasattr(self, '_last_excluded_sync'):
                    self._last_excluded_sync = 0
                
                if current_time - self._last_excluded_sync > 900: # 15 minutes
                     self._sync_excluded_faces()
                     self._last_excluded_sync = current_time
            
            # Wait for next interval
            self.stop_sync.wait(self.sync_interval)

    def _shutdown(self) -> None:
        """Realiza el apagado limpio del sistema."""
        self.logger.info("Iniciando apagado del sistema...")
        
        # Stop sync thread
        if hasattr(self, 'stop_sync'):
             self.stop_sync.set()
             if hasattr(self, 'sync_thread') and self.sync_thread.is_alive():
                 self.sync_thread.join(timeout=2)
        
        # Detener stream de video
        self.video_stream.stop()
        
        # Save tracking state
        if self.tracking_enabled and self.face_tracker:
            self.logger.info("Guardando estado del tracker antes de salir...")
            self.face_tracker.save_state()
        
        # Log final de estadísticas
        self._log_stats()
        
        buffer_stats = self.local_buffer.get_stats()
        self.logger.info(f"Eventos pendientes de sincronizar: {buffer_stats['pending_events']}")
        
        self.logger.info("=" * 50)
        self.logger.info("TRANSPORT MONITOR - APAGADO COMPLETO")
        self.logger.info("=" * 50)
    
    def get_stats(self) -> dict:
        """
        Obtiene estadísticas completas del sistema.
        
        Returns:
            Diccionario con todas las estadísticas
        """
        stats = {
            "monitor": self.stats,
            "video_stream": self.video_stream.get_stats(),
            "motion_detector": self.motion_detector.get_stats(),
            "face_counter": self.face_counter.get_stats(),
            "local_buffer": self.local_buffer.get_stats()
        }
        if self.tracking_enabled and self.face_tracker:
            stats["face_tracker"] = self.face_tracker.get_stats()
        return stats


def parse_arguments() -> argparse.Namespace:
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
        Namespace con los argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Transport Monitor - Sistema de monitoreo de pasajeros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
    # Usar cámara física (índice 0)
    python transport_monitor.py --source 0
    
    # Modo desarrollo sin AWS
    python transport_monitor.py --dry-run --verbose
    
    # Usar archivo de video
    python transport_monitor.py --source video.mp4
    
    # Configuración personalizada
    python transport_monitor.py --config my_config.yaml
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Ruta al archivo de configuración YAML (default: config.yaml)"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=None,
        help="Fuente de video: índice de cámara (0, 1, ...) o ruta a archivo"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo simulación: no hace llamadas a AWS Rekognition"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Habilita logging verbose (DEBUG)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Ruta al archivo de log"
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Función principal del script.
    
    Returns:
        Código de salida (0 = éxito)
    """
    # Cargar variables de entorno desde .env
    load_dotenv()
    
    args = parse_arguments()
    print("[1/8] Iniciando script transport_monitor.py...")
    print(f"[1/8] ENABLE_DEBUG_LOGS = {os.getenv('ENABLE_DEBUG_LOGS')}")
    
    # Cargar configuración
    config = get_default_config()
    file_config = load_config(args.config)
    
    # Merge de configuraciones (archivo sobrescribe defaults)
    for key, value in file_config.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    # Aplicar argumentos de línea de comandos (sobrescriben config)
    if args.source is not None:
        # Intentar convertir a int si es un índice de cámara
        try:
            config["camera"]["source"] = int(args.source)
        except ValueError:
            config["camera"]["source"] = args.source
    
    if args.dry_run:
        config["detector"]["dry_run"] = True
    
    if args.verbose:
        config["system"]["log_level"] = "DEBUG"
        config["system"]["verbose"] = True
    
    if args.log_file:
        config["system"]["log_file"] = args.log_file
    
    # Configuración de logging desde .env
    if os.getenv("ENABLE_DEBUG_LOGS", "").lower() == "true":
        config["system"]["log_level"] = "DEBUG"
        config["system"]["verbose"] = True
        print("DEBUG MODE ENABLED via .env")

    # Configurar logging
    setup_logging(
        level=config["system"].get("log_level", "INFO"),
        log_file=config["system"].get("log_file")
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Iniciando Transport Monitor v1.0.0")
    logger.info(f"Dry-run: {config['detector'].get('dry_run', False)}")
    
    # Crear e iniciar monitor
    print("[2/8] Configuración cargada. Creando TransportMonitor...")
    try:
        monitor = TransportMonitor(config)
        print("[8/8] TransportMonitor creado. Iniciando bucle principal...")
        monitor.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Interrupción por teclado")
        return 0
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
