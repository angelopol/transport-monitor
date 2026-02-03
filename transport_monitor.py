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
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from stream_count_faces import VideoStream, MotionDetector, LocalBuffer, FaceCounter, FaceTracker, extract_face_image


# Configuración de logging
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
            "min_area": 5000,
            "threshold": 25,
            "blur_kernel": 21,
            "cooldown_frames": 5
        },
        "detector": {
            "face_confidence_threshold": 90,
            "face_occluded_threshold": 80,
            "frontal_threshold": 35,
            "dry_run": False
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
        
        # VideoStream
        self.video_stream = VideoStream(
            source=cam_config.get("source", 0),
            width=cam_config.get("width", 1280),
            height=cam_config.get("height", 720)
        )
        
        # MotionDetector
        self.motion_detector = MotionDetector(
            min_area=motion_config.get("min_area", 5000),
            threshold=motion_config.get("threshold", 25),
            blur_kernel=motion_config.get("blur_kernel", 21),
            cooldown_frames=motion_config.get("cooldown_frames", 5)
        )
        
        # FaceCounter
        self.face_counter = FaceCounter(
            face_confidence_threshold=detector_config.get("face_confidence_threshold", 90),
            face_occluded_threshold=detector_config.get("face_occluded_threshold", 80),
            frontal_threshold=detector_config.get("frontal_threshold", 35),
            dry_run=detector_config.get("dry_run", False),
            region=aws_config.get("region", "us-east-1")
        )
        
        # LocalBuffer
        self.local_buffer = LocalBuffer(
            db_path=storage_config.get("database_path", "data/transport_events.db")
        )
        
        # FaceTracker (deduplicación de pasajeros)
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
            "timestamp": datetime.now().isoformat(),
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
        
        try:
            while self.running:
                # 1. Leer frame
                frame = self.video_stream.read()
                
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.stats["frames_processed"] += 1
                
                # 2. Verificar movimiento
                motion_detected = self.motion_detector.detect(frame)
                
                if not motion_detected:
                    time.sleep(loop_delay)
                    continue
                
                self.stats["motion_detected_count"] += 1
                self.logger.debug("Movimiento detectado, analizando rostros...")
                
                # 3. Detectar rostros
                faces = self.face_counter.count_faces(frame)
                face_count = len(faces)
                
                if face_count == 0:
                    time.sleep(loop_delay)
                    continue
                
                self.stats["faces_detected_total"] += face_count
                self.logger.debug(f"Rostros detectados en frame: {face_count}")
                
                # 4. Filtrar pasajeros duplicados (si tracking está habilitado)
                new_passengers = []
                if self.tracking_enabled and self.face_tracker:
                    for face in faces:
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
                
                if len(new_passengers) == 0:
                    self.logger.debug("Sin nuevos pasajeros en este frame")
                    time.sleep(loop_delay)
                    continue
                
                self.logger.info(f"Nuevos pasajeros: {len(new_passengers)} (de {face_count} rostros)")
                
                # 5. Guardar evento solo para nuevos pasajeros
                event_data = self._create_face_event(len(new_passengers))
                event_id = self.local_buffer.save_event("face_count", event_data)
                self.stats["events_saved"] += 1
                
                self.logger.debug(f"Evento guardado: id={event_id}, nuevos_pasajeros={len(new_passengers)}")
                
                # 5. Dormir para controlar CPU
                time.sleep(loop_delay)
                
                # Log periódico de estadísticas
                if self.stats["frames_processed"] % 100 == 0:
                    self._log_stats()
                    
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
    
    def _shutdown(self) -> None:
        """Realiza el apagado limpio del sistema."""
        self.logger.info("Iniciando apagado del sistema...")
        
        # Detener stream de video
        self.video_stream.stop()
        
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
    args = parse_arguments()
    
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
    
    # Configurar logging
    setup_logging(
        level=config["system"].get("log_level", "INFO"),
        log_file=config["system"].get("log_file")
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Iniciando Transport Monitor v1.0.0")
    logger.info(f"Dry-run: {config['detector'].get('dry_run', False)}")
    
    # Crear e iniciar monitor
    try:
        monitor = TransportMonitor(config)
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
