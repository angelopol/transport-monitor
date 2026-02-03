"""
test_modules.py - Script de prueba para todos los modulos

Ejecutar con: python test_modules.py
"""

import sys
import numpy as np

def test_motion_detector():
    print("\n=== Test MotionDetector ===")
    from stream_count_faces import MotionDetector
    
    md = MotionDetector(min_area=5000, threshold=25)
    
    # Frame negro (establece referencia)
    f1 = np.zeros((480, 640, 3), dtype=np.uint8)
    result1 = md.detect(f1)
    print(f"Frame negro (referencia): motion={result1}")
    assert result1 == False, "Primer frame deberia ser False"
    
    # Frame identico
    f2 = np.zeros((480, 640, 3), dtype=np.uint8)
    result2 = md.detect(f2)
    print(f"Frame negro (igual): motion={result2}")
    assert result2 == False, "Frame igual deberia ser False"
    
    # Frame con cambio
    f3 = np.ones((480, 640, 3), dtype=np.uint8) * 255
    result3 = md.detect(f3)
    print(f"Frame blanco (diferente): motion={result3}")
    assert result3 == True, "Frame diferente deberia ser True"
    
    stats = md.get_stats()
    print(f"Stats: {stats}")
    print("[OK] MotionDetector: PASSED")
    return True

def test_local_buffer():
    print("\n=== Test LocalBuffer ===")
    from stream_count_faces import LocalBuffer
    
    lb = LocalBuffer(":memory:")
    
    # Guardar eventos
    id1 = lb.save_event("face_count", {"count": 5, "location": "entrance"})
    id2 = lb.save_event("motion", {"area": 1000})
    id3 = lb.save_event("face_count", {"count": 2})
    print(f"Eventos guardados: {id1}, {id2}, {id3}")
    
    # Obtener pendientes
    pending = lb.get_pending_events()
    print(f"Eventos pendientes: {len(pending)}")
    assert len(pending) == 3, "Deberian haber 3 eventos pendientes"
    
    # Marcar como sincronizados
    lb.mark_synced([id1, id2])
    
    # Verificar stats
    stats = lb.get_stats()
    print(f"Stats: pending={stats['pending_events']}, synced={stats['synced_events']}")
    assert stats['pending_events'] == 1, "Deberia haber 1 evento pendiente"
    assert stats['synced_events'] == 2, "Deberian haber 2 eventos sincronizados"
    
    print("[OK] LocalBuffer: PASSED")
    return True

def test_face_counter():
    print("\n=== Test FaceCounter (dry_run) ===")
    from stream_count_faces import FaceCounter
    
    fc = FaceCounter(dry_run=True)
    
    # Frame negro (sin rostros)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = fc.count_faces(frame)
    print(f"Rostros en frame negro: {len(faces)}")
    assert len(faces) == 0, "Frame negro no deberia tener rostros"
    
    stats = fc.get_stats()
    print(f"Stats: frames={stats['total_frames_processed']}, dry_run={stats['dry_run']}")
    assert stats['dry_run'] == True, "Deberia estar en dry_run"
    assert stats['total_frames_processed'] == 1, "Deberia haber procesado 1 frame"
    
    print("[OK] FaceCounter: PASSED")
    return True

def test_face_tracker():
    print("\n=== Test FaceTracker (dry_run) ===")
    from stream_count_faces import FaceTracker
    
    tracker = FaceTracker(dry_run=True, ttl_minutes=5, max_faces=10)
    
    # Simular deteccion de pasajeros
    fake_face_1 = b"fake_face_image_1"
    fake_face_2 = b"fake_face_image_2"
    
    # Primer pasajero - deberia ser nuevo
    is_new1, face_id1, is_excluded1 = tracker.is_new_passenger(fake_face_1)
    print(f"Pasajero 1: is_new={is_new1}, face_id={face_id1}, is_excluded={is_excluded1}")
    assert is_new1 == True, "Primer pasajero deberia ser nuevo"
    assert is_excluded1 == False, "No deberia ser excluido"
    
    # Segundo pasajero - deberia ser nuevo (dry_run siempre es nuevo)
    is_new2, face_id2, is_excluded2 = tracker.is_new_passenger(fake_face_2)
    print(f"Pasajero 2: is_new={is_new2}, face_id={face_id2}, is_excluded={is_excluded2}")
    assert is_new2 == True, "Segundo pasajero deberia ser nuevo en dry_run"
    
    stats = tracker.get_stats()
    print(f"Stats: tracked={stats['tracked_faces']}, new={stats['new_passengers']}, excluded={stats['excluded_faces']}")
    assert stats['tracked_faces'] == 2, "Deberian haber 2 rostros rastreados"
    assert stats['new_passengers'] == 2, "Deberian haber 2 nuevos pasajeros"
    
    print("[OK] FaceTracker: PASSED")
    return True

def test_transport_monitor_import():
    print("\n=== Test Transport Monitor Import ===")
    # Importar el modulo principal
    import transport_monitor
    print(f"transport_monitor importado: {transport_monitor}")
    print(f"TransportMonitor class: {transport_monitor.TransportMonitor}")
    print("[OK] Transport Monitor Import: PASSED")
    return True

def main():
    print("=" * 50)
    print("TRANSPORT MONITOR - TEST DE MODULOS")
    print("=" * 50)
    
    all_passed = True
    
    try:
        all_passed &= test_motion_detector()
    except Exception as e:
        print(f"[FAIL] MotionDetector: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_local_buffer()
    except Exception as e:
        print(f"[FAIL] LocalBuffer: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_face_counter()
    except Exception as e:
        print(f"[FAIL] FaceCounter: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_face_tracker()
    except Exception as e:
        print(f"[FAIL] FaceTracker: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_transport_monitor_import()
    except Exception as e:
        print(f"[FAIL] Transport Monitor Import: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_geolocation()
    except Exception as e:
        print(f"[FAIL] Geolocation: FAILED - {e}")
        all_passed = False
    
    try:
        all_passed &= test_passenger_event_store()
    except Exception as e:
        print(f"[FAIL] PassengerEventStore: FAILED - {e}")
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("TODOS LOS TESTS PASARON [OK]")
        return 0
    else:
        print("ALGUNOS TESTS FALLARON [FAIL]")
        return 1


def test_geolocation():
    print("\n=== Test LocationProvider ===")
    from stream_count_faces import LocationProvider
    
    # Crear provider sin GPS (solo IP fallback)
    provider = LocationProvider(use_ip_fallback=True)
    
    # Obtener ubicación
    location = provider.get_location()
    print(f"Ubicación obtenida: lat={location.latitude}, lng={location.longitude}, source={location.source}")
    
    # Verificar estadísticas
    stats = provider.get_stats()
    print(f"Stats: ip_fallback={stats['ip_fallback_enabled']}, cached={stats['cached_location'] is not None}")
    
    provider.close()
    print("[OK] LocationProvider: PASSED")
    return True


def test_passenger_event_store():
    print("\n=== Test PassengerEventStore ===")
    from stream_count_faces import PassengerEventStore
    
    # Usar base de datos en memoria
    store = PassengerEventStore(":memory:")
    
    # Registrar algunos abordajes
    id1 = store.record_boarding(face_id="abc123", latitude=10.5, longitude=-66.9, location_source="gps")
    id2 = store.record_boarding(face_id="def456", latitude=10.5, longitude=-66.9, location_source="gps")
    id3 = store.record_boarding(face_id="ghi789", latitude=None, longitude=None, location_source="none")
    
    print(f"Eventos registrados: {id1}, {id2}, {id3}")
    
    # Verificar estadísticas
    stats = store.get_stats()
    print(f"Stats: total={stats['total_events']}, con_ubicacion={stats['events_with_location']}, sin_ubicacion={stats['events_without_location']}")
    
    assert stats['total_events'] == 3, "Deberian haber 3 eventos"
    assert stats['events_with_location'] == 2, "Deberian haber 2 eventos con ubicación"
    assert stats['events_without_location'] == 1, "Deberia haber 1 evento sin ubicación"
    
    # Verificar stats por ubicación
    loc_stats = store.get_location_stats()
    print(f"Ubicaciones: {len(loc_stats)} paradas unicas")
    if loc_stats:
        print(f"Parada 1: lat={loc_stats[0]['latitude']}, lng={loc_stats[0]['longitude']}, pasajeros={loc_stats[0]['passenger_count']}")
    
    print("[OK] PassengerEventStore: PASSED")
    return True


if __name__ == "__main__":
    sys.exit(main())
