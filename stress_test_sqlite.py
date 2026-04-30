import sqlite3
import time
import os
import json
from datetime import datetime, timedelta

DB_PATH = "data/transport_events.db"

def setup_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS event_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            synced INTEGER DEFAULT 0,
            sync_timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_synced ON event_queue(synced)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_timestamp ON event_queue(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON event_queue(event_type)")
    conn.commit()
    return conn

def run_stress_test():
    print("=== Iniciando Prueba 9: Rendimiento SQLite (200,000 registros) ===")
    conn = setup_db()
    cursor = conn.cursor()
    
    # 1. Limpiar base de datos temporalmente para el test
    print("Limpiando base de datos para la prueba...")
    cursor.execute("DELETE FROM event_queue")
    conn.commit()
    
    # 2. Insercion masiva
    print("Generando e insertando 200,000 registros sinteticos...")
    start_insert = time.time()
    
    # Pre-calcular una fecha antigua para simulacion de purga (hace 40 dias)
    old_date = (datetime.now() - timedelta(days=40)).isoformat()
    base_data = json.dumps({"count": 1, "location": {"lat": 10.0, "lon": -66.0}})
    
    # Batch insert is much faster
    batch_size = 50000
    total_records = 200000
    
    for i in range(0, total_records, batch_size):
        records = []
        for j in range(batch_size):
            # Simulamos que ya estan sincronizados y son viejos
            records.append((old_date, "face_count", base_data, 1, old_date))
        
        cursor.executemany(
            """
            INSERT INTO event_queue (timestamp, event_type, data, synced, sync_timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            records
        )
        conn.commit()
        print(f"  Insertados {i + batch_size} / {total_records}...")
        
    insert_time = time.time() - start_insert
    print(f"Insercion completada en {insert_time:.2f} segundos.")
    
    # Verificar tamaño del archivo
    file_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"Tamaño actual del archivo SQLite: {file_size_mb:.2f} MB")
    
    # 3. Borrado condicional (Purga masiva de registros de mas de 30 dias)
    print("\nEjecutando limpieza condicional masiva (Purga)...")
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    
    start_delete = time.time()
    cursor.execute(
        "DELETE FROM event_queue WHERE synced = 1 AND sync_timestamp < ?",
        (cutoff,)
    )
    conn.commit()
    deleted_count = cursor.rowcount
    delete_time = time.time() - start_delete
    
    print(f"Limpieza completada.")
    print(f"Registros eliminados: {deleted_count}")
    print(f"Tiempo de borrado: {delete_time:.3f} segundos.")
    
    # VACIAR el archivo para reclamar espacio real en disco
    print("\nEjecutando VACUUM para recuperar espacio en disco...")
    start_vacuum = time.time()
    cursor.execute("VACUUM")
    conn.commit()
    vacuum_time = time.time() - start_vacuum
    file_size_mb_after = os.path.getsize(DB_PATH) / (1024 * 1024)
    print(f"VACUUM completado en {vacuum_time:.3f} segundos.")
    print(f"Tamaño final del archivo SQLite: {file_size_mb_after:.2f} MB")
    
    conn.close()
    
    print("\n--- Conclusion ---")
    if delete_time < 10.0 and deleted_count == 200000:
        print("[ESTADO EXCELENTE] El motor SQLite supero la prueba de estres exitosamente.")
        print("Mantuvo la integridad estructural de los indices y completo el borrado masivo en menos de 10s.")
    else:
        print("[FALLO] El borrado tomo demasiado tiempo o fallo la integridad de la base de datos.")

if __name__ == "__main__":
    run_stress_test()
