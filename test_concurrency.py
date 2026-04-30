import sqlite3
import threading
import requests
import json
import time
from datetime import datetime

# Rutas locales
ADMIN_DB = r"C:\Users\Angel\OneDrive\Escritorio\Courses\Tesis\software\transport-admin\database\database.sqlite"
API_URL = "http://localhost:8000/api/v1/sync"

def get_valid_token_and_bus():
    """Obtiene un token válido y el MAC de un dispositivo vinculado a un bus activo."""
    try:
        conn = sqlite3.connect(ADMIN_DB)
        cursor = conn.cursor()
        
        # Buscar un dispositivo vinculado a un bus activo
        cursor.execute("""
            SELECT d.api_token, b.id, d.mac_address 
            FROM devices d
            JOIN buses b ON d.mac_address = b.device_mac
            WHERE b.is_active = 1
            LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row[0], row[1], row[2]
        return None, None, None
    except Exception as e:
        print(f"Error accediendo a la DB: {e}")
        return None, None, None

def send_request(thread_id, token, payload, results):
    """Envía la solicitud HTTP al backend."""
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    print(f"[{thread_id}] Iniciando envío concurrente...")
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        elapsed = time.time() - start_time
        
        status = response.status_code
        print(f"[{thread_id}] Respuesta recibida en {elapsed:.3f}s: {status}")
        
        results.append({
            'thread_id': thread_id,
            'status': status,
            'body': response.text
        })
    except Exception as e:
        print(f"[{thread_id}] Error de red: {e}")
        results.append({'thread_id': thread_id, 'status': 0, 'body': str(e)})

def run_test():
    print("=== Iniciando Prueba de Concurrencia (Prueba 5) ===")
    
    token, bus_id, mac = get_valid_token_and_bus()
    
    if not token:
        print("[ERROR] No se encontro un dispositivo con token vinculado a un bus activo en la base de datos.")
        return
        
    print(f"[OK] Dispositivo encontrado (MAC: {mac}, Bus ID: {bus_id})")
    print(f"Utilizando API URL: {API_URL}")
    
    # Preparar carga útil idéntica para ambas solicitudes
    timestamp = datetime.now().isoformat()
    
    payload = {
        "events": [
            {
                "timestamp": timestamp,
                "count": 1,
                "type": "boarding",
                "location": {
                    "lat": 10.4806,
                    "lon": -66.9036
                }
            }
        ]
    }
    
    print(f"Payload de prueba (Timestamp: {timestamp}) preparado.")
    print("Lanzando 3 peticiones identicas simultaneamente para forzar condicion de carrera...")
    
    results = []
    threads = []
    
    # Lanzar hilos concurrentes
    for i in range(3):
        t = threading.Thread(target=send_request, args=(f"Hilo-{i+1}", token, payload, results))
        threads.append(t)
        
    for t in threads:
        t.start()
        
    for t in threads:
        t.join()
        
    print("\n=== Resultados de la Prueba ===")
    success_count = 0
    
    for r in results:
        if r['status'] == 200:
            success_count += 1
            print(f"[EXITO] {r['thread_id']}: Aceptado (200 OK)")
        else:
            print(f"[RECHAZADO] {r['thread_id']}: ({r['status']}) - Probablemente por duplicidad u error interno")
            print(f"   Mensaje: {r['body']}")
            
    print("\n--- Conclusion ---")
    if success_count == 1:
        print("[ESTADO EXCELENTE] El sistema manejo la concurrencia correctamente. Acepto 1 peticion y rechazo las demas gracias a la restriccion matematica UNIQUE.")
    elif success_count > 1:
        print("[FALLO] El sistema acepto multiples peticiones identicas. Posible vulnerabilidad de duplicacion de afluencia.")
    else:
        print("[FALLO] Ninguna peticion fue aceptada (revisa que el servidor PHP este corriendo o el token sea valido).")

if __name__ == "__main__":
    run_test()
