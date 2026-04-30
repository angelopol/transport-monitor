# 🔄 Ciclo de Orquestación: Transport Monitor (Daemon)

El script `transport_monitor.py` no implementa algoritmos matemáticos profundos por sí mismo, sino que actúa como el **Director de Orquesta (Daemon/Servicio)**. Es el punto de entrada principal que corre de manera perpetua dentro del hardware perimetral (Edge/Raspberry Pi) de cada autobús.

> [!IMPORTANT]
> Este algoritmo define el "Bucle de Vida" del dispositivo de transporte. Rige cómo los órganos (clases de la librería `stream-faces-counter`) interactúan entre sí y con el mundo exterior.

---

## 🛤️ El Algoritmo de Ejecución Principal (Main Loop)

El comportamiento del orquestador se divide en dos fases: Inicialización (Boot) y el Bucle Perpetuo de Trabajo (While True).

### Fase 1: Inicialización y Negociación (Handshake)
Al recibir energía eléctrica, el script arranca y ejecuta:
1.  **Levantamiento de Componentes:** Instancia los objetos (Cámara, GPS, Bases de Datos Locales, Motores de IA).
2.  **Extracción de Identidad:** Recupera la `Mac Address` de la tarjeta de red física del dispositivo.
3.  **Handshake Criptográfico:** Llama al módulo `CloudSync`. Envía sus credenciales al servidor central Laravel.
    *   *Si es denegado o baneado:* El script lanza un error fatal y suspende la ejecución para apagar la cámara y ahorrar energía de la batería del bus.
    *   *Si es exitoso:* Recibe confirmación de a qué autobús pertenece y obtiene un Token JWT.
4.  **Descarga de Inteligencia (Sync Excluded):** Tras el handshake, descarga silenciosamente las fotos de expediente de los conductores de guardia asignados a ese bus y las pre-carga en la memoria RAM del Tracker para armar su "Lista Negra" biométrica.

### Fase 2: El Bucle Perpetuo (Pipeline de Eventos)
Una vez estabilizado, entra en un `while self.running:` que se ejecuta idealmente 5 a 10 veces por segundo:

1.  **Extracción Dinámica:** Lee el último fotograma del hilo asíncrono del `VideoStream`.
2.  **Cribado de Movimiento:** Lanza el frame al `MotionDetector`.
    *   *Punto de Inflexión:* Si el detector devuelve `False` (bus vacío/quieto), el orquestador usa `time.sleep()` durante milisegundos, descarta el frame y vuelve al Paso 1. Esto ahorra un 90% del esfuerzo computacional.
3.  **Inferencia Perimetral:** Si hay movimiento (`True`), manda el frame al `FaceCounter`.
    *   Si se hallan rostros (ej. `face_count = 2`), se avanza al siguiente escudo de defensa.
4.  **Deduplicación y Filtrado Exclusivo:** Se itera sobre los rostros encontrados enviándolos al `FaceTracker.is_new_passenger()`.
    *   Si el algoritmo dictamina que es un rostro conocido (Duplicado) o es el Chofer (Staff), el evento **se ignora por completo**.
    *   Si el algoritmo devuelve `True` (Humano Inédito), se empuja a un arreglo temporal de `new_passengers`.
5.  **Persistencia Georeferenciada:**
    *   Contacta al `LocationProvider` por el puerto Serial (o IP) para pedir la Coordenada GPS de ese microsegundo.
    *   Crea el paquete JSON con el timestamp exacto, ID del bus, Coordenadas y Conteo (+1).
    *   Inyecta el objeto forzosamente a la base de datos segura SQLite (`LocalBuffer`).
6.  **Reinicio del Ciclo:** Limpia la memoria transitoria del frame y vuelve a esperar el siguiente.

### Fase 3: Subprocesos en Segundo Plano (Daemons Asíncronos)
Paralelo al Bucle Perpetuo, existe un hilo independiente (`self.sync_thread`) que jamás interrumpe a las cámaras:
*   **Cronjob de Sincronización:** Cada X segundos (ej. 60s), el hilo revisa la tabla local SQLite.
*   Si hay eventos represados (porque el bus pasó por un túnel o cerro sin señal), empaca todos los registros JSON no sincronizados en un solo bloque (Batch).
*   Dispara la solicitud HTTP `POST` hacia la Nube. Si la nube devuelve código `200 OK`, el hilo elimina físicamente las filas de SQLite, manteniendo el disco duro ligero y limpio.
*   En paralelo, envía un "Heartbeat" (Latido de vida) para que el dueño sepa en su Dashboard que la unidad sigue encendida.

---

## 📈 Tolerancia a Fallos Integrada

*   **Bloques Try-Catch Cíclicos:** Un error en un fotograma corrupto no crashea el script; el error se registra en los Logs y el bucle continúa con el siguiente fotograma ininterrumpidamente.
*   **Graceful Shutdown:** Atrapa señales del sistema operativo (`SIGINT`, `SIGTERM`) de la Raspberry para guardar de forma segura los rastros de RAM en disco antes de que el voltaje caiga a 0.

---

**Fuente:** Documentación Técnica de Algoritmos (2026).
