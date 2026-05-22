#!/usr/bin/env python3
"""
test_gps.py - Diagnóstico completo de GPS y ubicación

Prueba todas las fuentes de ubicación disponibles:
  1. Windows Location API (WinRT) - la misma que usa Google Maps en el browser
  2. GPSD (daemon Linux)
  3. GPS Serial (USB/COM)
  4. IP Geolocation (fallback)
  5. LocationProvider completo (cadena de fallback del proyecto)

Uso:
    python test_gps.py
    python test_gps.py --verbose
    python test_gps.py --source windows
    python test_gps.py --source ip
"""

import argparse
import logging
import platform
import sys
import time
from datetime import datetime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 60

def title(text: str) -> None:
    print(f"\n{SEP}")
    print(f"  {text}")
    print(SEP)

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")

def info(msg: str) -> None:
    print(f"  [INFO] {msg}")

def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


# ---------------------------------------------------------------------------
# 1. Windows Location API (WinRT)
# ---------------------------------------------------------------------------

def test_windows_location(verbose: bool = False) -> dict:
    """Prueba la Windows Location API via PowerShell (no requiere compilación C++)."""
    title("1. Windows Location API (WinRT via PowerShell)")
    result = {"source": "windows", "available": False, "location": None, "error": None}

    if platform.system() != "Windows":
        warn("No estás en Windows — esta fuente no aplica.")
        return result

    import subprocess

    # PowerShell: usa el bridge .NET-WinRT para llamar Windows.Devices.Geolocation.
    # - System.Runtime.WindowsRuntime.dll se carga por ruta absoluta (no está en el GAC)
    # - AsTask es un método genérico; se invoca via reflexión con Geoposition como TResult
    ps_script = (
        "[System.Reflection.Assembly]::LoadFrom("
        "'C:\\Windows\\Microsoft.NET\\Framework64\\v4.0.30319\\System.Runtime.WindowsRuntime.dll') | Out-Null; "
        "[Windows.Devices.Geolocation.Geolocator,Windows.Devices.Geolocation,ContentType=WindowsRuntime] | Out-Null; "
        "[Windows.Devices.Geolocation.Geoposition,Windows.Devices.Geolocation,ContentType=WindowsRuntime] | Out-Null; "
        "$gpType = [Windows.Devices.Geolocation.Geoposition,Windows.Devices.Geolocation,ContentType=WindowsRuntime]; "
        "$geo = [Windows.Devices.Geolocation.Geolocator,Windows.Devices.Geolocation,ContentType=WindowsRuntime]::new(); "
        "$asyncOp = $geo.GetGeopositionAsync(); "
        "$m = [System.WindowsRuntimeSystemExtensions].GetMethods() | "
        "Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethodDefinition -and "
        "$_.GetParameters().Length -eq 1 -and $_.GetParameters()[0].ParameterType.Name -like 'IAsyncOperation*' } | "
        "Select-Object -First 1; "
        "$task = $m.MakeGenericMethod($gpType).Invoke($null, @(,$asyncOp)); "
        "if ($task.Wait(10000)) { "
        "  $c = $task.Result.Coordinate; "
        "  Write-Output \"OK,$($c.Latitude),$($c.Longitude),$($c.Accuracy)\" "
        "} else { Write-Output 'TIMEOUT' }"
    )

    info("Invocando Windows.Devices.Geolocation via PowerShell...")
    info("(Configuración > Privacidad > Ubicación debe estar activada)")
    t0 = time.time()

    try:
        proc = subprocess.run(
            ["powershell", "-NonInteractive", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        fail("Timeout: PowerShell no respondió en 20 s")
        result["error"] = "timeout"
        return result
    except FileNotFoundError:
        fail("powershell.exe no encontrado en PATH")
        result["error"] = "powershell not found"
        return result

    elapsed = time.time() - t0
    output = (proc.stdout or "").strip()

    if verbose:
        info(f"stdout: {output!r}")
        if proc.stderr:
            info(f"stderr: {proc.stderr.strip()!r}")

    if output.startswith("OK,"):
        parts = output[3:].split(",")
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            accuracy = float(parts[2]) if len(parts) > 2 and parts[2] else None
        except ValueError as exc:
            fail(f"Error parseando respuesta: {exc} — output={output!r}")
            result["error"] = str(exc)
            return result

        ok(f"Latitud  : {lat}")
        ok(f"Longitud : {lon}")
        ok(f"Precisión: {accuracy} metros")
        ok(f"Tiempo   : {round(elapsed, 2)} s")
        info(f"Maps URL : https://maps.google.com/?q={lat},{lon}")
        result["available"] = True
        result["location"] = {
            "latitude": lat,
            "longitude": lon,
            "accuracy": accuracy,
            "elapsed_s": round(elapsed, 2),
        }
        return result

    if output == "TIMEOUT":
        fail("GetGeopositionAsync() no retornó en 10 s")
        fail("Posible causa: Location Services desactivado o GPS sin señal")
        info("Ve a: Configuración > Privacidad > Ubicación")
        result["error"] = "GetGeopositionAsync timeout"
        return result

    # Error de PowerShell — mostrar stderr para diagnóstico
    fail(f"PowerShell retornó código {proc.returncode}")
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            fail(f"  {line}")
    if "access" in output.lower() or "denied" in output.lower():
        fail("Permiso de ubicación denegado.")
        info("Activa Ubicación en: Configuración > Privacidad > Ubicación")
        result["error"] = "access denied"
    else:
        result["error"] = f"powershell exit {proc.returncode}: {output[:200]}"

    return result


# ---------------------------------------------------------------------------
# 2. GPSD
# ---------------------------------------------------------------------------

def test_gpsd(verbose: bool = False) -> dict:
    title("2. GPSD (daemon Linux)")
    result = {"source": "gpsd", "available": False, "location": None, "error": None}

    try:
        from gps import gps, WATCH_ENABLE
    except ImportError:
        fail("Módulo 'gps' no instalado (normal en Windows). pip install gps")
        result["error"] = "gps module not installed"
        return result

    ok("Módulo gps disponible")

    try:
        session = gps(host="localhost", port=2947)
        session.stream(WATCH_ENABLE)
        info("Conectado a GPSD, esperando fix...")

        for _ in range(10):
            report = session.next()
            if verbose:
                info(f"Report: {report}")
            if report.get("class") == "TPV":
                lat = report.get("lat")
                lon = report.get("lon")
                if lat is not None and lon is not None:
                    ok(f"Latitud  : {lat}")
                    ok(f"Longitud : {lon}")
                    result["available"] = True
                    result["location"] = {"latitude": lat, "longitude": lon}
                    return result

        fail("GPSD conectado pero sin fix GPS")
        result["error"] = "no fix"
    except Exception as exc:
        fail(f"No se pudo conectar a GPSD: {exc}")
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# 3. Serial GPS (COM ports)
# ---------------------------------------------------------------------------

def test_serial_gps(verbose: bool = False) -> dict:
    title("3. GPS Serial (puertos COM / tty)")
    result = {"source": "serial", "available": False, "location": None, "error": None}

    try:
        import serial
        import serial.tools.list_ports
    except ImportError:
        fail("PySerial no instalado. pip install pyserial")
        result["error"] = "pyserial not installed"
        return result

    ports = list(serial.tools.list_ports.comports())
    if not ports:
        warn("No se encontraron puertos COM/tty disponibles.")
        info("Si tienes un USB GPS, verifica que esté conectado.")
        result["error"] = "no serial ports found"
        return result

    info("Puertos disponibles:")
    for p in ports:
        info(f"  {p.device} — {p.description}")

    # Intentar leer NMEA de cada puerto
    for port in ports:
        info(f"Probando {port.device}...")
        try:
            conn = serial.Serial(port.device, baudrate=9600, timeout=2)
            for _ in range(30):
                raw = conn.readline()
                line = raw.decode("ascii", errors="ignore").strip()
                if verbose and line:
                    info(f"  RAW: {line}")
                if line.startswith(("$GPGGA", "$GNGGA", "$GPRMC", "$GNRMC")):
                    parts = line.split(",")
                    if line.startswith(("$GPGGA", "$GNGGA")) and len(parts) >= 7 and parts[6] != "0":
                        try:
                            def nmea2deg(coord, direc):
                                if direc in ("N", "S"):
                                    deg = float(coord[:2]); mins = float(coord[2:])
                                else:
                                    deg = float(coord[:3]); mins = float(coord[3:])
                                val = deg + mins / 60.0
                                return -val if direc in ("S", "W") else val

                            lat = nmea2deg(parts[2], parts[3])
                            lon = nmea2deg(parts[4], parts[5])
                            ok(f"Fix en {port.device}: lat={lat}, lon={lon}")
                            result["available"] = True
                            result["location"] = {"latitude": lat, "longitude": lon}
                            conn.close()
                            return result
                        except Exception:
                            pass
            conn.close()
            fail(f"Sin datos GPS NMEA válidos en {port.device}")
        except Exception as exc:
            fail(f"Error en {port.device}: {exc}")

    result["error"] = "no valid NMEA on any port"
    return result


# ---------------------------------------------------------------------------
# 4. IP Geolocation
# ---------------------------------------------------------------------------

def test_ip_geolocation(verbose: bool = False) -> dict:
    title("4. IP Geolocation (fallback)")
    result = {"source": "ip", "available": False, "location": None, "error": None}

    try:
        import geocoder
    except ImportError:
        fail("geocoder no instalado. pip install geocoder")
        result["error"] = "geocoder not installed"
        return result

    ok("geocoder disponible")
    info("Consultando IP pública...")

    try:
        g = geocoder.ip("me")
        if verbose:
            info(f"Raw response: {g.json}")
        if g.ok and g.latlng:
            lat, lon = g.latlng
            ok(f"Latitud  : {lat}")
            ok(f"Longitud : {lon}")
            ok(f"Ciudad   : {getattr(g, 'city', 'N/A')}, {getattr(g, 'country', 'N/A')}")
            warn("Precisión: ~5 km (nivel ciudad) — es solo un fallback")
            result["available"] = True
            result["location"] = {"latitude": lat, "longitude": lon, "accuracy": 5000}
        else:
            fail(f"geocoder.ip falló: {g.status}")
            result["error"] = g.status
    except Exception as exc:
        fail(f"Error en IP geolocation: {exc}")
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# 5. LocationProvider completo (cadena del proyecto)
# ---------------------------------------------------------------------------

def test_location_provider(verbose: bool = False) -> dict:
    title("5. LocationProvider (cadena completa del proyecto)")
    result = {"source": "provider", "available": False, "location": None, "error": None}

    # Ajustar path si se ejecuta desde transport-monitor
    import os
    stream_path = os.path.join(os.path.dirname(__file__), "..", "stream-faces-counter")
    if os.path.isdir(stream_path):
        sys.path.insert(0, stream_path)

    try:
        from stream_count_faces import LocationProvider
    except ImportError:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stream-faces-counter"))
            from stream_count_faces import LocationProvider
        except ImportError as exc:
            fail(f"No se pudo importar LocationProvider: {exc}")
            info("Asegúrate de instalar el paquete: pip install -e ../stream-faces-counter")
            result["error"] = str(exc)
            return result

    ok("LocationProvider importado correctamente")
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    provider = LocationProvider(use_ip_fallback=True)
    stats = provider.get_stats()
    info(f"Stats: {stats}")

    info("Obteniendo ubicación...")
    t0 = time.time()
    location = provider.get_location()
    elapsed = time.time() - t0

    if location.is_valid():
        ok(f"Latitud  : {location.latitude}")
        ok(f"Longitud : {location.longitude}")
        ok(f"Fuente   : {location.source}")
        ok(f"Precisión: {location.accuracy} metros")
        ok(f"Tiempo   : {round(elapsed, 2)} s")
        result["available"] = True
        result["location"] = location.to_dict()
    else:
        fail(f"LocationProvider no obtuvo ubicación (source='{location.source}')")
        result["error"] = "no valid location"

    provider.close()
    return result


# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------

def print_summary(results: list) -> None:
    title("RESUMEN")
    any_ok = False
    for r in results:
        src = r["source"]
        if r["available"]:
            loc = r["location"]
            lat = loc.get("latitude") or loc.get("lat", "?")
            lon = loc.get("longitude") or loc.get("lon", "?")
            print(f"  [OK]   {src:15s} lat={lat}, lon={lon}")
            any_ok = True
        else:
            err = r.get("error", "desconocido")
            print(f"  [FAIL] {src:15s} {err}")

    print()
    if any_ok:
        print("  Al menos una fuente de ubicacion funciona correctamente.")
    else:
        print("  NINGUNA fuente de ubicacion funciono. Verifica:")
        print("    - Configuracion de privacidad en Windows (Ubicacion activa)")
        print("    - pip install winsdk")
        print("    - Conexion a internet para IP fallback")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnóstico de GPS y ubicación para transport-monitor"
    )
    parser.add_argument(
        "--source",
        choices=["windows", "gpsd", "serial", "ip", "provider", "all"],
        default="all",
        help="Fuente a probar (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostrar salida detallada",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nDiagnóstico GPS - transport-monitor")
    print(f"Sistema: {platform.system()} {platform.version()}")
    print(f"Python : {sys.version}")
    print(f"Fecha  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    src = args.source
    v = args.verbose

    if src in ("windows", "all"):
        results.append(test_windows_location(v))
    if src in ("gpsd", "all"):
        results.append(test_gpsd(v))
    if src in ("serial", "all"):
        results.append(test_serial_gps(v))
    if src in ("ip", "all"):
        results.append(test_ip_geolocation(v))
    if src in ("provider", "all"):
        results.append(test_location_provider(v))

    print_summary(results)


if __name__ == "__main__":
    main()
