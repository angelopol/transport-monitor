import os
import sqlite3
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def extract_collages(db_path="data/transport_events.db", output_dir="data/debug_collages"):
    """
    Extrae las imágenes de collages almacenados en tracker_collages 
    y las guarda como archivos individuales.
    """
    db_file = Path(db_path)
    out_path = Path(output_dir)

    if not db_file.exists():
        logging.error(f"La base de datos no existe en: {db_file.absolute()}")
        return

    # Crear directorio de salida si no existe
    out_path.mkdir(parents=True, exist_ok=True)

    try:
        # Conectar a SQLite
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Verificar que la tabla exista
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tracker_collages'")
        if not cursor.fetchone():
            logging.error("No se encontró la tabla 'tracker_collages' en la base de datos.")
            return

        # Extraer collages
        cursor.execute("SELECT idx, image_data FROM tracker_collages")
        rows = cursor.fetchall()

        if not rows:
            logging.info("La tabla 'tracker_collages' está vacía. No hay imágenes para extraer.")
            return

        extracted_count = 0
        for idx, image_data in rows:
            if not image_data:
                continue

            # Construir ruta del archivo (asumiendo formato jpeg)
            image_filename = out_path / f"collage_{idx}.jpg"
            
            with open(image_filename, "wb") as f:
                f.write(image_data)
                
            extracted_count += 1
            logging.info(f"Extraído: {image_filename}")

        logging.info(f"Se extrajeron {extracted_count} collages exitosamente en {out_path.absolute()}")

    except sqlite3.Error as e:
        logging.error(f"Error de SQLite: {e}")
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    extract_collages()
