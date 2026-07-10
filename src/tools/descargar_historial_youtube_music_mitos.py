"""Auto-generado por MITOS ToolBuilder.

  Operador pidió: descargar el historial de youtube music mitos
  Proveedor: gemini (gemini-3.5-flash)
  Generado: 2026-06-21 10:34:10
  Válido Python: True
"""

# pip install ytmusicapi

import json
from pathlib import Path
from typing import Any
from ytmusicapi import YTMusic


def download_ytmusic_history(
    auth_cookie_path: str | Path,
    output_json_path: str | Path = "ytmusic_history.json",
    filter_keyword: str | None = None,
) -> list[dict[str, Any]]:
    """Descarga el historial de reproducción de YouTube Music utilizando credenciales de usuario.

    Suposición sobre 'mitos':
    Se asume que 'mitos' es un error de digitación para 'mi' (mi historial) o 'hitos'.
    No obstante, se incluye un parámetro `filter_keyword` por si se desea filtrar
    específicamente canciones o álbumes que contengan la palabra 'mitos' o similar.

    Para usar esta función, se requiere un archivo de autenticación (headers_auth.json).
    Instrucciones para obtenerlo:
    1. Abre YouTube Music en tu navegador con tu sesión iniciada.
    2. Abre las herramientas de desarrollador (F12) -> pestaña Red (Network).
    3. Busca una petición como 'browse' o 'v1/browse'.
    4. Copia los Request Headers (Cabeceras de petición).
    5. Ejecuta en tu terminal: `ytmusicapi setup` y pega las cabeceras,
       o usa `YTMusic.setup(filepath="headers_auth.json", headers_raw="...")`.

    Args:
        auth_cookie_path: Ruta al archivo JSON con las cabeceras de autenticación.
        output_json_path: Ruta donde se guardará el historial descargado en formato JSON.
        filter_keyword: Palabra clave opcional para filtrar el historial (ej. "mitos").

    Returns:
        Una lista de diccionarios con el historial de canciones obtenidas.

    Raises:
        FileNotFoundError: Si el archivo de autenticación no existe.
        ValueError: Si ocurre un error con el formato de autenticación.
        RuntimeError: Para fallos generales en la comunicación con la API.
    """
    auth_path = Path(auth_cookie_path)
    if not auth_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de autenticación en: {auth_path.resolve()}.\n"
            "Por favor, sigue las instrucciones en el docstring para generarlo."
        )

    print(f"[*] Inicializando cliente de YouTube Music con: {auth_path.name}...")
    try:
        # Inicializar la API con el archivo de cabeceras autenticadas
        ytm = YTMusic(str(auth_path))
    except Exception as e:
        raise ValueError(
            f"Error al inicializar YTMusic con el archivo provisto: {e}"
        ) from e

    print("[*] Descargando historial de reproducción...")
    try:
        # Obtener el historial del usuario autenticado
        raw_history = ytm.get_history()
    except Exception as e:
        raise RuntimeError(
            f"Error al conectar con YouTube Music. Verifica tus credenciales: {e}"
        ) from e

    processed_history: list[dict[str, Any]] = []

    # Procesar y limpiar la respuesta de la API
    for item in raw_history:
        # Extraer información relevante de forma segura
        title = item.get("title", "Desconocido")
        artists = ", ".join(
            [
                artist.get("name", "Desconocido")
                for artist in item.get("artists", [])
            ]
        )
        album = item.get("album", {}).get("name", "N/A") if item.get("album") else "N/A"
        video_id = item.get("videoId", "")
        duration = item.get("duration", "N/A")

        song_entry = {
            "title": title,
            "artists": artists,
            "album": album,
            "video_id": video_id,
            "duration": duration,
            "play_url": f"https://music.youtube.com/watch?v={video_id}"
            if video_id
            else "",
        }

        # Aplicar filtro si se especifica (ej. "mitos")
        if filter_keyword:
            keyword = filter_keyword.lower()
            in_title = keyword in title.lower()
            in_artists = keyword in artists.lower()
            in_album = keyword in album.lower()
            if not (in_title or in_artists or in_album):
                continue  # Omitir si no coincide con el filtro

        processed_history.append(song_entry)

    # Guardar los resultados en un archivo JSON local
    output_path = Path(output_json_path)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_history, f, indent=4, ensure_ascii=False)
        print(
            f"[+] Historial guardado exitosamente ({len(processed_history)} elementos) en: {output_path.resolve()}"
        )
    except IOError as e:
        print(f"[-] No se pudo escribir el archivo de salida: {e}")

    return processed_history


if __name__ == "__main__":
    # Ejemplo de uso ejecutable
    # Nota: Reemplaza 'headers_auth.json' con tu archivo de credenciales real.
    auth_file = Path("headers_auth.json")

    if not auth_file.exists():
        print("=== INSTRUCCIONES DE CONFIGURACIÓN ===")
        print("Para descargar tu historial real, necesitas autenticarte.")
        print("1. Instala ytmusicapi: pip install ytmusicapi")
        print("2. Crea un archivo llamado 'headers_auth.json' siguiendo esto:")
        print("   https://ytmusicapi.readthedocs.io/en/stable/setup.html")
        print("\nCreando un archivo de plantilla 'headers_auth.json' para pruebas...")

        # Creamos una plantilla vacía para que el usuario sepa dónde va
        dummy_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
            "Cookie": "PASTE_YOUR_COOKIE_HERE",
        }
        with open(auth_file, "w", encoding="utf-8") as f:
            json.dump(dummy_headers, f, indent=4)

        print(
            f"Plantilla creada en {auth_file.resolve()}. Rellénala antes de continuar."
        )
    else:
        try:
            # Intentar descargar el historial completo
            historial = download_ytmusic_history(
                auth_cookie_path=auth_file,
                output_json_path="mi_historial_ytmusic.json",
            )

            # Ejemplo filtrando por un término específico (ej. "mitos")
            # historial_mitos = download_ytmusic_history(
            #     auth_cookie_path=auth_file,
            #     output_json_path="historial_mitos.json",
            #     filter_keyword="mitos"
            # )

            if historial:
                print("\nPrimeras 3 canciones de tu historial:")
                for i, song in enumerate(historial[:3], 1):
                    print(
                        f"{i}. {song['title']} - {song['artists']} (Álbum: {song['album']})"
                    )
        except Exception as err:
            print(f"\n[-] Ocurrió un error durante la ejecución: {err}")