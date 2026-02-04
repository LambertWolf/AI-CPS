from config import VBB_ACCESS_ID, VBB_BASE_URL, CORRIDOR_STOPS
import requests
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import csv
import os
import json

# ✅ Neu: Retry Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

BERLIN_TZ = ZoneInfo("Europe/Berlin")

# WICHTIG für Cron: Dateipfade immer relativ zum Skript-Ordner!
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_FILE = os.path.join(SCRIPT_DIR, "s7_delays.csv")

# Persistenter Buffer für collected_data über Cron-Runs hinweg
BUFFER_FILE = os.path.join(SCRIPT_DIR, "collected_data.jsonl")

# Persistenter Dedup-State: welche Datensätze schon in CSV sind
SEEN_FILE = os.path.join(SCRIPT_DIR, "seen_keys.json")

# Liste für gesammelte Datenpunkte (nur pro aktuellem Run im RAM)
collected_data = []


# -------------------------------------------------------------------
# ✅ Requests Session + Retry (gegen InvalidChunkLength/ChunkedEncodingError)
# -------------------------------------------------------------------

DEFAULT_HEADERS = {
    "Connection": "close",          # hilft oft gegen kaputte Keep-Alive Streams
    "Accept-Encoding": "identity",  # optional: vermeidet manche Chunk/Gzip Edgecases
    "User-Agent": "s7-delay-collector/1.0",
}

def build_retry_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.7,  # 0.7s, 1.4s, 2.8s, ...
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

SESSION = build_retry_session()



# Persistenz / Dedup Helpers


def make_key(dp: dict) -> str:
    """
    Eindeutiger Key pro Datensatz.
    Damit wird derselbe Stop nicht mehrfach in CSV geschrieben.
    """
    return f"{dp['datum']}|{dp['trip_id']}|{dp['stop_id']}|{dp['actual_time']}"


def load_seen_keys() -> set:
    """Lädt die bereits geschriebenen keys aus SEEN_FILE."""
    if not os.path.exists(SEEN_FILE):
        return set()
    try:
        with open(SEEN_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    except Exception as e:
        logger.error(f"seen_keys konnte nicht geladen werden: {e}")
        return set()


def save_seen_keys(seen_keys: set) -> None:
    """Speichert die bereits geschriebenen keys in SEEN_FILE."""
    try:
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(seen_keys)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"seen_keys konnte nicht gespeichert werden: {e}")


def buffer_append(data_point: dict) -> None:
    """
    Speichert einen Datapoint sofort in den persistenten Buffer (JSONL),
    sodass collected_data über Programmende hinaus existiert.
    """
    try:
        with open(BUFFER_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data_point, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Fehler beim Schreiben in BUFFER_FILE: {e}")


def buffer_load_all() -> list:
    """Lädt alle gepufferten Datapoints aus JSONL."""
    if not os.path.exists(BUFFER_FILE):
        return []

    data = []
    try:
        with open(BUFFER_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Fehler beim Laden aus BUFFER_FILE: {e}")
        return []

    return data


def buffer_clear() -> None:
    """Leert den Buffer nach erfolgreichem CSV-Write."""
    try:
        open(BUFFER_FILE, "w", encoding="utf-8").close()
    except Exception as e:
        logger.error(f"Fehler beim Leeren des BUFFER_FILE: {e}")



# API Calls


def getdepartureBoard(station_id, products=1, duration=120, lookback_minutes=25):
    """
    Hole Abfahrten von einer Station.

    Wichtig:
    departureBoard liefert sonst fast nur Züge in der Zukunft.
    Wir setzen deshalb Startzeit = (jetzt - lookback_minutes),
    damit auch Züge drin sind, die gerade eben abgefahren sind.
    """
    url = f"{VBB_BASE_URL}/departureBoard"

    now = datetime.now(BERLIN_TZ)
    start_dt = now - timedelta(minutes=lookback_minutes)

    params = {
        "accessId": VBB_ACCESS_ID,
        "id": station_id,
        "products": products,
        "duration": duration,
        "format": "json",

        # Lookback aktivieren:
        "date": start_dt.strftime("%Y-%m-%d"),
        "time": start_dt.strftime("%H:%M"),
    }

    try:
        # ✅ Neu: SESSION + Timeout + Headers
        response = SESSION.get(
            url,
            params=params,
            timeout=(3.05, 20),
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        data = response.json()

        if "Departure" not in data:
            logger.warning("Keine 'Departure' in Response")
            return None

        return data["Departure"]

    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        logger.error(f"departureBoard HTTP Error {status}")
        return None

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ) as e:
        logger.warning(f"departureBoard temporärer Netzwerkfehler: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return None


def getJourneyDetail(trip_id, date):
    url = f"{VBB_BASE_URL}/journeyDetail"
    params = {"accessId": VBB_ACCESS_ID, "id": trip_id, "date": date, "format": "json"}

    try:
        # ✅ Neu: SESSION + Timeout + Headers
        response = SESSION.get(
            url,
            params=params,
            timeout=(3.05, 20),
            headers=DEFAULT_HEADERS,
        )
        response.raise_for_status()
        data = response.json()

        if not data:
            logger.error(f"[journeyDetail] Leere Response für trip_id={trip_id}")
            return None

        return data

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.HTTPError,
    ) as e:
        logger.warning(f"[journeyDetail] temporärer Netzwerk/HTTP Fehler: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in der Abfrage:{e}")
        return None


def extract_stop_id(full_stop_id):
    """
    Extrahierung der kurzen ID aus der kompletten VBB ID.
    Wir brauchen nur die Zahl nach @L=...
    """
    if not full_stop_id:
        return None

    if "@L=" in full_stop_id:
        try:
            parts = full_stop_id.split("@L=")
            if len(parts) > 1:
                return parts[1].split("@")[0]
        except Exception:
            logger.warning(f"Konnte Stop-ID nicht extrahieren: {full_stop_id}")
            return None

    return full_stop_id


def calc_delay(planned_time, actual_time):
    """
    Berechnung der Verspätung in Minuten.
    """
    try:
        fmt = "%H:%M:%S"
        planned = datetime.strptime(planned_time, fmt)
        actual = datetime.strptime(actual_time, fmt)
        diff = (actual - planned).total_seconds() / 60
        return int(diff)
    except Exception as e:
        logger.error(f"Fehler bei Delay Berechnung: {e}")
        return 0



# Daten sammeln


def collect_data():
    """
    Sammelt Datenpunkte und speichert sie:
      im RAM in collected_data (nur dieser Run)
      persistent im BUFFER_FILE (über Cron-Runs hinweg)
    """
    global collected_data

    timestamp = datetime.now(BERLIN_TZ)
    print(f"\n{'='*70}")
    print(f"Sammlung gestartet: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    result = getdepartureBoard("900003201", lookback_minutes=25)
    if not result:
        print("Keine Abfahrten gefunden.")
        return

    # Filter S7
    s7_list = [d for d in result if d.get("ProductAtStop", {}).get("line") == "S7"]
    print(f"{len(s7_list)} S7 gefunden\n")

    if not s7_list:
        return

    for s7 in s7_list:
        trip_id = s7["JourneyDetailRef"]["ref"]
        datum = s7["date"]
        s7_rt_time = s7.get("rtTime")

        # Filter 1: Echtzeit muss existieren
        if s7_rt_time is None:
            continue

        # Filter 2: Zug muss Berlin Hbf schon verlassen haben
        try:
            s7_rt_datetime = datetime.strptime(
                f"{datum} {s7_rt_time}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=BERLIN_TZ)

            if s7_rt_datetime > timestamp:
                continue
        except Exception:
            continue

        journey = getJourneyDetail(trip_id, datum)
        if not journey:
            continue

        stops = journey.get("Stops", {}).get("Stop", [])

        # Corridor Stops filtern
        corridor_stops = []
        for stop in stops:
            clean_id = extract_stop_id(stop.get("id"))
            if clean_id and clean_id in CORRIDOR_STOPS:
                corridor_stops.append((stop, clean_id))

        for stop, clean_id in corridor_stops:
            stop_name = CORRIDOR_STOPS.get(clean_id, "Unbekannt")

            planned = stop.get("depTime", "N/A")
            actual = stop.get("rtDepTime")

            # Filter: Stop braucht Echtzeit
            if actual is None:
                continue

            # Filter: Stop muss abgefahren sein
            try:
                actual_datetime = datetime.strptime(
                    f"{datum} {actual}", "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=BERLIN_TZ)

                if actual_datetime > timestamp:
                    continue
            except Exception:
                continue

            delay = 0
            if planned != "N/A":
                delay = calc_delay(planned, actual)

            data_point = {
                "timestamp": timestamp.isoformat(),
                "trip_id": trip_id,
                "datum": datum,
                "stop_id": clean_id,
                "stop_name": stop_name,
                "planned_time": planned,
                "actual_time": actual,
                "delay_minutes": delay,
            }

            # 1) im RAM sammeln (nur für diesen Run)
            collected_data.append(data_point)

            # 2) persistent in den Buffer schreiben
            buffer_append(data_point)

    print(f"Neue Datapoints gesammelt (RAM): {len(collected_data)}")
    print(f"Buffer-Datei: {BUFFER_FILE} | exists={os.path.exists(BUFFER_FILE)}")



# CSV speichern: dedup + clear()

def save_to_csv():
    """
    Schreibt alle Daten aus dem persistenten Buffer in CSV (append),
    dedupliziert über seen_keys.json und macht danach:
      collected_data.clear()
      buffer_clear()
    """
    global collected_data

    buffered_points = buffer_load_all()
    if not buffered_points:
        print("\nKeine Daten im Buffer zum Speichern.")
        return None

    seen_keys = load_seen_keys()

    # Dedup: nur neue Datensätze schreiben
    new_points = []
    for dp in buffered_points:
        key = make_key(dp)
        if key in seen_keys:
            continue
        dp["_key"] = key
        new_points.append(dp)

    if not new_points:
        print("\nKeine neuen (deduplizierten) Daten zum Speichern.")
        buffer_clear()  # alles war schon gespeichert -> Buffer kann geleert werden
        return CSV_FILE

    file_exists = os.path.isfile(CSV_FILE)

    fieldnames = [
        "timestamp",
        "trip_id",
        "datum",
        "stop_id",
        "stop_name",
        "planned_time",
        "actual_time",
        "delay_minutes",
    ]

    try:
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerows([{k: dp[k] for k in fieldnames} for dp in new_points])

        # Nach erfolgreichem Schreiben: keys merken
        for dp in new_points:
            seen_keys.add(dp["_key"])
        save_seen_keys(seen_keys)

        print(f"\n{len(new_points)} neue Datenpunkte gespeichert in: {CSV_FILE}")

        # WICHTIG: Liste leeren (wie gewünscht)
        collected_data.clear()

        # WICHTIG: Buffer leeren (sonst würde Cron später alles erneut laden)
        buffer_clear()

        return CSV_FILE

    except Exception as e:
        logger.error(f"Fehler beim Speichern der CSV: {e}")
        return None



def main_request():
    """
    Genau 1x ausführen pro Cron-Run:
      1) collect_data -> schreibt in Buffer (persistent)
      2) save_to_csv  -> dedup + schreibt neue Datensätze + clear()
    """
    collect_data()
    save_to_csv()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main_request()
