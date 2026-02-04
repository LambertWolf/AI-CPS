from config import VBB_ACCESS_ID, VBB_BASE_URL, CORRIDOR_STOPS
import requests
import logging
from datetime import datetime
import time
from zoneinfo import ZoneInfo
import csv
import os

logger = logging.getLogger(__name__)

# Globale Varibale für bereits gesehen Trips
seen_trips = set()
# Liste für gesammelte Datenpunke
collected_data = []


def getdepartureBoard(station_id, products=1, duration=120):
    """
    Hole Abfahrten von einer Station

    Args:
        station_id: "900003201" (Berlin Hbf)collect_data
        products: 1 = S-Bahn
        duration: 60 = nächste 60 Minuten

    Returns:
        list: Response["Departure"] oder None
    """
    url = f"{VBB_BASE_URL}/departureBoard"
    # URL zusammen bauen
    params = {
        "accessId": VBB_ACCESS_ID,
        "id": station_id,
        "products": products,
        "duration": duration,
        "format": "json",
    }
    # 1. departureBoard Endpunkt aufrufen
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # Wirft Error bei Status != 200
        data = response.json()

        # Prüfe ob "Departure" in Response
        if "Departure" not in data:
            logger.warning("Keine 'Departure' in Response")
            return None

        return data["Departure"]

    # Erros abgreifen
    except requests.exceptions.Timeout:
        logger.error("departureBoard Timeout")
        return None

    except requests.exceptions.ConnectionError:
        logger.error(" departureBoard Connection Error")
        return None

    except requests.exceptions.HTTPError as e:
        logger.error(f" HTTP Error {response.status_code}")
        return None

    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        return None


# JourneyDetail bauen
def getJourneyDetail(trip_id, date):

    url = f"{VBB_BASE_URL}/journeyDetail"

    params = {"accessId": VBB_ACCESS_ID, "id": trip_id, "date": date, "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Debug: Was ist in der Response?
        if not data:
            logger.error(f"[journeyDetail] Leere Response für trip_id={trip_id}")
            return None

        logger.debug(f"[journeyDetail] Response Keys: {list(data.keys())}")

        return data

    except Exception as e:
        logger.error(f"Error in der Abfrage:{e}")
        return None


def extract_stop_id(full_stop_id):
    """
    Extrahierung der kurzen ID aus der kompletten VBB ID.

    Format: 'A=1@O=S Potsdam Hauptbahnhof@X=13067160@Y=52391443@U=86@L=900230999@'
    Wir brauchen nur: '900230999'
    """
    if not full_stop_id:
        return None

    if "@L=" in full_stop_id:
        try:
            # Extrahieren der Zahl nach "@L=" und vor derm nächsten "@"
            parts = full_stop_id.split("@L=")
            if len(parts) > 1:
                return parts[1].split("@")[0]
        except Exception as e:
            logger.warning(f"Konnte Stops nicht extrahieren: {full_stop_id}")
            return None
        return full_stop_id


def calc_delay(planned_time, actual_time):
    """
    Berechnung der Verspätung in Minuten

    Args:
    planned_time:"HH:MM:SS (Sollzeit)
    actual_time: "HH:MM:SS" (Istzeit)

    Returns:
    int: Verspätung in Minuten
    """
    try:
        format = "%H:%M:%S"
        planned = datetime.strptime(planned_time, format)
        actual = datetime.strptime(actual_time, format)
        diff = (actual - planned).total_seconds() / 60
        return int(diff)

    except Exception as e:
        logger.error(f"Fehler bei Delay Berechnung: {e}")
    return 0


# Daten sammeln Start:
def collect_data():
    berlin_tz = ZoneInfo("Europe/Berlin")
    timestamp = datetime.now(berlin_tz)
    print(f"\n{'='*70}")
    print(f"Sammlung gestartet: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    print(f"Aktueller Timestamp (Berlin): {timestamp}")

    # Hole Abfahrten aus Berlin Hbf
    result = getdepartureBoard("900003201")
    if result:
        print(f"{len(result)}Abfahrten gefunden")
    else:
        print("Keine Abfahrten gefunden.")
        return

    # Filtere S7
    s7_list = [
        d for d in result if d["ProductAtStop"]["line"] == "S7"
    ]  # d append if d entsprechende Kriterien erfüllt
    print(f"{len(s7_list)} S7 gefunden\n")

    if not s7_list:
        print("Keine S7 gefunden")

    # 4. Hole Journey Details für die letzten S7 Züge
    for s7 in s7_list:
        trip_id = s7["JourneyDetailRef"]["ref"]
        date = s7["date"]
        s7_rt_time = s7.get("rtTime")

        print(f"Prüfe S7 mit trip_id: {trip_id}")
        print(f"   Geplant: {s7.get('time')} | Echtzeit: {s7_rt_time}")

        # Filter 1): Hat S7 bereits Echtzeitinfo?
        if s7_rt_time is None:
            print("Noch keine Echtzeitinfo vorhanden.")
            continue

        # Filter 2): Ist von Berlin Hbf bereits abgefahren?
        try:
            # Kombiniere date + time für Vergleich
            s7_rt_datetime = datetime.strptime(
                f"{date} {s7_rt_time}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=berlin_tz)

            if s7_rt_datetime > timestamp:
                print(f"    Noch nicht abgefahren: überspringe\n")
                continue
        except Exception as e:
            print("Fehler beim Parsen der Zeit")

        # Filter 3) Wurde der trip schon gesammelt?
        if trip_id in seen_trips:
            print(f"    Bereits gesammelt: überspringe\n")
            continue
        # Alle Filter bestanden!
        print(f"   Filter bestanden: sammle Journey Details\n")

        # Hole Journey Details
        journey = getJourneyDetail(trip_id, date)

        if not journey:
            print("Keine Journey Details erhalten")
            continue

        # for debug
        print(f"\nJourney Response Keys: {list(journey.keys())}")

        # 5. Extrahiere Stops
        stops = journey.get("Stops", {}).get("Stop", [])
        print(f"\n{len(stops)} Haltestellen in dieser Fahrt.")

        # 6. Filtere Corridor Stops
        corridor_stops = []
        for stop in stops:
            full_stop_id = stop.get("id")
            clean_id = extract_stop_id(full_stop_id)  # clean Funktion aufrufen

            if clean_id and clean_id in CORRIDOR_STOPS:  # doppelte Prüfung
                corridor_stops.append((stop, clean_id))  # speichern als Tupel

        print(
            f"\n{len(corridor_stops)} Haltestellen im Streckenabschnitt Berlin Hbf - Potsdam Hbf\n"
        )

        # 7. Zeige Corridor Stops mit Delays
        stops_saved = 0

        for stop, clean_id in corridor_stops:
            stop_name = CORRIDOR_STOPS.get(clean_id, "Unbekannt")

            planned = stop.get("depTime", "N/A")
            actual = stop.get("rtDepTime")

            if actual is None:
                print(f"Die Haltestelle {stop_name} hat noch keine Echtzeit.")
                continue

            # Prüfe ob Stop abgefahren
            try:
                actual_datetime = datetime.strptime(
                    f"{date} {actual}", "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=berlin_tz)

                if actual_datetime > timestamp:
                    print(f"  {stop_name}: Noch nicht abgefahren")
                    continue

            except Exception as e:
                print(f"  {stop_name}: Fehler beim Parsen: {e}")
                continue

            # für den ausgefilterten Stop - Zeige Verspätung
            print(f"{stop_name}")
            if actual and planned != "N/A":
                delay = calc_delay(
                    planned, actual
                )  # delay Berechnungs Funktion aufrufen
                delay_str = f"+{delay}" if delay > 0 else str(delay)
                print(
                    f"    Soll: {planned} | Ist: {actual} | Verspätung: {delay_str} min"
                )
                stops_saved += 1

                # Wenn ein Stop gesammelt wird speichere:
                data_point = {
                    "timestamp": timestamp.isoformat(),
                    "trip_id": trip_id,
                    "datum": date,
                    "stop_id": clean_id,
                    "stop_name": stop_name,
                    "planned_time": planned,
                    "actual_time": actual,
                    "delay_minutes": delay,
                }
                collected_data.append(data_point)
            elif actual:
                print(f"    Soll: {planned} | Ist: {actual}")
            else:
                print(f"    Soll: {planned} | Noch nicht erreicht")
            print()

        # Stop als gesehen markieren
        seen_trips.add(trip_id)
        print(f"   {stops_saved} Stops werden aus dem Datensatz gespeichert werden")
        print(f"   Trips im Memory: {len(seen_trips)}\n")
        print("=" * 70 + "\n")


def save_to_csv():
    """Gesammelte Daten in CSV speichern (append mode)"""
    if not collected_data:
        print("\n Keine Daten zum Speichern.")
        return

    filename = "s7_delays.csv"
    file_exists = os.path.isfile(filename)  # ← Prüfe ob existiert

    try:
        with open(filename, "a", newline="", encoding="utf-8") as f:  # ← "a" = append!
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
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:  # ← Header nur beim ersten Mal
                writer.writeheader()

            writer.writerows(collected_data)

        print(
            f" {len(collected_data)} Datenpunkte {'gespeichert' if not file_exists else 'angehängt'} in: {filename}"
        )
        return filename
    except Exception as e:
        logger.error(f"Fehler beim Speichern der CSV: {e}")
        return None


# Loop - den ganzen Tag, alle 5 Minuten (Bahnen fahren im 10 Minuten Takt)

def main_loop():
    """
    Hauptschleife für kontinuierliche Datensammlung

    Args:
        duration_minutes: Wie lange soll gesammelt werden (Standard: 1440 min = 24h)
        interval_minutes: Wie oft abfragen (Standard: 5 min)
    """
    berlin_tz = ZoneInfo("Europe/Berlin")
    start_time = time.time()
    end_time = time.time()


    iteration = 0

    try:
        while iteration <= 1:
            iteration += 1
            print(f"\n{'#'*70}")
            
            print(f"{'#'*70}")

            collect_data()

            # Speichere nach jeder Iteration
            if collected_data:
                save_to_csv()

            # Berechne Zeit bis zur nächsten Abfrage

    except KeyboardInterrupt:
        print("\n\n  ABBRUCH durch Nutzer (Ctrl+C)")

    finally:
        # Finale Speicherung
        if collected_data:
            print(f"\n{'='*70}")
            print(" FINALE SPEICHERUNG...")
            print(f"{'='*70}\n")
            save_to_csv()

        # Zusammenfassung



if __name__ == "__main__":
    # Logging konfigurieren
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    """
    # TEST-MODUS aktiviert
    print("TEST-MODUS: 3 Minuten, alle 1 Minute\n")
    main_loop(duration_minutes=3, interval_minutes=1)
    """
    # PROD-MODUS aktiviert
    print("PROD-MODUS: 24 Stunden, alle 5 Minuten\n")
    main_loop()