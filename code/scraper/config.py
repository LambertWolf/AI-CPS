CORRIDOR_STOPS = {
"900003201": "Berlin Hbf",
"900003102": "S Bellevue",
"900003103": "S Tiergarten",
"900023201": "S+U Zoologischer Garten",
"900024203": "Savignyplatz",
"900024101": "S Charlottenburg Bhf",
"900024102": "S Westkreuz",
"900048101": "Grunewald",
"900052201": "Nikolassee",
"900053301": "Wannsee Bhf",
"900230003": "Griebnitzsee Bhf",
"900230000": "Babelsberg",
"900230999": "Potsdam Hbf"
}

BERLIN_HBF = "900003201"
POTSDAM_HBF = "900230999"

VBB_BASE_URL = "https://vbb-demo.demo2.hafas.cloud/api/fahrinfo/2.49"

VBB_ACCESS_ID = "wolf-4af9-884e-91a1fe245e62"

#OUTPUT_CSV =

CSV_FIELDS =  [
    "collected_at_utc",
    "querschnitt_id",
    "trip_id",
    "stop_id",
    "departure_time_planned",
    "departure_time_real",
    "trip_headsign"
]
#später --> immer den letzten vorhergesagten Wert eines Trips nehmen 