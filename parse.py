import csv
import json

# Read JSON file
filename = 'features.json'
f = open(filename).read()
data = json.loads(f)

# Write CSV
c = csv.writer(open("test.csv", "w+"))

for line in data["audio_features"]:
    c.writerow([line["danceability"],
                line["energy"],
                line["key"],
                line["loudness"],
                line["mode"],
                line["speechiness"],
                line["acousticness"],
                line["instrumentalness"],
                line["liveness"],
                line["valence"],
                line["tempo"],
                ])
