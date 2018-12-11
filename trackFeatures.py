import spotipy
import spotipy.util as util
import json
import pandas as pd
import flask
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import csv
import random
from sklearn.utils import shuffle

username = "kchesner98"
clientID = "1bb4ca2adbe14ec299ae000ec92fd78e"
clientSecret = "51f6ca11380a425e95ae29b704ed5069"
redirectURI = "http://google.com/"
SCOPE = "playlist-modify-public"

# prompt for user permissions
token = util.prompt_for_user_token(username,scope = SCOPE,client_id=clientID,client_secret = clientSecret,redirect_uri = redirectURI)

# create spotify object
sp = spotipy.Spotify(auth=token)

# hardcoded identifiers for different genre playlists
# country
c1 = "spotify:playlist:6nU0t33tQA2i0qTI5HiyRV"
c2 = "spotify:playlist:37i9dQZF1DWTkxQvqMy4WW"
c3 = "spotify:playlist:37i9dQZF1DXaiEFNvQPZrM"
# pop
p1 = "spotify:playlist:5uInm2hE4YeoFDPENhLWS9"
p2 = "spotify:playlist:4vnp557oBcIpJ6HSv5IGPK"
# rock
r1 = "spotify:playlist:29RO0worSJjuu77SI7GXoR"
r2 = "spotify:playlist:37i9dQZF1DWXRqgorJj26U"
# jazz
j1 = "spotify:playlist:79DVeXaeZvwfrppldclP7T"
j2 = "spotify:playlist:37i9dQZF1DXbITWG1ZJKYt"
j3 = "spotify:playlist:37i9dQZF1DWTR4ZOXTfd9K"
j4 = "spotify:playlist:37i9dQZF1DX2vYju3i0lNX"
# hip-hop/rap
h1 = "spotify:playlist:4ykrONBhOgPr4sMhAVCoPx"
h2 = "spotify:playlist:1HDS5MxpQEMvZVBl2QAqzC"
h3 = "spotify:playlist:37i9dQZF1DWY4xHQp97fN6"
h4 = "spotify:playlist:37i9dQZF1DX0XUsuxWHRQd"
# edm
e1 = "spotify:playlist:37i9dQZF1DX0hvSv9Rf41p"
e2 = "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n"
e3 = "spotify:playlist:37i9dQZF1DXaXB8fQg7xif"
e4 = "spotify:playlist:37i9dQZF1DX6VdMW310YC7"
e5 = "spotify:playlist:2wxTE8gZCjfBgDHm4ljpaL"
# r&b
b1 = "spotify:playlist:37i9dQZF1DX4SBhb3fqCJd"
b2 = "spotify:playlist:37i9dQZF1DWYmmr74INQlb"
b3 = "spotify:playlist:37i9dQZF1DX6VDO8a6cQME"
b4 = "spotify:playlist:37i9dQZF1DWXnexX7CktaI"

#lists of useful information for collecting data
listOfPlaylistIDs = [c1,c2,c3,p1,p2,j1,j2,j3,j4,h1,h2,h3,h4,e1,e2,e3,e4,e5,b1,b2,b3,b4]
listOfFeatureKeys = ["id","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
listOfTimbreKeys = ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]
listOfPitchKeys = ["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12"]
titleKeys = ["title"]

#listOfFeatureKeys += listOfTimbreKeys
listOfTracks = []  # list of track data

def getTrackFeatures(trackID,playlistID):
    listOfTrackFeatures = []
    features = sp.audio_features(trackID)
    features=features[0]
    for key in listOfFeatureKeys:
        listOfTrackFeatures.append(features[key])
    listOfTrackFeatures += getTimbrePitch(trackID)
    # listOfTrackFeatures += getPitch(trackID)
    listOfTrackFeatures += getTitle(trackID)
    listOfTrackFeatures.append(getGenre(playlistID))
    return listOfTrackFeatures

def getTitle(trackID):
    try:
        return [sp.track(trackID)["name"]]
    except:
        return [""]

def getGenre(playlist):
    if playlist == c1 or playlist == c2 or playlist == c3:
        return "country"
    if playlist == p1 or playlist == p2:
        return "pop"
    if playlist == r1 or playlist == r2:
        return "rock"
    if playlist == j1 or playlist == j2 or playlist == j3 or playlist == j4:
        return "jazz"
    if playlist == h1 or playlist == h2 or playlist == h3 or playlist == h4:
        return "hip-hop/rap"
    if playlist == (e1 or e2 or e3 or e4 or e5):
        return "edm"
    if playlist == b1 or playlist == b2 or playlist == b3 or playlist == b4:
        return "rnb"

def getTimbrePitch(trackId):
    audioAnalysis = sp.audio_analysis(trackId)
    audioSegments = audioAnalysis["segments"]
    timbreVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 timbre attributes
    for s in audioSegments:
        timbre = s["timbre"]
        for i in range(len(timbre)):
            timbreVector[i] += timbre[i]
    for i in range(len(timbreVector)):
        timbreVector[i] /= len(audioSegments)

    pitchVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 pitch attributes
    for s in audioSegments:
        pitch = s["pitches"]
        for i in range(len(pitch)):
            pitchVector[i] += pitch[i]
    for i in range(len(pitchVector)):
        pitchVector[i] /= len(audioSegments)
    return timbreVector + pitchVector

def getPitch(trackId):
    audioAnalysis = sp.audio_analysis(trackId)
    audioSegments = audioAnalysis["segments"]
    pitchVector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 pitch attributes
    for s in audioSegments:
        pitch = s["pitches"]
        for i in range(len(pitch)):
            pitchVector[i] += pitch[i]
    for i in range(len(pitchVector)):
        pitchVector[i] /= len(audioSegments)
    return pitchVector

def iteratePlaylists():
    for playlistID in listOfPlaylistIDs:
        print(78,playlistID)
        try:
            playlist = sp.user_playlist_tracks(username,playlist_id=playlistID,fields=None,limit=None,offset=0,market=None)
            for track in playlist["items"]:
                try:
                    trackFeatures = getTrackFeatures(track["track"]["id"],playlistID)
                    listOfTracks.append(trackFeatures)
                except:
                    continue
        except:
            continue
    listOfTracks.insert(0,listOfFeatureKeys + listOfTimbreKeys + listOfPitchKeys + titleKeys + ["Genre Class"])
    df = pd.DataFrame(listOfTracks)
    df.to_csv('test.csv', index=False, header=False)

def train_with_kFold():
    from copy import deepcopy
    plotPointsCount = []
    plotPointsPrecision = []
    dataset = pd.read_csv("final_noid.csv", names=(listOfFeatureKeys[1:] + listOfTimbreKeys + listOfPitchKeys + ["Genre Class"]))
    dataset = shuffle(dataset)
    datasetSize = len(dataset)-1
    i = 1
    while i < datasetSize-1:
        datasetSub = deepcopy(dataset)
        x = pd.DataFrame()
        y = pd.DataFrame()
        for j in range(i):
            ind = random.randint(1,len(datasetSub)-1)
            datasetSub = datasetSub.drop(datasetSub.index[ind])
            x = datasetSub.iloc[:, :-2].values  # attribute values
            y = datasetSub.iloc[:, (len(listOfFeatureKeys) + len(listOfTimbreKeys) + len(listOfPitchKeys)) - 1].values  # genre class
        plotPointsCount.append(datasetSize-i)
        plotPointsPrecision.append(kFold(x,y))
        i+=25

    import matplotlib.pyplot as plt
    plt.scatter(plotPointsCount,plotPointsPrecision)
    plt.title('Accuracy vs. Dataset Size')
    plt.xlabel('Dataset Size')
    plt.ylabel('Accuracy')
    plt.show()

def kFold(X,y):
    from sklearn.ensemble import RandomForestClassifier
    rfc_model = RandomForestClassifier()
    print(145, len(X), len(y))
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    rfc_model.fit(X_train, y_train)
    rfc_predictions = rfc_model.predict(X_test)
    accuracy = 0
    for i in range(len(y_test)):
        if y_test[i] == rfc_predictions[i]:
            accuracy += 1
    return accuracy / len(y_test)

def randomForestTraining(x,y):
    from sklearn.ensemble import RandomForestClassifier
    rfc_model = RandomForestClassifier()
    print(145,len(x),len(y))
    # dataset = pd.read_csv("trainingWithTimbrePitchCut.csv", names=(listOfFeatureKeys[1:] + listOfTimbreKeys + listOfPitchKeys+ ["Genre Class"]))
    # x = dataset.iloc[:, :-2].values  # attribute values
    # print(len(dataset))
    # print(dataset.iloc[10])
    # y = dataset.iloc[:, (len(listOfFeatureKeys) + len(listOfTimbreKeys) + len(listOfPitchKeys)) - 1].values  # genre class
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    rfc_model.fit(x_train, y_train)
    rfc_predictions = rfc_model.predict(x_test)
    accuracy = 0
    for i in range(len(y_test)):
        if y_test[i] == rfc_predictions[i]:
            accuracy += 1
    return accuracy/len(y_test)

def train_rf_with_subset():
    from copy import deepcopy
    plotPointsCount = []
    plotPointsPrecision = []
    dataset = pd.read_csv("trainingWithTimbrePitchCut.csv", names=(listOfFeatureKeys[1:] + listOfTimbreKeys + listOfPitchKeys+ ["Genre Class"]))
    datasetSize = len(dataset) -1
    i = 25
    while i < datasetSize-1:
        datasetSub = deepcopy(dataset)
        x = pd.DataFrame()
        y = pd.DataFrame()
        for j in range(i):
            ind = random.randint(1,len(datasetSub)-1)
            datasetSub = datasetSub.drop(datasetSub.index[ind])
            x = datasetSub.iloc[:, :-2].values  # attribute values
            y = datasetSub.iloc[:, (len(listOfFeatureKeys) + len(listOfTimbreKeys) + len(listOfPitchKeys)) - 1].values  # genre class
        plotPointsCount.append(datasetSize-i)
        plotPointsPrecision.append(randomForestTraining(x,y))
        i+=50

    import matplotlib.pyplot as plt
    plt.scatter(plotPointsCount,plotPointsPrecision)
    plt.title('Accuracy vs. Training Size')
    plt.xlabel('Size')
    plt.ylabel('Accuracy')
    plt.show()

#iteratePlaylists()
#train_rf_with_subset()
train_with_kFold()
#randomForestTraining()

