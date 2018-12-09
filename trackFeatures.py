import spotipy
import spotipy.util as util
import json
import pandas as pd
import flask
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import csv
import random

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
countryPlaylist = "spotify:playlist:6nU0t33tQA2i0qTI5HiyRV"
popPlaylist = "spotify:playlist:5uInm2hE4YeoFDPENhLWS9"
pop2Playlist = "spotify:playlist:4vnp557oBcIpJ6HSv5IGPK"
rockPlaylist = "spotify:playlist:29RO0worSJjuu77SI7GXoR"
jazzPlaylist = "spotify:playlist:79DVeXaeZvwfrppldclP7T"
hipHopPlaylist = "spotify:playlist:4ykrONBhOgPr4sMhAVCoPx"
rapPlaylist = "spotify:playlist:1HDS5MxpQEMvZVBl2QAqzC"
edmPlaylist = "spotify:playlist:37i9dQZF1DX0hvSv9Rf41p"
edm2Playlist = "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n"
edm3Playlist = "spotify:playlist:37i9dQZF1DXaXB8fQg7xif"
edm4Playlist = "spotify:playlist:37i9dQZF1DX6VdMW310YC7"
edm5Playlist = "spotify:playlist:2wxTE8gZCjfBgDHm4ljpaL"
rnbPlaylist = "spotify:playlist:37i9dQZF1DX4SBhb3fqCJd"

#lists of useful information for collecting data
listOfPlaylistIDs = [countryPlaylist,popPlaylist,rockPlaylist,jazzPlaylist, pop2Playlist, hipHopPlaylist,rapPlaylist,rnbPlaylist,edm3Playlist, edm4Playlist, edm5Playlist, edmPlaylist,edm2Playlist]
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
    if playlist == countryPlaylist:
        return "country"
    if playlist == popPlaylist:
        return "pop"
    if playlist == rockPlaylist:
        return "rock"
    if playlist == pop2Playlist:
        return "pop"
    if playlist == jazzPlaylist:
        return "jazz"
    if playlist == hipHopPlaylist:
        return "rap"
    if playlist == rapPlaylist:
        return "rap"
    if playlist == rnbPlaylist:
        return "rnb"
    # if playlist == edm2Playlist or playlist == edmPlaylist:
    else:
        return "edm"

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
    df.to_csv('trainingWithTimbrePitch.csv', index=False, header=False)



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
train_rf_with_subset()
#randomForestTraining()


