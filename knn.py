import spotipy
import spotipy.util as util
import json
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import csv

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
rockPlaylist = "spotify:playlist:29RO0worSJjuu77SI7GXoR"
jazzPlaylist = "spotify:playlist:79DVeXaeZvwfrppldclP7T"
hipHopPlaylist = "spotify:playlist:4ykrONBhOgPr4sMhAVCoPx"
rapPlaylist = "spotify:playlist:1HDS5MxpQEMvZVBl2QAqzC"
edmPlaylist = "spotify:playlist:37i9dQZF1DX0hvSv9Rf41p"
edm2Playlist = "spotify:playlist:37i9dQZF1DX4dyzvuaRJ0n"

#lists of useful information for collecting data
listOfPlaylistIDs = [countryPlaylist,popPlaylist,rockPlaylist,jazzPlaylist,hipHopPlaylist,rapPlaylist,edmPlaylist,edm2Playlist]
listOfFeatureKeys = ["id","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
listOfTracks = []  # list of track data
trackClassifications = []


def getTrackFeatures(trackID,playlistID):
    listOfTrackFeatures = []
    features = sp.audio_features(trackID)
    features=features[0]
    for key in listOfFeatureKeys:
        listOfTrackFeatures.append(features[key])
    #trackClassifications.append(getGenre(playlistID))
    listOfTrackFeatures.append(getGenre(playlistID))
    return listOfTrackFeatures


def getGenre(playlist):
    if playlist == countryPlaylist:
        return "country"
    if playlist == popPlaylist:
        return "pop"
    if playlist == rockPlaylist:
        return "rock"
    if playlist == jazzPlaylist:
        return "jazz"
    if playlist == hipHopPlaylist:
        return "hip hop"
    if playlist == rapPlaylist:
        return "rap"
    # if playlist == edm2Playlist or playlist == edmPlaylist:
    else:
        return "edm"

def iteratePlaylists():
    for playlistID in listOfPlaylistIDs:
        try:
            playlist = sp.user_playlist_tracks(username,playlist_id=playlistID,fields=None,limit=None,offset=0,market=None)
            for track in playlist["items"]:
                trackFeatures = getTrackFeatures(track["track"]["id"],playlistID)
                print(trackFeatures)
                listOfTracks.append(trackFeatures)
        except:
            continue
    listOfTracks.insert(0,listOfFeatureKeys + ["Genre Class"])
    df = pd.DataFrame(listOfTracks)
    df.to_csv('test.csv', index=False, header=False)

def nearestNeighborTraining():
    # Read dataset to pandas dataframe
    dataset = pd.read_csv("train.csv", names=(listOfFeatureKeys[1:] + ["Genre Class"]))
    #print(dataset)
    x = dataset.iloc[:, :-1].values # attribute values
    y = dataset.iloc[:, len(listOfFeatureKeys)-1].values # genre class
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    classifier = KNeighborsClassifier(n_neighbors=20)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred))






#iteratePlaylists()

nearestNeighborTraining()
