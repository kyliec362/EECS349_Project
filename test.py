import spotipy
import spotipy.util as util
import json

username = "kchesner98"
clientID = "1bb4ca2adbe14ec299ae000ec92fd78e"
clientSecret = "51f6ca11380a425e95ae29b704ed5069"
redirectURI = "http://google.com/"
SCOPE = "playlist-modify-public"

# prompt for user permissions
token = util.prompt_for_user_token(username,scope = SCOPE,client_id=clientID,client_secret = clientSecret,redirect_uri = redirectURI)

# create spotify object
sp = spotipy.Spotify(auth=token)
artistID = "spotify:artist:4UXqAaa6dQYAk18Lv7PEgX"
artistAlbums = sp.artist_albums(artistID,album_type=None, country=None, limit=20, offset=0)
someAlbum = artistAlbums["items"][0]
someAlbumTracks = sp.album_tracks(someAlbum["id"],limit=50,offset=0)
for track in someAlbumTracks["items"]:
    #print(track["name"],track["id"])
    #print(json.dumps(sp.audio_analysis(track["id"]),sort_keys=False,indent=4),"\n")
    print(json.dumps(sp.audio_features(track["id"]),sort_keys=False,indent=4),"\n")


# print(json.dumps(artistAlbums["items"],sort_keys=True,indent=4))
# for album in artistAlbums:
#     print(album)
# results = sp.search(q='artist:' + "Fall Out Boy", type='artist')
# print(results)
