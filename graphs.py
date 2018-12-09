import matplotlib.pyplot as plt
import pandas as pd

listOfFeatureKeys = ["id","danceability","energy","key","loudness","mode","speechiness","acousticness","instrumentalness","liveness","valence","tempo","duration_ms","time_signature"]
listOfTimbreKeys = ["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]
listOfPitchKeys = ["p1","p2","p3","p4","p5","p6","p7","p8","p9","p10","p11","p12"]
titleKeys = ["title"]


data = pd.read_csv("trainingWithTimbrePitch.csv")
for key in listOfFeatureKeys[1:]:
    data.boxplot(key,"Genre Class")
plt.show()
