# Genre-Classifier
A genre classifier that uses machine learning techniques and Spotify song data to label songs with genres.

## Machine Learning Model
We use a random forest model that was trained on over a thousand songs.

## Data
To collect training data, we used thousands of songs from various Spotify playlists. Using the Spotify API, we collected over thirty attribute values for each song. Genre classifications were determined based on playlists - for example, we used a large country playlist and classified all songs in it as country.

## Classifying Songs
Based on user queries, we use Spotify to search for possible songs and select the most closely matched song. We then collect the values for each attribute that the model was trained on. We output a classification using the model to predict the genre based on the attribute values.

## Try It!
The project website is http://users.eecs.northwestern.edu/~rma7510/index.html where you can learn more about the process and try out the classifier.
