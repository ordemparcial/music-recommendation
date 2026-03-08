# ETL for Music Recommendation

This ETL pipeline uses two Kaggle datasets to generate recommendations based on user preferences.

The [first dataset](https://www.kaggle.com/datasets/saichaitanyareddyai/spotify-tracks-dataset-audio-features) contains music features such as danceability, energy, and loudness. It is used to create a model that outputs a list of songs similar to the one we are interested in, based on NearestNeighbors. The [second dataset](https://www.kaggle.com/datasets/andrewmvd/spotify-playlists) contains songs that certain users shared online, and it is used to infer user preferences.

The loaded recommendations can be seen [here](https://www.kaggle.com/datasets/fabolaloterio/recommendation-dataset).


