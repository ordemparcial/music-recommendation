import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
import json


# Extract
df_musics = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "saichaitanyareddyai/spotify-tracks-dataset-audio-features",
  "spotify-tracks-dataset-detailed.csv"
)

df_users = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "andrewmvd/spotify-playlists",
  "spotify_dataset.csv",
  pandas_kwargs={'on_bad_lines': 'skip'}
)
df_users.columns = df_users.columns.str.strip().str.replace('"', '')

# Transform

df_musics.dropna(subset=["track_name"], inplace=True)

df_users.drop(columns=["playlistname"], inplace=True)

df_users_temp = df_users.copy()
df_musics_temp = df_musics.copy()

df_users_temp["trackname_clean"] = df_users_temp["trackname"].str.strip().str.lower()
df_users_temp["artist_clean"] = df_users_temp["artistname"].str.strip().str.lower()
df_musics_temp["track_id"] = df_musics_temp["track_id"]
df_musics_temp["track_clean"] = df_musics_temp["track_name"].str.strip().str.lower()
df_musics_temp["artists_clean"] = df_musics_temp["artists"].str.strip().str.lower()

df_users_valid = df_users_temp.merge(
    df_musics_temp[["track_clean", "artists_clean", "track_id"]],
    left_on=["trackname_clean", "artist_clean"],
    right_on=["track_clean", "artists_clean"],
    how='inner'
)

df_users_valid.drop(columns=["trackname_clean", "artist_clean", "track_clean", "artists_clean"], inplace=True)
df_users = df_users_valid

by_users = df_users.groupby("user_id").agg(count=('artistname', 'count'))

df_musics_cleaned = df_musics.set_index("track_id")
df_musics_cleaned = df_musics_cleaned.drop(columns=["artists", "explicit", "album_name", "track_name", "track_genre", "popularity"])

scaler = MinMaxScaler()
scaler.fit(df_musics_cleaned)
scaled_data = scaler.transform(df_musics_cleaned)

neighbors = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(scaled_data)

by_users["recommendation"] = np.nan
user_list = df_users["user_id"].unique().tolist()

for user in user_list:
  user_track_ids = df_users[df_users["user_id"]==user]["track_id"].tolist()
  user_track_ids = list(set(user_track_ids))
  user_track_ids = [str(x) for x in user_track_ids]

  listened_features = df_musics_cleaned.loc[user_track_ids]
  average_music = listened_features.mean().values.reshape(1, -1)
  average_music = scaler.transform(average_music)
  indices_in_scaled_data = neighbors.kneighbors(average_music, n_neighbors=10, return_distance=False)

  recommended_track_ids_ordered = df_musics_cleaned.iloc[indices_in_scaled_data[0]].index.tolist()

  recommended_track_name = None
  recommended_track_artist = None
  for rec_track_id in recommended_track_ids_ordered:
    if rec_track_id not in user_track_ids:
      track_info = df_musics[df_musics['track_id'] == rec_track_id]
      if not track_info.empty:
        recommended_track_name = track_info['track_name'].iloc[0]
        recommended_track_artist = track_info['artists'].iloc[0]
        break
  if recommended_track_name is None:
    track_info = df_musics[df_musics['track_id'] == recommended_track_ids_ordered[0]]
    recommended_track_name = track_info['track_name'].iloc[0]
    recommended_track_artist = track_info['artists'].iloc[0]

  by_users.loc[user, "recommendation"] = f"{recommended_track_name} by {recommended_track_artist}"

# Load

kaggle_token = userdata.get('KAGGLE_API_TOKEN')
kaggle_username = userdata.get('KAGGLE_USERNAME')
KAGGLE_DIR = os.path.join(os.path.expanduser('~'), '.kaggle')
os.makedirs(KAGGLE_DIR, exist_ok=True)

kaggle_json_path = os.path.join(KAGGLE_DIR, 'kaggle.json')
with open(kaggle_json_path, 'w') as f:
    json.dump({"username": kaggle_username, "key": kaggle_token}, f)

os.chmod(kaggle_json_path, 0o600)

with tempfile.TemporaryDirectory() as temp_dir:
  csv_path = os.path.join(temp_dir, 'by_users_dataset.csv')
  by_users.to_csv(csv_path, index=False)

  dataset_name = "recommendation-dataset"
  dataset_id = f"{kaggle_username}/{dataset_name}"

  metadata_content = {
      "title": "Recommendation Dataset",
      "id": dataset_id,
      "licenses": [{"name": "CC0-1.0"}],
      "resources": [
          {
              "path": os.path.basename(csv_path),
              "description": f"Individualized music recommendation based on musics that were played by the user in Spotify. Contains {by_users.shape[0]} rows and {by_users.shape[1]} columns."
          }
      ]
  }
  metadata_file_path = os.path.join(temp_dir, 'dataset-metadata.json')
  with open(metadata_file_path, 'w') as f:
    json.dump(metadata_content, f, indent=2)

  !kaggle datasets create -p {temp_dir}




