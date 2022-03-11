from sklearn.neighbors import DistanceMetric
import pandas as pd
import numpy as np

def cal_dist(dfo, lat, lon, neighborhood):
    df = dfo.copy()
    df = df.drop_duplicates(subset=[neighborhood])
    df.loc[:,lat] = np.radians(df.loc[:,lat])
    df.loc[:,lon] = np.radians(df.loc[:,lon])
    dist = DistanceMetric.get_metric('haversine')
    distances = pd.DataFrame(dist.pairwise(df[[lat,lon]].to_numpy())*6373,  columns=df[neighborhood].unique(), index=df[neighborhood].unique())
    return distances

def cal_depot(dfo, lat, lon, neighborhood, Cendis):
    df = dfo.copy()
    df = df.drop_duplicates(subset=[neighborhood])
    # approximate radius of earth in km
    R = 6373.0

    Cendis = np.radians(Cendis)
    df.loc[:,lat] = np.radians(df.loc[:,lat])
    df.loc[:,lon] = np.radians(df.loc[:,lon])

    df["dlon"] = df[lon] - Cendis[0]
    df["dlat"] = df[lat] - Cendis[1]

    df["a"] = np.sin(df["dlat"] / 2)**2 + np.cos(Cendis[1]) * np.cos(df[lat]) * np.sin(df["dlon"] / 2)**2
    df["c"] = 2 * np.arctan2(np.sqrt(df["a"]), np.sqrt(1 - df["a"]))

    df["CENDIS"] = R * df["c"]

    df_distances = df.loc[:,(neighborhood,"CENDIS")] #km
    df_distances = df_distances.set_index(neighborhood)

    return df_distances