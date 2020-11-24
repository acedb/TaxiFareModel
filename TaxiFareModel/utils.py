import numpy as np
import pandas as pd

def haversine_vectorized(df,
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df
        Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)),\
        np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)),\
        np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) *\
        np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def extract_time_features(X):
    """ Extracts each cat of datetime out of datetime column """

    timezone_name = 'America/New_York'
    time_column = 'pickup_datetime'
    X.index = pd.to_datetime(X[time_column])
    X.index = X.index.tz_convert(timezone_name)
    X["dow"] = X.index.weekday
    X["hour"] = X.index.hour
    X["month"] = X.index.month
    X["year"] = X.index.year
    return X.reset_index(drop=True)

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())
