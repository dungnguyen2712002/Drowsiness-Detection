import os
import glob
import pandas as pd
import numpy as np
import math

face_points_to_keep = []
face_points_to_keep += [9]                     # Nose
face_points_to_keep += [37,38,39,40,41,42]     # Left Eye
face_points_to_keep += [43,44,45,46,47,48]     # Right Eye
face_points_to_keep += [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59] # Outer Lip

columns_to_keep = ['participant', 'mood', 'time'] + \
                    [f'px_{x}' for x in face_points_to_keep] + \
                    [f'py_{x}' for x in face_points_to_keep] +\
                    ['face_x','face_y','face_w','face_h']

def distance(table, a, b):
    return np.sqrt((table[f'px_{a}'] - table[f'px_{b}']) ** 2
                   + (table[f'py_{a}'] - table[f'py_{b}']) ** 2)

def eye_aspect_ratio(table):
	ear = (distance(table, 38, 42) + distance(table, 39, 41)) \
                / (2 * distance(table, 37, 40))
	return ear

def mouth_aspect_ratio(table):
    A = distance(table, 52, 58)
    C = distance(table, 49, 55)
    mar = (A ) / (C)
    return mar

def circularity(table):
    A = distance(table, 38, 41)
    radius  = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance(table, 37, 38)
    p += distance(table, 38, 39)
    p += distance(table, 39, 40)
    p += distance(table, 40, 41)
    p += distance(table, 41, 42)
    p += distance(table, 42, 37)
    return 4 * math.pi * Area /(p**2)

def mouth_over_eye(table):
    ear = eye_aspect_ratio(table)
    mar = mouth_aspect_ratio(table)
    mouth_eye = mar/ear
    return mouth_eye


def get_table(participant, mood, start_time=61, stop_time=361, resample_interval='100ms',
              base_path=None):
    # Find File
    if base_path is None:
        base = os.path.join('output', 'csv')
    else:
        base = base_path

    files = glob.glob(os.path.join(base, f'{participant}_{mood}.csv'))

    # Load
    table = pd.read_csv(files[0])

    # Resample time
    table['date'] = pd.to_datetime(table.time, unit='s')
    if resample_interval is not None:
        table = table.resample(resample_interval, on='date').mean()
    else:
        table.set_index('date', inplace=True)

    # Drop columns we don't need
    table = table.filter(columns_to_keep)

    # Trim head and tail of the video
    table.drop(table[table['time'] > stop_time].index, inplace=True)
    table.drop(table[table['time'] < start_time].index, inplace=True)

    # Fill missing data
    table.replace(-1, np.NaN, inplace=True)
    table.interpolate(inplace=True, limit_direction='both')

    # Fix Data Types
    table[['participant', 'mood']] = table[['participant', 'mood']].astype('int32')

    return table


# process all data files
def filter_col(table):
    features = ['mood', 'EAR_N', 'MAR_N', 'PUC_N', 'MOE_N']

    table['EAR'] = eye_aspect_ratio(table)
    table['MAR'] = mouth_aspect_ratio(table)
    table['PUC'] = circularity(table)
    table['MOE'] = mouth_over_eye(table)

    table["EAR_N"] = (table["EAR"] - table["EAR"].mean()) / table["EAR"].std()
    table["MAR_N"] = (table["MAR"] - table["MAR"].mean()) / table["MAR"].std()
    table["PUC_N"] = (table["PUC"] - table["PUC"].mean()) / table["PUC"].std()
    table["MOE_N"] = (table["MOE"] - table["MOE"].mean()) / table["MOE"].std()

    #     table.filter(features)
    return table[features]

# print(get_table(31, 10))