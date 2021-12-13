import os
import glob
import pandas as pd
import numpy as np
import math

face_points_to_keep = []
face_points_to_keep += [9]  # Nose
face_points_to_keep += [37, 38, 39, 40, 41, 42]  # Left Eye
face_points_to_keep += [43, 44, 45, 46, 47, 48]  # Right Eye
face_points_to_keep += [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # Outer Lip

columns_to_keep = ['mood'] + \
                  [f'px_{x}' for x in face_points_to_keep] + \
                  [f'py_{x}' for x in face_points_to_keep] + \
                  ['face_x', 'face_y', 'face_w', 'face_h']


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
    mar = (A) / (C)
    return mar


def circularity(table):
    A = distance(table, 38, 41)
    radius = A / 2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += distance(table, 37, 38)
    p += distance(table, 38, 39)
    p += distance(table, 39, 40)
    p += distance(table, 40, 41)
    p += distance(table, 41, 42)
    p += distance(table, 42, 37)
    return 4 * math.pi * Area / (p ** 2)


def mouth_over_eye(table):
    mouth_eye = table['MAR'] / table['EAR']
    return mouth_eye


def get_table(index, base_path=None):
    # Find File
    if base_path is None:
        base = 'CSV_DATA'
    else:
        base = base_path

    files = glob.glob(os.path.join(base, f'{index}.csv'))

    # Load
    table = pd.read_csv(files[0])

    # Drop columns we don't need
    table = table.filter(columns_to_keep)

    table = table[::3]

    # Fill missing data
    table.replace(-1, np.NaN, inplace=True)
    table.interpolate(inplace=True, limit_direction='both')

    # Fix Data Types
    table[['mood']] = table[['mood']].astype('int32')

    return table


# process all data files
def filter_col(table):
    table['EAR'] = eye_aspect_ratio(table)
    table['MAR'] = mouth_aspect_ratio(table)
    table['PUC'] = circularity(table)
    table['MOE'] = mouth_over_eye(table)

    table['MOE'] = table['MOE'].replace([np.inf, -np.inf], np.nan)
    table.interpolate(inplace=True, limit_direction='both')

    features = ['EAR', 'MAR', 'PUC', 'MOE']
    table_loc = table.iloc[:30, :]

    for j in features:
        table[f'{j}_mean'] = table_loc[f'{j}'].mean()
        table[f'{j}_std'] = table_loc[f'{j}'].std()
        table[f'{j}_N'] = (table[f'{j}'] - table[f'{j}_mean']) / table[f'{j}_std']

    table = table.filter(['mood', 'EAR', 'MAR', 'PUC', 'MOE', 'EAR_N', 'MAR_N', 'PUC_N', 'MOE_N'])
    return table


# process all data files
def filter_col_live(table):
    table['EAR'] = eye_aspect_ratio(table)
    table['MAR'] = mouth_aspect_ratio(table)
    table['PUC'] = circularity(table)
    table['MOE'] = mouth_over_eye(table)

    table["EAR_N"] = (table["EAR"] - table["EAR"].mean()) / table["EAR"].std()
    table["MAR_N"] = (table["MAR"] - table["MAR"].mean()) / table["MAR"].std()
    table["PUC_N"] = (table["PUC"] - table["PUC"].mean()) / table["PUC"].std()
    table["MOE_N"] = (table["MOE"] - table["MOE"].mean()) / table["MOE"].std()

    table = table.filter(['mood', 'EAR', 'MAR', 'PUC', 'MOE', 'EAR_N', 'MAR_N', 'PUC_N', 'MOE_N'])
    return table

# print(get_table(31, 10))