import pandas as pd
import numpy as np


# function will get data from 2 csv files (from each camera), and will combine the track feature list into a 2x150 np array, based on their same frameNos.
def combine_match(filename1, filename2):
    df1 = pd.read_csv(filename1, sep=',', header=["trackplot ID", "time-in", "time-out", "frame-in", "frame-out", "feature signal"], index=False)
    print(df1.head())
    df2 = pd.read_csv(filename2, sep=',', header=["trackplot ID", "time-in", "time-out", "frame-in", "frame-out", "feature signal"], index=False)
    print(df2.head())
    camera1_tracks_list = df1["trackplot ID"]
    camera2_tracks_list = df2["trackplot ID"]
    print(camera1_tracks_list)
    print(camera2_tracks_list)
    num = 5     # num of comparisons for matched trackIDs
    #for index in range(num):
        # for track_ID1 in camera1_tracks_list:
        #     for track_ID2 in camera2_tracks_list




if __name__ == '__main__':
    filename1 = '../data/vidtest3_target_data.csv'
    filename2 = '../data/vidtest4_target_data.csv'
    combine_match(filename1, filename2)