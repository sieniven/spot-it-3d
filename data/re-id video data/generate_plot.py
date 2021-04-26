import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import math
import time
import csv

fig, axs = plt.subplots(1,2, figsize=(30,15))
fig.suptitle('Kalman Filter')


with open('track_plots_KF3.csv','r') as csvfile: 
    plots = csv.reader(csvfile, delimiter=',')
    while True:
        try:
            track_plot_id = plots.__next__()[0]
            print(track_plot_id)
            track_plot_xs = plots.__next__()
            print(track_plot_xs)
            track_plot_ys = plots.__next__()
            track_plot_frameNo = plots.__next__()

            axs[0].plot(track_plot_xs, track_plot_ys, label='Drone 'str(track_plot_id), color='blue')
            axs[1].plot(track_plot_frameNo, track_plot_ys, label='Drone 'str(track_plot_id), color='blue')
        except StopIteration:
            break


axs[0].set(xlabel='x', ylabel='y')
axs[1].set(xlabel='frames', ylabel='y')


plt.gca().invert_yaxis()

axs[0].xaxis.set_major_locator(tick.MultipleLocator(200))
axs[1].xaxis.set_major_locator(tick.MultipleLocator(200))

axs[0].yaxis.set_major_locator(tick.MultipleLocator(200))
axs[1].yaxis.set_major_locator(tick.MultipleLocator(200))

plt.legend()
#plt.savefig('../data/plot_KF.pdf')
#print("Plot saved")
plt.show()