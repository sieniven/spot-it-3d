import csv
import numpy as np
import matplotlib.pyplot as plt

def calculate_3d_position(tracks, frame_size, b, f1, f2, e1, e2):

    alpha
    z = calculate_z(tracks, b, )


    return np.array([x,y,z])



if __name__ == '__main__':
    fx = 1454.6
    cx = 960.9
    fy = 1450.3
    cy = 534.7

    B = 1.5

    # [frame_no, x_L, y_L, x_R, y_R]
    position_3D_temp = []

    # Left Camera
    with open('data_out_canon.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            position_3D_temp.append([frame_no, None, None, None, None])

            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                # if id == 42 or id == 43:  # Tello
                if id == 13:
                    position_3D_temp[frame_no][1] = int(track_data[1])
                    position_3D_temp[frame_no][2] = int(track_data[2])

    # Right Camera
    with open('data_out_sony.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for frame_no, row in enumerate(reader):
            tracks_in_frame = int((len(row) - 1) / 4)
            for i in range(tracks_in_frame):
                track_data = row[(i * 4) + 1:(i * 4) + 4 + 1]
                id = int(track_data[0])

                # if id == 0:  # Tello
                if id == 12:
                    x = int(track_data[1])
                    position_3D_temp[frame_no][3] = int(track_data[1])
                    position_3D_temp[frame_no][4] = int(track_data[2])

    frames = []
    epsilon = 7

    position_3D = []

    for frame in position_3D_temp:
        frame_no = frame[0]
        position_3D.append([frame_no, 0, 0, 0])
        if None not in frame:  # All values present
            x_L, y_L, x_R, y_R = frame[1:]

            alpha_L = np.arctan2(x_L - cx, fx) / np.pi * 180
            alpha_R = np.arctan2(x_R - cx, fx) / np.pi * 180

            gamma = epsilon + alpha_L - alpha_R

            # l = l_0 * np.sqrt(1 / (2 * (1 - np.cos(gamma / 180 * np.pi))))

            Z = B / (np.tan((alpha_L + epsilon / 2) * (np.pi / 180)) - np.tan((alpha_L + -epsilon / 2) * (np.pi / 180)))
            print(f"X from X_L: {Z * np.tan((alpha_L + epsilon / 2) * (np.pi / 180)) - B / 2},"
                  f"X from X_R:{Z * np.tan((alpha_R + -epsilon / 2) * (np.pi / 180)) + B / 2}")
            X = (Z * np.tan((alpha_L + epsilon / 2) * (np.pi / 180)) - B / 2
                 + Z * np.tan((alpha_R + -epsilon / 2) * (np.pi / 180)) + B / 2) / 2
            print(f"Y from L: {Z * -(y_L - cy) / fy},"
                  f"Y from R:{Z * -(y_R - cy) / fy}")
            Y = (Z * -(y_L - cy) / fy + Z * -(y_R - cy) / fy) / 2

            tilt = 10 * np.pi / 180
            R = np.array([[1, 0, 0],
                          [0, np.cos(tilt), np.sin(tilt)],
                          [0, -np.sin(tilt), np.cos(tilt)]])

            [X, Y, Z] = np.matmul(R, np.array([X, Y, Z]))

            Y += 1

            print(f"(X, Y, Z) = {X, Y, Z}")

            position_3D[frame_no][1:] = [X, Y, Z]

    # with open('depth_data.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     for row in out_data:
    #         writer.writerow(row)
    #
    with open('3D_positions.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in position_3D:
            if None not in row:
                writer.writerow(row)

    xs = []
    ys = []
    zs = []

    for position in position_3D:
        xs.append(position[1])
        ys = np.append(ys, position[2])
        zs.append(position[3])

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')

    ax_3d.set_title('Segment_0 Mavic mini')

    ax_3d.set_xlabel('X/m')
    ax_3d.set_ylabel('Z/m')
    ax_3d.set_zlabel('Y/m')

    ax_3d.scatter(xs, zs, ys)

    plt.show()
