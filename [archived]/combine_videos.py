import cv2
import numpy as np

from object_tracking_rt import imshow_resized

# video_1 = cv2.VideoCapture('videos/00014_x264_x264.mp4')
# video_2 = cv2.VideoCapture('videos/IMG_2059_HEVC_Segment_0_x264.mp4')
video_1 = cv2.VideoCapture('Binocular Camera/segment_1/video_plot_sony_x264.mp4')
video_2 = cv2.VideoCapture('Binocular Camera/segment_1/video_plot_phone_x264.mp4')

# # Top Bottom
# out_width = max(int(video_1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_2.get(cv2.CAP_PROP_FRAME_WIDTH)))
# out_height = int(video_1.get(cv2.CAP_PROP_FRAME_HEIGHT)) + int(video_2.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Left Right
out_width = int(video_1.get(cv2.CAP_PROP_FRAME_WIDTH)) + int(video_2.get(cv2.CAP_PROP_FRAME_WIDTH))
out_height = max(int(video_1.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video_2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Note fps
output = cv2.VideoWriter('out_stabilized_combined.mp4', cv2.VideoWriter_fourcc(*'h264'), 25, (out_width, out_height))

while video_1.isOpened() and video_2.isOpened():
    ret_1, frame_1 = video_1.read()
    ret_2, frame_2 = video_2.read()

    if ret_1 and ret_2:
        frame_out = np.zeros((out_height, out_width, 3), dtype=np.uint8)

        # # Top Bottom
        # frame_out[0:frame_1.shape[0], 0:frame_1.shape[1]] = frame_1
        # frame_out[frame_1.shape[0]:frame_1.shape[0] + frame_2.shape[0], 0:frame_2.shape[1]] = frame_2
        # Left Right
        frame_out[0:frame_1.shape[0], 0:frame_1.shape[1]] = frame_1
        frame_out[0:frame_2.shape[0], frame_1.shape[1]:frame_1.shape[1] + frame_2.shape[1]] = frame_2

        output.write(frame_out)
        imshow_resized("out", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video_1.release()
video_2.release()
output.release()
