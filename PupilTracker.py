import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy import spatial
import pandas as pd

TO_OUT = False
VERBOSE = False


video_file = os.path.abspath(r'C:\Users\Alex\PycharmProjects\EyeTracker\00091'
                             r'_short.mov')
# video_file = os.path.abspath(r'C:\Users\Alex\PycharmProjects\EyeTracker\00093'
#                              r'.mov')

cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

data = np.empty((2, num_frames, 2))
data.fill(np.NaN)


# to save videos
if TO_OUT:
    out_file = os.path.abspath(r'C:\Users\Alex\PycharmProjects\EyeTracker\00093'
                               r'_short_tracked.mov')
    out = cv2.VideoWriter(out_file,
                          fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                          fps=24,
                          frameSize=(960, 540))

# noise kernel
kernel = np.ones((3, 3), np.uint8)
#
# roi_pupil = [(513-100, 248-100), (513+100, 248+100)] # 84 85
# roi_pupil = [(486-100, 260-100), (486+100, 260+100)] # 89
# roi_pupil = [(471-100, 251-100), (471+100, 251+100)] # 90
roi_pupil = [(450-100, 260-100), (450+100, 260+100)] # 91
# roi_pupil = [(397-100, 333-100), (397+100, 333+100)] # 92
# roi_pupil = [(79, 109), (379, 409)] # 93

# roi_refl  = [(496-25, 239-25), (496+25, 239+25)] # 84
# roi_refl  = [(530, 206), (580, 281)] # 85
# roi_refl  = [(462-15, 247-15), (462+15, 247+15)] # 89
# roi_refl  = [(478-15, 240-15), (478+15, 240+15)] # 90
roi_refl  = [(411, 224), (461, 274)] # 91
# roi_refl  = [(382-15, 327-15), (382+15, 327+15)] # 92
# roi_refl  = [(189, 215), (239, 255)] # 93

# while cap.isOpened():
# for i in range(1):
for i in range(num_frames):
    # seek to next frame
    _, frame = cap.read()
    # rescale
    image = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # roi
    dx = roi_pupil[0][0]
    dy = roi_pupil[0][1]
    roi = image[roi_pupil[0][1]:roi_pupil[1][1], roi_pupil[0][0]:roi_pupil[1][0]]
    # gauss
    gauss = cv2.GaussianBlur(roi, (5,5), 0)
    # gauss = cv2.GaussianBlur(image, (5,5), 0)
    # make grayscale
    gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
    # threshold
    _, thresh_pupil = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    _, thresh_refl = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # remove noise
    filt_pupil = cv2.morphologyEx(thresh_pupil, cv2.MORPH_CLOSE, kernel,
                                  iterations=4)
    filt_refl = cv2.morphologyEx(thresh_refl, cv2.MORPH_CLOSE, kernel,
                                 iterations=2)
    # find contours
    cont_pupil = filt_pupil.copy()
    cont_refl = filt_refl.copy()
    _, contours_pupil, _ = cv2.findContours(cont_pupil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours_refl, _ = cv2.findContours(cont_refl, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    # process pupil contours
    if len(contours_pupil) != 0:
        for cnt in contours_pupil:
            # drop small and large
            area = cv2.contourArea(cnt)
            if area == 0:
                continue
            if not 1000 < area < 5000:
                continue

            # drop concave points
            hull = cv2.convexHull(cnt)
            if hull.shape[0] < 5:
                continue

            # drop obvious non circular
            circumference = cv2.arcLength(hull, True)
            circularity = circumference ** 2 / (4*np.pi*area)
            if circularity >= 1.6:
                continue

            # print area, circularity

            hull[:, :, 0] += dx
            hull[:, :, 1] += dy

            # fit ellipse
            ellipse = cv2.fitEllipse(hull)

            # centroid
            cx = int(ellipse[0][0])
            cy = int(ellipse[0][1])

            # see if centroid in roi
            # if not roi_pupil[0][0] < cx < roi_pupil[1][0] or not roi_pupil[0][1] < cy < roi_pupil[1][1]:
            #     continue
            # print 'pupil:', (cx, cy)

            # add to data
            data[0][i] = [cx, cy]

            # draw ellipse, contours, centroid
            if VERBOSE:
                cv2.drawContours(image, hull, -1, (0, 0, 255), 2)
            cv2.ellipse(image, ellipse, (0, 255, 100), 1)
            cv2.circle(image, (cx, cy), 2, (255,255,255))

            # reset roi
            if cx < 100:
                cx = 100
            if cy < 100:
                cy = 100
            roi_pupil = [(cx - 100, cy -100), (cx + 100, cy + 100)]

            # draw roi
            if VERBOSE:
                cv2.rectangle(image, (cx-100, cy-100), (cx+100, cy+100),
                              (255, 255, 255))
    else:
        print 'no pupil found (frame {})'.format(i)
        data[0][i] = [np.NaN, np.NaN]

    # process reflection contours
    if len(contours_refl) != 0:
        for cnt in contours_refl:
            # drop small and large
            area = cv2.contourArea(cnt)
            # print area
            if area == 0:
                continue
            if not 25 < area < 400:
                continue

            cnt[:, :, 0] += dx
            cnt[:, :, 1] += dy

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # rect center
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            # see if center in roi
            if not roi_refl[0][0] < cx < roi_refl[1][0] or not roi_refl[0][1] <\
                    cy < roi_refl[1][1]:
                continue
            # print 'refl:', (cx, cy)

            # add to data
            data[1][i] = [cx, cy]

            # reset roi
            roi_refl = [(cx - 15, cy - 15), (cx + 15, cy + 15)]

            # draw
            if VERBOSE:
                cv2.rectangle(image, (cx-15, cy-15), (cx+15, cy+15), (255, 255, 255))
                cv2.drawContours(image, cnt, -1, (0, 0, 255), 2)
            cv2.drawContours(image, [box], 0, (0, 255, 100), 1)
            cv2.circle(image, (cx, cy), 2, (100, 100, 100))
    else:
        print 'no reflection found (frame {})'.format(i)
        data[0][i] = [np.NaN, np.NaN]


    # show image
    # cv2.imshow('roi', roi)
    # cv2.imshow('gauss', gauss)
    # cv2.imshow('gray', gray)
    # cv2.imshow('thresh_pupil', thresh_pupil)
    # cv2.imshow('thresh_refl', thresh_refl)
    # cv2.imshow('filt_pupil', filt_pupil)
    # cv2.imshow('filt_refl', filt_refl)
    if TO_OUT:
        out.write(image)

    cv2.imshow('image', image)

    # cv2.waitKey(1)
    # cv2.waitKey(0)
    cv2.waitKey(int(1000/24))

if TO_OUT:
    out.release()
cap.release()
cv2.destroyAllWindows()

pupil_data = data[0]
refl_data = data[1]

x_norm = pupil_data[0][0] - refl_data[0][0]
y_norm = pupil_data[0][1] - refl_data[0][1]

pupil_x = pupil_data[:, 0] - refl_data[:, 0] - x_norm
pupil_y = pupil_data[:, 1] - refl_data[:, 1] - y_norm

def moving_average(ar, n=3):
    # use pandas so can handle NaN
    df = pd.Series(ar)
    return pd.rolling_mean(df, n, min_periods=2).values

def plot_data(pupil_x, pupil_y):
    running_avg_x = moving_average(pupil_x, n=24)
    running_avg_y = moving_average(pupil_y, n=24)

    fig, ax = plt.subplots(2, 2)
    fig.suptitle('movements data')
    ax[0][0].set_title('raw positions')
    ax[0][1].set_title('running average (50 ms)')

    x_center = 0
    y_center = 0

    ax[0][0].plot(pupil_x, 'r-')
    ax[0][0].set_xlabel('frame number')
    ax[0][0].set_ylabel('x axis position', color='r')
    ax[0][0].set_ylim([x_center - 30, x_center + 30])

    ax[1][0].plot(pupil_y, 'b-')
    ax[1][0].set_xlabel('frame number')
    ax[1][0].set_ylabel('y axis position', color='b')
    ax[1][0].set_ylim([y_center - 30, y_center + 30])

    ax[0][1].plot(running_avg_x, 'r-')
    ax[0][1].set_xlabel('frame number')
    # ax[0][1].set_ylabel('x axis position', color='r')
    ax[0][1].set_ylim([x_center - 30, x_center + 30])

    ax[1][1].plot(running_avg_y, 'b-')
    ax[1][1].set_xlabel('frame number')
    # ax[1][1].set_ylabel('y axis position', color='b')
    ax[1][1].set_ylim([y_center - 30, y_center + 30])

    plt.show()

plot_data(pupil_x, pupil_y)

if __name__ == '__main__':
    pass