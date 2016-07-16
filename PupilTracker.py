#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
Pupil tracking class.
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

from __future__ import division, print_function
import cv2
import numpy as np





class PupilTracker(object):
    """
    Image processing class.
    """
    def __init__(self, app):
        """
        Constructor.

        :param app: parent window
        """
        self.app = app

        # capture and output
        self.cap = None
        self.out = None

        # frames
        self.frame = None
        self.display_frame = None
        self.orig_frame = None

        # frame info
        self.frame_num = None
        self.num_frames = None
        self.vid_size = None
        self.display_scale = None
        self.scaled_size = None

        # pupil and reflection centers
        self.cx_pupil = None
        self.cy_pupil = None
        self.cx_refle = None
        self.cy_refle = None
        self.scaled_cx = None
        self.scaled_cy = None

        # param values were set for a 1080p image; this rescales params to whatever the current img size is
        self.param_scale = None

        # roi and processing params
        self.noise_kernel = None
        self.dx = None
        self.dy = None
        self.roi_pupil = None
        self.roi_refle = None
        self.roi_size = None
        self.scaled_roi_size = None
        self.can_pip = None
        self.tracking = True

        # data to track
        self.data = None
        self.angle = None
        self.angle_data = None

    def init_cap(self, video_file, window_width):
        """
        Creates capture object for video

        :param video_file: video path
        :param window_width: width of the window
        """
        if self.cap is not None:
            self.cap.release()

        # create capture and get info
        if video_file == 'webcam':
            self.cap = cv2.VideoCapture(0)
            self.num_frames = 200
        else:
            self.cap = cv2.VideoCapture(video_file)
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.vid_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.get_set_scaled_size(window_width)

        # init data holders
        self.data = np.empty((2, self.num_frames, 2))
        self.angle_data = np.empty(self.num_frames)
        self.clear_data()

        # init noise kernel
        self.noise_kernel = np.ones((3, 3), np.uint8)
        self.param_scale = self.vid_size[0] / 1920

        # load first frame
        self.load_first_frame()

    def release_cap(self):
        """
        Destroys cap object.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        else:
            raise IOError('VideoCapture not created. Nothing to release.')

    def next_frame(self):
        """
        Gets next frame.

        :return: next frame
        :raise EOFError: if at end of video file
        :raise IOError: if no video file loaded
        """
        if self.cap is not None:
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.display_frame = cv2.resize(self.frame,
                                                (self.scaled_size[0],
                                                 self.scaled_size[1]))
                self.orig_frame = self.display_frame.copy()
                self.frame_num += 1
            else:
                # at end; clear locations and return to first frame
                self.roi_pupil = None
                self.roi_refle = None
                self.roi_size = None
                self.load_first_frame()
                raise EOFError('Video end.')
        else:
            raise IOError('No video loaded.')

    def prev_frame(self):
        """
        Gets previous frame.

        :return: previous frame
        :raise EOFError: if at beginning of video file
        :raise IOError: if no video file loaded
        """
        if self.frame_num < 0:
            raise EOFError('Already at beginning')

        if self.cap is not None:
            self.frame_num -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES,
                         self.frame_num)
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                self.display_frame = cv2.resize(self.frame,
                                                (self.scaled_size[0],
                                                 self.scaled_size[1]))
                self.orig_frame = self.display_frame.copy()
        else:
            raise IOError('No video loaded.')

    def get_frame(self):
        """
        Gets the current display frame.

        :return: current display frame
        """
        if self.display_frame is not None:
            return self.display_frame

    def load_first_frame(self):
        """
        Loads the first frame.
        """
        # seek to first frame
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.frame_num = -1

            self.next_frame()

            # go back a frame so play will start at first frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_num = -1

            # clear data
            self.app.toggle_to_dump_data(set_to=False)
            self.clear_data()

        else:
            raise IOError('No video loaded.')

        # uncheck save and dump
        # self.app.toggle_to_save_video(set_to=False)

    def init_out(self, path):
        """
        Creates out object to write video to file.

        :param path: file save path
        """
        if self.out is None:
            self.out = cv2.VideoWriter(path,
                                       fourcc=cv2.VideoWriter_fourcc('m',
                                                                     'p',
                                                                     '4',
                                                                     'v'),
                                       fps=60,
                                       frameSize=(960, 540))
        else:
            raise IOError('VideoWriter already created. Release first.')

    def write_out(self):
        """
        Writes frames to file.
        """
        if self.out is not None:
            frame = cv2.cvtColor(self.display_frame, cv2.COLOR_RGB2BGR)
            self.out.write(frame)
        else:
            raise IOError('VideoWriter not created. Nothing with which to '
                          'write.')

    def release_out(self):
        """
        Destroys out object.
        """
        if self.out is not None:
            self.out.release()
            self.out = None
            print('Recording saved.')
        else:
            raise IOError('VideoWriter not created. Nothing to release.')

    def get_set_scaled_size(self, width):
        """
        Tracks the scale of the window relative to original frame size.

        :param width: window size
        :return: scaled size of video
        """
        if self.vid_size is not None:
            self.scaled_size = (width,
                                int(width * self.vid_size[1] / self.vid_size[0]))

            self.display_scale = self.vid_size[0] / width

        return self.scaled_size

    def on_size(self):
        """
        Resizes frame on size events.
        """
        if self.display_frame is not None:
            self.orig_frame = cv2.resize(self.frame,
                                         (self.scaled_size[0],
                                             self.scaled_size[1]))
            self.display_frame = self.orig_frame.copy()

        else:
            raise IOError('No video selected.')

    def clear_frame(self):
        """
        Clears frame of drawings.
        """
        if self.orig_frame is not None:
            self.display_frame = self.orig_frame.copy()
        else:
            raise IOError('Nothing here.')

    def clear_rois(self):
        """
        Clears out rois.
        """
        self.roi_pupil = None
        self.roi_refle = None

    def clear_data(self):
        """
        Fills the data arrays with NaN.
        """
        self.data.fill(np.NaN)
        self.angle_data.fill(np.NaN)

    def dump_data(self, path):
        """
        Dumps the data to file.

        :param path: file save path
        """
        with open(path, 'w') as f:
            np.savetxt(f, self.data[0],
                       delimiter=',',
                       fmt='%.0f',
                       header='pupil data\nx,y (pixels)',
                       footer='end pupil data\n')

            np.savetxt(f, self.data[1],
                       delimiter=',',
                       fmt='%.0f',
                       header='reflection data\nx,y (pixels)',
                       footer='end reflection data\n')

            np.savetxt(f, self.angle_data,
                       delimiter=',',
                       fmt='%f',
                       header='angle data\ndegrees',
                       footer='end angle data')

        print('data dumped')

    def process_image(self, img, roi=None):
        """
        Blurs, grayscales, and ROIs either entire frame or only certain
        region.

        :param img: frame being processed
        :param roi: region of interest being processed
        :return: grayscaled, blurred, ROIed frame
        """
        if roi is not None:
            # roi
            self.dx = roi[0][0]
            self.dy = roi[0][1]
            roi_image = img[roi[0][1]:roi[1][1],
                            roi[0][0]:roi[1][0]]
            # gaussian filter
            gauss = cv2.GaussianBlur(roi_image, (5, 5), 0)

        else:
            self.dx = 0
            self.dy = 0
            # gaussian filter
            gauss = cv2.GaussianBlur(img, (5, 5), 0)

        # make grayscale
        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)
        return gray

    def get_filtered(self, which):
        """
        Returns the filtered image blended with the original, to display how
        thresholding is happening to help the user better select a threshold.

        :param which: whether to return pupil or reflection image
        """
        grayed = self.process_image(self.frame)

        if which == 'pupil':
            _, threshed = cv2.threshold(grayed, self.app.pupil_thresh, 255,
                                        cv2.THRESH_BINARY)
            filtered = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE,
                                        self.noise_kernel, iterations=2)
        elif which == 'refle':
            _, threshed = cv2.threshold(grayed, self.app.refle_thresh, 255,
                                        cv2.THRESH_BINARY)
            filtered = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE,
                                        self.noise_kernel, iterations=1)
        else:
            raise AttributeError('Wrong parameter.')

        scaled_filtered = cv2.resize(filtered,
                                     (self.scaled_size[0],
                                      self.scaled_size[1]))

        color_scaled_filtered = cv2.cvtColor(scaled_filtered,
                                             cv2.COLOR_GRAY2BGR)

        if which == 'pupil':
            blended = cv2.addWeighted(color_scaled_filtered, 0.4,
                                      self.display_frame, 0.6,
                                      0)
        elif which == 'refle':
            blended = cv2.addWeighted(color_scaled_filtered, 0.7,
                                      self.display_frame, 0.3,
                                      0)
        else:
            raise AttributeError('Wrong parameter.')

        return blended

    def find_pupils(self, roi=None):
        """
        Searches for possible pupils in processed image

        :param roi: region of interest
        :return: list of possible pupil contours
        """
        # roi and gauss
        grayed = self.process_image(self.frame, roi)
        # threshold and remove noise
        _, thresh_pupil = cv2.threshold(grayed, self.app.pupil_thresh, 255,
                                        cv2.THRESH_BINARY)
        filtered_pupil = cv2.morphologyEx(thresh_pupil, cv2.MORPH_CLOSE,
                                          self.noise_kernel, iterations=2)

        # cv2.imshow('filtered_pupil', filtered_pupil.copy())
        # find contours
        _, contours_pupil, _ = cv2.findContours(filtered_pupil, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

        found_pupils = []
        # process contours
        if len(contours_pupil) != 0:
            for cnt in contours_pupil:

                # drop small and large
                area = cv2.contourArea(cnt)
                if area == 0:
                    # print('pupil area zero', self.frame_num)
                    continue

                if self.roi_size is None:
                    if not 2000 < area / self.param_scale < 120000:
                        # print('pupil too small/large', self.frame_num,
                        #       int(area / self.param_scale))
                        continue
                else:
                    if not self.param_scale * 2000 < area < self.roi_size**2:
                        continue

                # remove concavities, drop too few points
                hull = cv2.convexHull(cnt)
                if hull.shape[0] < 5:
                    # print('too few points', self.frame_num)
                    continue

                # drop too eccentric
                circumference = cv2.arcLength(hull, True)
                circularity = circumference ** 2 / (4*np.pi*area)
                if circularity >= 1.6:
                    # print('not circle', self.frame_num)
                    continue

                # rescale to full image
                hull[:, :, 0] += self.dx
                hull[:, :, 1] += self.dy

                found_pupils.append(hull)

        return found_pupils

    def draw_pupil(self, index=None, roi=None, verbose=True):
        """
        Draws the currently selected pupil to the frame.

        :param index: which pupil in the list of possible pupils to draw
        :param roi: region of interest
        :param verbose: if true, draws extra content to the frame (roi, etc)

        :raise AttributeError: if list of pupils is empty
        """

        # if no index passed, means we are tracking single pupil, so will be
        # first in list returned
        if index is None:
            index = 0

        # use already selected roi if only adjusting thresholds
        if roi == 'pupil':
            roi = self.roi_pupil

        if not self.tracking:
            self.roi_size = None

        # get list of pupil contours
        cnt_list = self.find_pupils(roi)

        if len(cnt_list) > 0:
            cnt = cnt_list[index]
        else:
            raise AttributeError('No pupils found.')

        # fit ellipse
        ellipse = cv2.fitEllipse(cnt)

        # centroid
        self.cx_pupil = int(np.rint(ellipse[0][0]))
        self.cy_pupil = int(np.rint(ellipse[0][1]))

        # angle of ellipse
        self.angle = ellipse[2]
        if self.angle > 90:
            self.angle -= 90
        else:
            self.angle += 90

        # scale for drawing
        scaled_cx = int(self.cx_pupil / self.display_scale)
        scaled_cy = int(self.cy_pupil / self.display_scale)
        self.scaled_cx = scaled_cx
        self.scaled_cy = scaled_cy

        # draw scaled
        cv2.line(self.display_frame,
                 (scaled_cx-2, scaled_cy),
                 (scaled_cx+2, scaled_cy),
                 (255, 255, 255), 1)
        cv2.line(self.display_frame,
                 (scaled_cx, scaled_cy-2),
                 (scaled_cx, scaled_cy+2),
                 (255, 255, 255), 1)

        scaled_cnt = np.rint(cnt / self.display_scale)
        scaled_cnt = scaled_cnt.astype(int)
        scaled_ellipse = cv2.fitEllipse(scaled_cnt)
        cv2.ellipse(self.display_frame, scaled_ellipse, (0, 255, 100), 1)

        if self.roi_size is None:
            self.roi_size = int(np.rint(max(ellipse[1][0], ellipse[1][1]) *
                                        1.75))
        self.scaled_roi_size = int(self.roi_size / self.display_scale)

        # extra drawings
        if verbose:
            cv2.drawContours(self.display_frame, scaled_cnt, -1, (255, 255,
                                                                  255), 2)
            cv2.rectangle(self.display_frame,
                          (scaled_cx - self.scaled_roi_size, scaled_cy - self.scaled_roi_size),
                          (scaled_cx + self.scaled_roi_size, scaled_cy + self.scaled_roi_size),
                          (255, 255, 255))
            # box = cv2.boxPoints(ellipse)
            # box = np.int0(box)
            # cv2.drawContours(self.display_frame, [box], 0,(0,0,255),1)

        # correct out of bounds roi
        roi_lu_x = self.cx_pupil - self.roi_size
        roi_lu_y = self.cy_pupil - self.roi_size
        roi_rl_x = self.cx_pupil + self.roi_size
        roi_rl_y = self.cy_pupil + self.roi_size
        if roi_lu_x < 0:
            roi_lu_x = 0
        if roi_lu_y < 0:
            roi_lu_y = 0

        self.roi_pupil = [(roi_lu_x, roi_lu_y),
                          (roi_rl_x, roi_rl_y)]

        self.tracking = False

    def track_pupil(self, verbose=True):
        """
        Makes call to draw pupil with proper roi and handles errors.

        :param verbose: whether or not to draw extra
        """
        if self.roi_pupil is not None:
            try:
                self.draw_pupil(roi='pupil', verbose=verbose)
                try:
                    self.data[0][self.frame_num] = [self.cx_pupil, self.cy_pupil]
                    self.angle_data[self.frame_num] = self.angle
                except IndexError:
                    self.frame_num = 0
                    self.track_pupil(verbose)
                self.can_pip = True
                self.tracking = True
                # TODO: make tracking tracker
                # bc when loses roi then resets shape because can't pip...

            # except IndexError as e:
            #     # print(e)
            #     pass

            # no pupils found
            except AttributeError:
                # print(e)
                self.can_pip = False
        else:
            pass

    def find_refle(self, roi=None):
        """
        Searches for possible reflections in processed image.

        :param roi: region of interest
        :return: list of possible reflection contours
        """
        # roi and gauss
        grayed = self.process_image(self.frame, roi)
        # threshold and remove noise
        _, thresh_refle = cv2.threshold(grayed, self.app.refle_thresh, 255,
                                        cv2.THRESH_BINARY)
        filtered_refle = cv2.morphologyEx(thresh_refle, cv2.MORPH_CLOSE,
                                          self.noise_kernel, iterations=1)

        # cv2.imshow('filtered_refle', filtered_refle.copy())
        # find contours
        _, contours_refle, _ = cv2.findContours(filtered_refle, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)

        found_reflections = []
        # process contours
        if len(contours_refle) != 0:
            for cnt in contours_refle:

                # drop small and large
                area = cv2.contourArea(cnt)
                if area == 0:
                    # print('refle area zero', self.frame_num)
                    continue

                if not 80 < area / self.param_scale < 8000:
                    # print('refle too small/large', self.frame_num, int(area / self.param_scale))
                    continue

                # rescale to full image
                cnt[:, :, 0] += self.dx
                cnt[:, :, 1] += self.dy

                # test squareness
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1][0], rect[1][1]
                squareness = h / w
                if not 0.5 < squareness < 2:
                    # print('refle not square', self.frame_num)
                    continue

                # see if center in roi
                if roi is not None:
                    # rect center
                    cx = int(rect[0][0])
                    cy = int(rect[0][1])
                    if not roi[0][0] < cx < roi[1][0] \
                            or not \
                            roi[0][1] < cy < roi[1][1]:
                        # print('refle not in roi', self.frame_num)
                        continue

                found_reflections.append(cnt)

        return found_reflections

    def draw_refle(self, index=None, roi=None, verbose=True):
        """
        Draws the currently selected reflection to the frame.

        :param index: which pupil in the list of possible reflections to draw
        :param roi: region of interest
        :param verbose: if true, draws extra content to the frame (roi, etc)
        :raise AttributeError: if list of reflections is empty
        """
        # if no index passed, means we are tracking single reflection
        if index is None:
            index = 0

        # use already selected roi if found pupil or only adjusting thresholds
        if roi == 'pupil':
            roi = self.roi_pupil
        elif roi == 'refle':
            roi = self.roi_refle

        # get list of reflection contours
        cnt_list = self.find_refle(roi)

        if len(cnt_list) > 0:
            cnt = cnt_list[index]
        else:
            raise AttributeError('No reflections found.')

        # fit rectangle to contour
        rect = cv2.minAreaRect(cnt)
        # rect center
        self.cx_refle = int(rect[0][0])
        self.cy_refle = int(rect[0][1])

        # reset roi
        # TODO: don't let ROI get too small
        roi_size = int(np.rint(max(rect[1][0], rect[1][1])) * 1.25)
        scaled_roi_size = int(roi_size / self.display_scale)
        self.roi_refle = [(self.cx_refle - roi_size, self.cy_refle - roi_size),
                          (self.cx_refle + roi_size, self.cy_refle + roi_size)]

        # scale for drawing
        scaled_cx = int(self.cx_refle / self.display_scale)
        scaled_cy = int(self.cy_refle / self.display_scale)

        # draw
        cv2.line(self.display_frame,
                 (scaled_cx-2, scaled_cy),
                 (scaled_cx+2, scaled_cy),
                 (0, 0, 0), 1)
        cv2.line(self.display_frame,
                 (scaled_cx, scaled_cy-2),
                 (scaled_cx, scaled_cy+2),
                 (0, 0, 0), 1)

        scaled_cnt = np.rint(cnt / self.display_scale)
        scaled_cnt = scaled_cnt.astype(int)
        scaled_rect = cv2.minAreaRect(scaled_cnt)
        box = cv2.boxPoints(scaled_rect)
        box = np.int0(box)
        cv2.drawContours(self.display_frame, [box], 0, (0, 255, 100), 1)

        # draw extra
        if verbose:
            cv2.rectangle(self.display_frame,
                          (scaled_cx - scaled_roi_size, scaled_cy - scaled_roi_size),
                          (scaled_cx + scaled_roi_size, scaled_cy + scaled_roi_size),
                          (255, 255, 255))
            cv2.drawContours(self.display_frame, scaled_cnt, -1, (0, 0, 255), 2)

    def track_refle(self, verbose=True):
        """
        Makes call to draw reflection with proper roi and handles errors.

        :param verbose: whether or not to draw extra
        """
        if self.roi_refle is not None:
            try:
                self.draw_refle(roi='refle', verbose=verbose)
                self.data[1][self.frame_num] = [self.cx_refle, self.cy_refle]

            # except IndexError as e:
            #     # print(e)
            #     pass

            # no reflections found
            except AttributeError:
                # print(e)
                pass
        else:
            pass

    def pip(self):
        """
        Creates picture in picture of pupil ROI
        """
        if self.roi_pupil is not None and self.can_pip:
            # get roi
            roi_size = self.scaled_roi_size

            y1, y2 = self.scaled_cy-roi_size+1, self.scaled_cy+roi_size
            x1, x2 = self.scaled_cx-roi_size+1, self.scaled_cx+roi_size

            coords = [x1, x2, y1, y2]

            for ind, element in enumerate(coords):
                if element < 0:
                    coords[ind] = 0

            roi_image = self.display_frame[
                        coords[2]:coords[3],
                        coords[0]:coords[1]]

            # replace in frame
            self.display_frame[0:roi_image.shape[0],
                               self.display_frame.shape[1]-roi_image.shape[1]:
                               self.display_frame.shape[1]] = roi_image
