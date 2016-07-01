#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
Pupil tracking software.
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

from __future__ import division, print_function
import wx
import wxmplot # wx matplotlib library
import cv2
import numpy as np
from os import path
from sys import platform
# from psychopy.core import MonotonicClock  # for getting display fps


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

        # roi and processing params
        self.noise_kernel = None
        self.param_scale = None
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
        self.app.toggle_to_save_video(set_to=False)

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
                    if not 2000 < area / self.param_scale < 100000:
                        # print('pupil too small/large', self.frame_num,
                        #       int(area / self.param_scale))
                        continue
                else:
                    if not self.param_scale * 2000 < area < self.roi_size**2:
                        continue

                # drop too few points
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
                self.data[0][self.frame_num] = [self.cx_pupil, self.cy_pupil]
                self.angle_data[self.frame_num] = self.angle
                self.can_pip = True
                self.tracking = True
                # TODO: make tracking tracker
                # bc when loses roi then resets shape because can't pip...

            # except IndexError as e:
            #     # print(e)
            #     pass

            # no pupils found
            except AttributeError as e:
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
                    print('refle area zero', self.frame_num)
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
            except AttributeError as e:
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
            roi_image = self.display_frame[
                        self.scaled_cy-roi_size+1:self.scaled_cy+roi_size,
                        self.scaled_cx-roi_size+1:self.scaled_cx+roi_size]

            # replace in frame
            self.display_frame[0:roi_image.shape[0],
                               self.display_frame.shape[1]-roi_image.shape[1]:
                               self.display_frame.shape[1]] = roi_image


class ImagePanel(wx.Panel):
    """
    Class for panel holding the images of the rabbit eyes.

    :param parent: parent window panel (MyFrame in this case)
    """
    def __init__(self, parent):
        """
        Constructor.
        """
        # super instantiation
        super(ImagePanel, self).__init__(parent, size=(960, 540))

        # instance attributes
        self.app = parent
        self.image_bmp = None
        self.orig_image = None
        # self.t = None

        self.SetDoubleBuffered(True)
        self.fps = 1000  # why not
        self.fps_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.draw, self.fps_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def start_timer(self):
        """
        Starts timer for draw timing.
        """
        self.fps_timer.Start(1000 // self.fps)
        # self.t = MonotonicClock()

    def stop_timer(self):
        """
        Stops timer.
        """
        self.fps_timer.Stop()
        # try:
        #     t = self.t.getTime()
        #     f = self.app.tracker.num_frames
        #     print(t, f),
        #     print(f/t)
        # except:
        #     pass

    def load_image(self, img):
        """
        Creates buffer loader and loads first image

        :param img: image to be loaded (first frame of video)
        """
        h = img.shape[0]
        w = img.shape[1]
        self.image_bmp = wx.BitmapFromBuffer(w, h, img)
        self.Refresh()  # causes paint

    def draw(self, evt=None, img=None):
        """
        Draws frame passed from tracking class.

        :param evt: Required event parameter
        """
        if self.app.playing:
            try:
                self.app.next_frame()
                self.app.SetStatusText(str(self.app.tracker.frame_num+1) + '/'
                                       + str(self.app.tracker.num_frames), 1)
            except EOFError as e:
                print(e)
                self.app.toggle_playing(set_to=False)
                self.stop_timer()
                self.app.clear_rois()
                self.app.clear_indices()
                self.load_image(self.app.get_frame())
                self.app.SetStatusText(str(self.app.tracker.frame_num+1) + '/'
                                       + str(self.app.tracker.num_frames), 1)
                return
            except IOError as e:
                print(e)
                self.app.toggle_playing(set_to=False)
                self.stop_timer()
                return

            self.app.track_pupil()
            self.app.track_refle()
            if self.app.to_pip:
                self.app.pip()
            try:
                self.app.write_out()
            except IOError:
                pass

            if self.app.to_plot and self.app.tracker.frame_num % 3 == 0:
                self.app.update_plot()

        if img is None:
            if self.image_bmp is not None:
                self.image_bmp.CopyFromBuffer(self.app.get_frame())
            else:
                raise AttributeError('Nothing here.')
        else:
            self.image_bmp.CopyFromBuffer(img)
        self.Refresh()  # causes paint

        # TODO: fix setstatus
        # self.app.SetStatusText(str(self.app.tracker.frame_num), 1)
        if evt is not None:
            evt.Skip()

    def on_paint(self, evt):
        """
        Pulls bitmap from buffer and draws to panel.

        :param evt: paint event, required param
        """
        if self.image_bmp is not None:
            dc = wx.BufferedPaintDC(self)
            dc.Clear()
            dc.DrawBitmap(self.image_bmp, 0, 0)
        evt.Skip()

    def on_size(self, size, img):
        """
        On resize refreshes the bitmap buffer and redraws.

        :param size: size of the panel
        :param img: new resized image to draw
        """
        h = size[1]
        w = size[0]
        self.image_bmp = wx.BitmapFromBuffer(w, h, img)
        self.Refresh()  # causes paint


# class LabeledTextCtrl(wx.BoxSizer):
#     """
#     Class which returns a horizontal boxsizer object with a wx.StaticText as
#     the label and wx.TextCtrl as the box to fill out. Also returns the ctrl.
#     """
#     def __init__(self, label, default, parent):
#         """Constructor.
#
#         :param label: text of the label
#         :param default: default value to populate ctrl
#         """
#         # super instantiation
#         super(LabeledTextCtrl, self).__init__(wx.HORIZONTAL)
#         self.label = str(label)
#         self.default = str(default)
#         self.parent = parent
#
#     def make(self):
#         label = wx.StaticText(self.parent,
#                               label=self.label + ':')
#         ctrl = wx.TextCtrl(self.parent,
#                            value=self.default)
#
#         self.Add(label,
#                  border=5,
#                  flag=wx.RIGHT)
#         self.Add(ctrl)
#
#         return ctrl, self


class ToolsPanel(wx.Panel):
    """
    Class for panel with buttons.
    """
    def __init__(self, parent):
        """
        Constructor
        """
        # super instantiation
        super(ToolsPanel, self).__init__(parent)

        # instance attributes
        self.app = parent
        self.pupil_index = None
        self.refle_index = None

        # buttons
        self.find_pupil_button = wx.Button(self, label='Find pupil')
        self.find_refle_button = wx.Button(self, label='Find refle')
        self.clear_button = wx.Button(self, label='Clear')
        self.load_button = wx.Button(self, label='Load')
        self.play_button = wx.Button(self, label='Play')
        self.pause_button = wx.Button(self, label='Pause')
        self.stop_button = wx.Button(self, label='Stop')
        self.default_button = wx.Button(self, label='Default')

        # toggles
        self.plot_toggle = wx.CheckBox(self, label='Plot')
        self.plot_toggle.SetValue(False)
        self.pip_toggle = wx.CheckBox(self, label='PIP')
        self.pip_toggle.SetValue(False)
        self.verbose_toggle = wx.CheckBox(self, label='Verbose')
        self.verbose_toggle.SetValue(False)
        self.save_video_toggle = wx.CheckBox(self, label='Save video')
        self.save_video_toggle.SetValue(False)
        self.dump_data_toggle = wx.CheckBox(self, label='Dump data')
        self.dump_data_toggle.SetValue(False)

        # threshold sliders
        self.pupil_slider = wx.Slider(self,
                                      value=50,
                                      minValue=0,
                                      maxValue=150,
                                      style=wx.SL_VERTICAL | wx.SL_LABELS)
        self.refle_slider = wx.Slider(self,
                                      value=190,
                                      minValue=155,
                                      maxValue=255,
                                      style=wx.SL_VERTICAL | wx.SL_LABELS |
                                            wx.SL_INVERSE)

        # sizer for sliders
        slider_sizer = wx.BoxSizer(wx.HORIZONTAL)
        slider_sizer.Add(self.pupil_slider,
                         border=5,
                         flag=wx.EXPAND | wx.RIGHT,
                         proportion=1)
        slider_sizer.Add(self.refle_slider,
                         # border=5,
                         flag=wx.EXPAND,  # | wx.RIGHT,
                         proportion=1)

        # button sizer
        button_sizer = wx.BoxSizer(wx.VERTICAL)

        # add to sizer
        button_sizer.Add(self.find_pupil_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.find_refle_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.clear_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.load_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.play_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.pause_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.stop_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.plot_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.pip_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.verbose_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.save_video_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.dump_data_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        # button_sizer.Add(self.pupil_slider,
        #                  flag=wx.LEFT | wx.RIGHT | wx.TOP,
        #                  border=5)
        # button_sizer.Add(self.refle_slider,
        #                  flag=wx.LEFT | wx.RIGHT | wx.TOP,
        #                  border=5)
        button_sizer.Add(slider_sizer,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND,
                         border=5,
                         proportion=1)
        button_sizer.Add(self.default_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)

        # event binders
        self.Bind(wx.EVT_BUTTON,
                  self.on_find_pupil_button,
                  self.find_pupil_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_find_refle_button,
                  self.find_refle_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_clear_button,
                  self.clear_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_load_button,
                  self.load_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_play_button,
                  self.play_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_pause_button,
                  self.pause_button)
        self.Bind(wx.EVT_BUTTON,
                  self.on_stop_button,
                  self.stop_button)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_plot_toggle,
                  self.plot_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_pip_toggle,
                  self.pip_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_verbose_toggle,
                  self.verbose_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_save_video_toggle,
                  self.save_video_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_dump_data_toggle,
                  self.dump_data_toggle)
        self.Bind(wx.EVT_SCROLL_THUMBTRACK,
                  self.on_pupil_slider_thumbtrack,
                  self.pupil_slider)
        self.Bind(wx.EVT_SCROLL_THUMBRELEASE,
                  self.on_slider_release,
                  self.pupil_slider)
        self.Bind(wx.EVT_SCROLL_CHANGED,
                  self.on_pupil_slider_changed,
                  self.pupil_slider)
        self.Bind(wx.EVT_SCROLL_THUMBTRACK,
                  self.on_refle_slider_thumbtrack,
                  self.refle_slider)
        self.Bind(wx.EVT_SCROLL_THUMBRELEASE,
                  self.on_slider_release,
                  self.refle_slider)
        self.Bind(wx.EVT_BUTTON,
                  self.on_default_button,
                  self.default_button)

        # set sizer
        self.SetSizer(button_sizer)

    def clear_indices(self):
        """
        Resets pupil and reflection indices to None.
        """
        self.pupil_index = None
        self.refle_index = None

    def on_find_pupil_button(self, evt):
        """
        Cycles through found pupils

        :param evt: required event parameter
        """
        try:
            if self.pupil_index is None:
                self.pupil_index = 0

            # clear to draw
            self.app.clear(draw=False, keep_roi=True)

            # redraw reflection if present
            if self.refle_index is not None:
                self.app.redraw_refle()

            # draw pupil
            self.app.draw_pupil(self.pupil_index)
            self.pupil_index += 1

            self.app.draw()

        # end of pupil list, so go back to beginning
        except IndexError as e:
            self.pupil_index = 0
            self.on_find_pupil_button(evt)

        # no pupils found, so draw blank (or with reflection if found)
        except AttributeError as e:
            self.pupil_index = None
            self.app.draw()
            print(e)

        except IOError as e:
            self.pupil_index = None
            print(e)

    def on_find_refle_button(self, evt):
        """
        Cycles through found reflections

        :param evt: required event parameter
        """
        try:
            if self.refle_index is None:
                self.refle_index = 0

            # clear to draw
            self.app.clear(draw=False, keep_roi=True)

            # redraw pupil if present
            if self.pupil_index is not None:
                self.app.redraw_pupil()

                # draw reflection, only search in pupil if present
                self.app.draw_refle(self.refle_index, roi='pupil')

            else:
                self.app.draw_refle(self.refle_index)


            self.refle_index += 1

            self.app.draw()

        # end of refle list, so go back to beginning
        except IndexError as e:
            self.refle_index = 0
            self.on_find_refle_button(evt)

        # no reflections found, so draw blank (or with pupil if found)
        except AttributeError as e:
            self.refle_index = None
            self.app.draw()
            print(e)

        except IOError as e:
            self.pupil_index = None
            print(e)

    def on_clear_button(self, evt):
        """
        Clears drawings.

        :param evt: required event parameter
        """
        self.clear_indices()

        try:
            self.app.clear(draw=True)
        except IOError as e:
            print(e)

    def on_load_button(self, evt):
        """
        Loads video.

        :param evt: required event parameter
        """
        self.clear_indices()

        self.app.load_dialog()

    def on_play_button(self, evt):
        """
        Starts the timer.

        :param evt: required event parameter
        """
        self.app.play()

    def on_pause_button(self, evt):
        """
        Stops the timer.

        :param evt: required event parameter
        """
        self.app.pause()

    def on_stop_button(self, evt):
        """
        Stops the timer.

        :param evt: required event parameter
        """
        self.clear_indices()

        self.app.stop()

    def on_plot_toggle(self, evt):
        """
        Toggles whether or not to plot.

        :param evt: required event parameter
        """
        self.app.toggle_to_plot()

    def on_pip_toggle(self, evt):
        """
        Toggles PiP (picture in picture)

        :param evt: required event parameter
        """
        self.app.toggle_to_pip()

    def on_verbose_toggle(self, evt):
        """
        Toggles verbosity.

        :param evt: required event parameter
        """
        self.app.toggle_verbose(self.pupil_index, self.refle_index)

    def on_save_video_toggle(self, evt):
        """
        Toggles video saving.

        :param evt: required event parameter
        """
        self.app.toggle_to_save_video()

    def on_dump_data_toggle(self, evt):
        """
        Toggles dumping data.

        :param evt: required event parameter
        """
        self.app.toggle_to_dump_data()

    def on_pupil_slider_thumbtrack(self, evt):
        """
        Dynamically adjusts threshold for pupils.

        :param evt: required event parameter
        """
        self.app.pause()
        self.app.pupil_thresh = int(evt.GetInt())

        try:
            self.app.clear(draw=False, keep_roi=True)
        except IOError as e:
            return

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError as e:
                pass

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError as e:
                pass

        self.app.draw(self.app.tracker.get_filtered('pupil'))

    def on_refle_slider_thumbtrack(self, evt):
        """
        Dynamically adjusts threshold for reflections.

        :param evt: required event parameter
        """
        self.app.pause()
        self.app.refle_thresh = int(evt.GetInt())

        try:
            self.app.clear(draw=False, keep_roi=True)
        except IOError as e:
            return

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError as e:
                pass

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError as e:
                pass

        self.app.draw(self.app.tracker.get_filtered('refle'))

    def on_slider_release(self, evt):
        """
        Returns the image to the display frame after done scrolling with the
        slider.

        :param evt: required event parameter
        """
        try:
            self.app.draw()
        except AttributeError as e:
            # print(e)
            pass

    def on_pupil_slider_changed(self, evt):
        """
        For when the slider is changed without being thumbtracked.

        :param evt:
        :return:
        """
        self.app.pause()
        self.app.pupil_thresh = int(evt.GetInt())

        try:
            self.app.clear(draw=False, keep_roi=True)
        except IOError as e:
            return

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError as e:
                pass

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError as e:
                pass

        self.app.draw()

    def on_default_button(self, evt):
        """
        Returns sliders to their default position.

        :param evt: required event parameter
        """
        self.pupil_slider.SetValue(50)
        # generate event for slider
        evt = wx.CommandEvent(wx.EVT_SCROLL_THUMBTRACK.typeId,
                              self.pupil_slider.Id)
        evt.SetInt(50)
        self.pupil_slider.GetParent().GetEventHandler().ProcessEvent(evt)

        self.refle_slider.SetValue(190)
        # generate event for slider
        evt = wx.CommandEvent(wx.EVT_SCROLL_THUMBTRACK.typeId,
                              self.refle_slider.Id)
        evt.SetInt(190)
        self.refle_slider.GetParent().GetEventHandler().ProcessEvent(evt)

        self.on_slider_release(evt)


class PlotPanel(wxmplot.PlotPanel):
    """
    Class for panel with dynamic plots of coordinates and angle.
    """
    def __init__(self, parent):
        """
        Constructor.

        :param parent: parent app
        """
        self.plot_height = 300
        kwargs = dict(fontsize=5,
                      size=(960, self.plot_height),
                      # axisbg='black'
                      )

        super(PlotPanel, self).__init__(parent, **kwargs)

        self.frames = None
        self.data = None
        self.angle_data = None
        self.background = None
        self.x_delta = None
        self.y_delta = None
        self.x_apos = None
        self.y_apos = None
        self.pup_an = None
        self.pupil_x = None
        self.pupil_y = None
        self.x_norm = None
        self.y_norm = None

    def init_plot(self, data, angle_data):
        self.frames = np.arange(data.shape[1])
        self.data = data
        self.angle_data = angle_data

        self.calc()

        guess_dif = 200
        self.x_delta = self.plot(self.frames, self.pupil_x,
                                 ymin=0-guess_dif,
                                 ymax=0+guess_dif,
                                 color='red',
                                 label='x delta',
                                 markersize=5,
                                 labelfontsize=5,
                                 show_legend=True,
                                 legend_loc='ul',
                                 legendfontsize=5,
                                 linewidth=1,
                                 xlabel='frame number',
                                 ylabel='pixels')[0]

        self.y_delta = self.oplot(self.frames, self.pupil_y,
                                  color='blue',
                                  label='y delta',
                                  linewidth=1)[0]

        self.x_apos = self.oplot(self.frames, self.x_norm,
                                 color='orange',
                                 label='x pos',
                                 linewidth=1)[0]

        self.y_apos = self.oplot(self.frames, self.y_norm,
                                 color='purple',
                                 label='y pos',
                                 linewidth=1)[0]

        self.pup_an = self.oplot(self.frames, self.angle_data,
                                 color='green',
                                 label='angle',
                                 linewidth=1,
                                 side='right',
                                 ymin=0,
                                 ymax=180,
                                 # ylabel='angle (deg)'
                                 )[0]

        self.copy_background()

    def copy_background(self):
        """
        Copys background for quicker drawing.
        """
        self.background = self.fig.canvas.copy_from_bbox(self.axes.bbox)

    def calc(self):
        pupil_data = self.data[0]
        refle_data = self.data[1]

        self.x_norm = pupil_data[:, 0] - pupil_data[0][0]
        self.y_norm = pupil_data[:, 1] - pupil_data[0][1]

        refle_x_norm = refle_data[:, 0] - refle_data[0][0]
        refle_y_norm = refle_data[:, 1] - refle_data[0][1]

        self.pupil_x = self.x_norm - refle_x_norm
        self.pupil_y = self.y_norm - refle_y_norm

    def on_draw(self, verbose=False):
        self.calc()
        self.fig.canvas.restore_region(self.background)

        self.x_delta.set_ydata(self.pupil_x)
        self.y_delta.set_ydata(self.pupil_y)

        if verbose:
            self.x_apos.set_ydata(self.x_norm)
            self.y_apos.set_ydata(self.y_norm)
            self.pup_an.set_ydata(self.angle_data)

        self.axes.draw_artist(self.x_delta)
        self.axes.draw_artist(self.y_delta)

        if verbose:
            self.axes.draw_artist(self.x_apos)
            self.axes.draw_artist(self.y_apos)
            self.axes.draw_artist(self.pup_an)

        self.fig.canvas.blit(self.axes.bbox)

    def clear_plot(self):
        self.clear()


class MyFrame(wx.Frame):
    """
    Class for generating main frame. Holds other panels.
    """
    def __init__(self):
        """
        Constructor
        """
        # super instantiation
        super(MyFrame, self).__init__(None,
                                      title='PupilTracker',
                                      size=(-1, -1))

        # instance attributes
        self.playing = False
        self.verbose = False
        self.to_plot = False
        self.to_pip = False
        self.to_save_video = False
        self.to_dump_data = False
        self.save_video_name = None
        self.dump_file_name = None

        # tracker params
        self.pupil_thresh = 50
        self.refle_thresh = 190

        # instantiate tracker
        self.tracker = PupilTracker(self)

        # create panels
        self.image_panel = ImagePanel(self)
        self.tools_panel = ToolsPanel(self)
        self.plots_panel = PlotPanel(self)

        # sizer for image and tools panels
        image_tools_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # add panels to sizer
        image_tools_sizer.Add(self.image_panel,
                              flag=wx.EXPAND,
                              proportion=1)
        image_tools_sizer.Add(self.tools_panel,
                              flag=wx.EXPAND)

        # sizer for image/tools and plot
        panel_plot_sizer = wx.BoxSizer(wx.VERTICAL)

        # add to sizer
        panel_plot_sizer.Add(image_tools_sizer,
                             flag=wx.EXPAND,
                             proportion=1)
        panel_plot_sizer.Add(self.plots_panel,
                             flag=wx.EXPAND)

        # set sizer
        self.plots_panel.Hide()
        self.SetSizer(panel_plot_sizer)
        panel_plot_sizer.Fit(self)

        # status bar
        self.CreateStatusBar(2)
        self.SetStatusText('hi there', 0)
        # self.SetStatusWidths([-1, -1])

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_MAXIMIZE, self.on_maximize)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        # change background color to match panels on win32
        if platform == 'win32':
            self.SetBackgroundColour(wx.NullColour)

        # draw frame
        self.Show()

    def draw(self, img=None):
        """
        Draws frame.

        :param img: image to draw, will override getting frame
        """
        self.image_panel.draw(img=img)

    def play(self):
        """
        Plays video.
        """
        self.image_panel.start_timer()
        self.playing = True

    def pause(self):
        """
        Pauses video.
        """
        self.image_panel.stop_timer()
        self.playing = False

        if self.to_save_video:
            self.toggle_to_save_video(False)

        if self.to_dump_data:
            self.toggle_to_dump_data(False)

    def stop(self):
        """
        Stops the video, returning to beginning.
        """
        self.image_panel.stop_timer()
        self.playing = False

        if self.to_save_video:
            self.toggle_to_save_video(False)

        if self.to_dump_data:
            self.toggle_to_dump_data(False)

        # load first frame
        try:
            self.tracker.load_first_frame()
            self.tracker.clear_rois()
            self.load_frame(self.tracker.get_frame())
        except IOError as e:
            print(e)

    def clear(self, draw=False, keep_roi=False):
        """
        Clears drawings from the frame and optionally redraws the new blank
        frame. Also option to preserve ROIs (for when redrawing adjusted
        thresholds)

        :param draw: whether or not to redraw
        :param keep_roi: whether or not to keep ROIs
        """
        self.tracker.clear_frame()
        if not keep_roi:
            self.clear_rois()
        if draw:
            self.draw()

    def clear_rois(self):
        """
        Clears ROIs.
        """
        self.tracker.clear_rois()

    def clear_indices(self):
        """
        Clears indicies in tools panel.
        """
        self.tools_panel.clear_indices()

    def draw_pupil(self, pupil_index=None):
        """
        Draws the pupil to the frame.

        :param pupil_index: which pupil to draw
        """
        if pupil_index is not None:
            self.tracker.draw_pupil(index=pupil_index,
                                    roi=None,
                                    verbose=self.verbose)

    def draw_refle(self, refle_index=None, roi=None):
        """
        Draws the reflection to the frame.

        :param refle_index: which reflection to draw
        """
        if refle_index is not None:
            self.tracker.draw_refle(index=refle_index,
                                    roi=roi,
                                    verbose=self.verbose)

    def redraw_pupil(self):
        """
        Redraws the pupil in the same location.
        """
        try:
            self.tracker.draw_pupil(index=None,
                                    roi='pupil',
                                    verbose=self.verbose)
        except AttributeError as e:
            print(e)
            pass

    def redraw_refle(self):
        """
        Redraws the reflection in the same location.
        """
        try:
            self.tracker.draw_refle(index=None,
                                    roi='refle',
                                    verbose=self.verbose)
        except AttributeError as e:
            print(e)
            pass

    def track_pupil(self):
        """
        Tracks a pupil every frame.
        """
        self.tracker.track_pupil(verbose=self.verbose)

    def track_refle(self):
        """
        Tracks a reflection every frame.
        """
        self.tracker.track_refle(verbose=self.verbose)

    def next_frame(self):
        """
        Seeks to next frame.
        """
        self.tracker.next_frame()

    def get_frame(self):
        """
        Gets frame from tracker.
        """
        return self.tracker.get_frame()

    def pip(self):
        """
        Make picture in picture.
        """
        self.tracker.pip()

    def write_out(self):
        """
        Writes frames to file.
        """
        self.tracker.write_out()

    def update_plot(self):
        """
        Draws new data to plot panel.
        """
        self.plots_panel.on_draw(self.verbose)

    def toggle_playing(self, set_to=None):
        """
        Toggles playing variable.

        :param set_to: overrides toggle
        """
        if set_to is not None:
            self.playing = set_to

        else:
            if self.playing:
                self.playing = False
            else:
                self.playing = True

    def toggle_to_plot(self):
        """
        Toggles whether or not plot is shown.
        """
        if self.to_plot:
            self.to_plot = False
            size = self.Size
            if not self.IsMaximized():
                self.SetSize((size[0], size[1]-self.plots_panel.plot_height))
            self.plots_panel.Hide()
            self.Layout()
        else:
            self.to_plot = True
            size = self.Size
            self.SetSize((size[0], size[1]+self.plots_panel.plot_height))
            self.plots_panel.Show()
            self.Layout()

    def toggle_to_pip(self):
        """
        Toggles whether or not to show PiP (picture in picture).
        """
        if self.to_pip:
            self.to_pip = False
        else:
            self.to_pip = True

    def toggle_verbose(self, pupil_index, refle_index):
        """
        Toggles whether or not to show PiP (picture in picture).
        """
        if self.verbose:
            self.verbose = False
        else:
            self.verbose = True

        if not self.playing:
            try:
                self.clear(draw=False, keep_roi=True)
            except IOError as e:
                # print(e)
                return

            if pupil_index is not None:
                self.redraw_pupil()
            if refle_index is not None:
                self.redraw_refle()
            self.draw()

        self.update_plot()

    def toggle_to_save_video(self, set_to=None):
        """
        Toggles whether or not will save frames to video file.

        :param set_to: overrides toggle
        """
        if set_to is not None:
            if not set_to and self.to_save_video:
                self.to_save_video = False
                self.tracker.release_out()
                self.tools_panel.save_video_toggle.SetValue(False)

            elif set_to and not self.to_save_video:
                was_playing = False
                if self.playing:
                    was_playing = True
                    self.stop()

                self.to_save_video = True
                self.save_dialog('video')
                self.tracker.init_out(self.save_video_name)

                if was_playing:
                    self.play()

            else:
                return

        else:
            if self.to_save_video:
                self.to_save_video = False
                self.tracker.release_out()
                self.tools_panel.save_video_toggle.SetValue(False)

            else:
                was_playing = False
                if self.playing:
                    was_playing = True
                    self.stop()

                self.to_save_video = True
                self.tracker.init_out(self.save_video_name)
                self.save_dialog('video')

                if was_playing:
                    self.play()

    def toggle_to_dump_data(self, set_to=None):
        """
        Toggles whether or not will save frames to video file.

        :param set_to: overrides toggle
        """
        if set_to is not None:
            if not set_to and self.to_dump_data:
                self.to_dump_data = False
                self.tracker.dump_data(self.dump_file_name)
                self.tools_panel.dump_data_toggle.SetValue(False)

            elif set_to and not self.to_dump_data:
                was_playing = False
                if self.playing:
                    was_playing = True
                    self.stop()

                self.to_dump_data = True
                self.save_dialog('data')

                if was_playing:
                    self.play()

            else:
                return

        else:
            if self.to_dump_data:
                self.to_dump_data = False
                self.tracker.dump_data(self.dump_file_name)
                self.tools_panel.dump_data_toggle.SetValue(False)

            else:
                was_playing = False
                if self.playing:
                    was_playing = True
                    self.stop()

                self.to_dump_data = True
                self.save_dialog('data')

                if was_playing:
                    self.play()

    def open_video(self, video_file):
        """
        Opens the video and loads the first frame. Makes plot.

        :param video_file: video file to open
        """
        self.SetStatusText(video_file, 0)

        width = self.image_panel.GetClientRect()[2]

        self.tracker.init_cap(video_file, width)

        # load first frame
        self.load_frame(self.tracker.get_frame())

        self.plots_panel.init_plot(self.tracker.data, self.tracker.angle_data)

    def load_frame(self, img):
        """
        Loads frame to image panel.

        :param img: frame to load
        """
        self.image_panel.load_image(img)

    def load_dialog(self):
        """
        Popup dialog to open file.
        """
        self.pause()
        default_dir = path.abspath(
            r'C:\Users\Alex\PycharmProjects\EyeTracker\vids')

        # popup save dialog
        load_dialog = wx.FileDialog(self,
                                    message='File path',
                                    defaultDir=default_dir,
                                    # wildcard='*.txt',
                                    style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)

        # to exit out of popup on cancel button
        if load_dialog.ShowModal() == wx.ID_CANCEL:
            return

        # get path from save dialog and open
        video_file = load_dialog.GetPath()
        self.open_video(video_file)

    def save_dialog(self, filetype):
        """
        Popup dialog to save file.
        """
        default_dir = path.abspath(
            r'C:\Users\Alex\PycharmProjects\EyeTracker\vids\saved')

        # popup save dialog
        if filetype == 'video':
            card = 'mov'
        elif filetype == 'data':
            card = 'txt'
        save_dialog = wx.FileDialog(self,
                                    message='File path',
                                    defaultDir=default_dir,
                                    wildcard='*.' + card,
                                    style=wx.FD_SAVE)

        # to exit out of popup on cancel button
        if save_dialog.ShowModal() == wx.ID_CANCEL:
            if filetype == 'video':
                self.toggle_to_save_video(False)
            elif filetype == 'data':
                self.toggle_to_dump_data(False)
            return

        # get path from save dialog and open
        file_path = save_dialog.GetPath()
        if filetype == 'video':
            self.save_video_name = file_path
        elif filetype == 'data':
            self.dump_file_name = file_path

    def on_close(self, evt):
        """
        Catches close event. Exits gracefully.

        :param evt: required event parameter
        """
        self.pause()
        try:
            self.tracker.release_cap()
        except IOError as e:
            # print(e)
            pass
        evt.Skip()

    def on_size(self, evt):
        """
        Catches resize event.

        :param evt: required event parameter
        """
        new_width = self.image_panel.GetClientRect()[2]
        size = self.tracker.get_set_scaled_size(new_width)

        try:
            self.tracker.on_size()
            self.image_panel.on_size(size, self.get_frame())
            # self.plots_panel.copy_background()
            # self.plots_panel.init_plot(
            # TODO: fix plot background sizing problem
        except IOError:
            pass

        evt.Skip()

    def on_maximize(self, evt):
        """
        Catches maximize event. Because of bug with getting size after
        maximize, just simulate another size event so can get updated size.

        :param evt: required event parameter
        """
        evt = wx.CommandEvent(wx.EVT_SIZE.typeId,
                              self.Id)
        self.ProcessEvent(evt)


def main():
    """
    Main function to start GUI
    """
    # instantiate app
    global app
    app = wx.App(False)
    # instantiate window
    frame = MyFrame()
    # run app
    app.MainLoop()

if __name__ == '__main__':
    main()