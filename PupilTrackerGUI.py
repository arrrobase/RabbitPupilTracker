#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
Pupil tracking software.
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

# from sys import platform
from __future__ import division
import wx
import wxmplot # wx matplotlib library
import os
import cv2
import numpy as np
from PupilTracker import plot_data
from psychopy.core import MonotonicClock


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
        self.cap = None
        self.out = None
        self.frame = None
        self.frame_num = None
        self.num_frames = None
        self.vid_size = None
        self.scaled_size = None
        self.scale = None
        self.display_frame = None
        self.dx = None
        self.dy = None
        self.orig_frame = None
        self.noise_kernel = None
        self.roi_pupil = None
        self.roi_refle = None
        self.data = None
        self.angle = None
        self.angle_data = None
        self.cx_pupil = None
        self.cy_pupil = None
        self.cx_refle = None
        self.cy_refle = None

    def load_video(self, video_file, width):
        """
        Creates capture object for video.

        :param video_file: video path
        :param width: width of the window
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.cap = cv2.VideoCapture(video_file)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vid_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.set_scaled_size(width)

        self.data = np.empty((2, self.num_frames, 2))
        self.angle_data = np.empty(self.num_frames)

        self.data.fill(np.NaN)
        self.angle_data.fill(np.NaN)

        self.noise_kernel = np.ones((3, 3), np.uint8)

        self.load_first_frame()

    def init_out(self):
        if self.out is None:
            self.out = cv2.VideoWriter(self.app.save_video_name,
                                       fourcc=cv2.VideoWriter_fourcc('m', 'p',
                                                                     '4', 'v'),
                                       fps=60,
                                       frameSize=(960, 540))
        else:
            raise IOError('VideoWriter already created. Release first.')

    def release_out(self):
        if self.out is not None:
            self.out.release()
            self.out = None
            print 'Recording saved.'

    def write_out(self):
        if self.out is not None:
            self.out.write(self.display_frame)

    def clear_data(self):
        self.data.fill(np.NaN)
        self.angle_data.fill(np.NaN)

    def load_first_frame(self):
        """
        Loads the first frame into the GUI.
        """
        # draw first frame
        self.frame_num = -1
        self.next_frame()
        self.app.load_video(self.display_frame)

        # go back to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_num = -1

        # clear data
        self.clear_data()
        self.app.toggle_save_video(False)

    def set_scaled_size(self, width):
        """
        Tracks the scale of the window relative to original frame size.

        :param width: window size
        :return: scaled size of video
        """
        if self.vid_size is not None:
            self.scaled_size = (width,
                                int(width * self.vid_size[1] / self.vid_size[0]))

            self.scale = self.vid_size[0] / width

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

            return self.display_frame
        else:
            raise IOError('No video selected.')

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
                self.display_frame = cv2.resize(self.frame,
                                                (self.scaled_size[0],
                                                 self.scaled_size[1]))
                self.orig_frame = self.display_frame.copy()
                self.frame_num += 1
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # clear locations
                self.roi_pupil = None
                self.roi_refle = None
                self.load_first_frame()
                raise EOFError('Video end.')
        else:
            raise IOError('No video loaded.')

    def get_frame(self):
        """
        Gets the current frame to display.

        :return: current display frame
        """
        if self.display_frame is not None:
            return self.display_frame

    def get_orig_frame(self):
        """
        Gets the current frame before any changes were made.

        :return: unedited frame
        :raise AttributeError: if no original frame loaded
        """
        if self.orig_frame is not None:
            self.display_frame = self.orig_frame.copy()
            self.roi_pupil = None
            self.roi_refle = None

            return self.orig_frame

        else:
            raise AttributeError('Nothing to clear.')

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

    def find_pupils(self, roi=None):
        """
        Searches for possible pupils in processed image

        :param roi: region of interest
        :return: list of possible pupil contours
        """

        # roi and gauss
        grayed = self.process_image(self.frame, roi)
        # threshold and remove noise
        _, thresh_pupil = cv2.threshold(grayed, 50, 255, cv2.THRESH_BINARY)
        filtered_pupil = cv2.morphologyEx(thresh_pupil, cv2.MORPH_CLOSE,
                                          self.noise_kernel, iterations=4)
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
                    continue

                if not 4000 < area < 20000:
                    continue

                # drop too few points
                hull = cv2.convexHull(cnt)
                if hull.shape[0] < 5:
                    continue

                # drop too eccentric
                circumference = cv2.arcLength(hull, True)
                circularity = circumference ** 2 / (4*np.pi*area)
                if circularity >= 1.6:
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
        # if no index passed, means we are tracking single pupil
        if index is None:
            index = 0

        cnt_list = self.find_pupils(roi)

        if len(cnt_list) > 0:
            cnt = cnt_list[index]
        else:
            raise AttributeError('No pupils found.')

        # fit ellipse
        ellipse = cv2.fitEllipse(cnt)

        # centroid
        self.cx_pupil = int(ellipse[0][0])
        self.cy_pupil = int(ellipse[0][1])

        # angle of ellipse
        self.angle = ellipse[2]
        if self.angle > 90:
            self.angle -= 90
        else:
            self.angle += 90

        # scale for drawing
        scaled_cx = int(self.cx_pupil / self.scale)
        scaled_cy = int(self.cy_pupil / self.scale)
        self.scaled_cx = scaled_cx
        self.scaled_cy = scaled_cy

        # draw
        cv2.circle(self.display_frame, (scaled_cx, scaled_cy), 2, (255, 0, 0))
        roi_size = 200
        scaled_roi_size = int(roi_size / self.scale)
        self.scaled_roi_size = scaled_roi_size
        scaled_cnt = np.rint(cnt / self.scale)
        scaled_cnt = scaled_cnt.astype(int)
        scaled_ellipse = cv2.fitEllipse(scaled_cnt)

        if verbose:
            cv2.drawContours(self.display_frame, scaled_cnt, -1, (0, 0, 255), 2)
            cv2.ellipse(self.display_frame, scaled_ellipse, (0, 255, 100), 1)
            cv2.rectangle(self.display_frame,
                          (scaled_cx - scaled_roi_size, scaled_cy - scaled_roi_size),
                          (scaled_cx + scaled_roi_size, scaled_cy + scaled_roi_size),
                          (255, 255, 255))
            # box = cv2.boxPoints(ellipse)
            # box = np.int0(box)
            # cv2.drawContours(self.display_frame, [box], 0,(0,0,255),1)

        # correct out of bounds roi
        roi_lu_x = self.cx_pupil - roi_size
        roi_lu_y = self.cy_pupil - roi_size
        roi_rl_x = self.cx_pupil + roi_size
        roi_rl_y = self.cy_pupil + roi_size
        if roi_lu_x < 0:
            roi_lu_x = 0
        if roi_lu_y < 0:
            roi_lu_y = 0

        self.roi_pupil = [(roi_lu_x, roi_lu_y),
                          (roi_rl_x, roi_rl_y)]

    def track_pupil(self):
        """
        Makes call to draw pupil with proper roi and handles errors.
        """
        if self.roi_pupil is not None:
            try:
                self.draw_pupil(roi=self.roi_pupil)
                self.data[0][self.frame_num] = [self.cx_pupil, self.cy_pupil]
                self.angle_data[self.frame_num] = self.angle

                if self.app.pip_toggle:
                    roi_size = self.scaled_roi_size
                    roi_image = self.display_frame[
                                self.scaled_cy-roi_size+1:self.scaled_cy+roi_size,
                                self.scaled_cx-roi_size+1:self.scaled_cx+roi_size]

                    self.display_frame[0:roi_image.shape[0],
                                       self.display_frame.shape[1]-roi_image.shape[1]:self.display_frame.shape[1]] = \
                        roi_image

            except IndexError as e:
                # print e
                pass
            except AttributeError as e:
                # print e
                pass
        else:
            pass

    def find_refle(self, roi=None):
        """
        Searches for possible reflections in processed image

        :param roi: region of interest
        :return: list of possible reflection contours
        """

        # roi and gauss
        grayed = self.process_image(self.frame, self.roi_pupil)
        # threshold and remove noise
        _, thresh_refle = cv2.threshold(grayed, 190, 255, cv2.THRESH_BINARY)
        filtered_refle = cv2.morphologyEx(thresh_refle, cv2.MORPH_CLOSE,
                                          self.noise_kernel, iterations=2)
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
                    continue

                if not 80 < area < 2000:
                    continue

                # rescale to full image
                cnt[:, :, 0] += self.dx
                cnt[:, :, 1] += self.dy

                # test squareness
                rect = cv2.minAreaRect(cnt)
                w, h = rect[1][0], rect[1][1]
                squareness = h / w
                if not 0.5 < squareness < 2:
                    continue

                # see if center in roi
                if roi is not None:
                    # rect center
                    cx = int(rect[0][0])
                    cy = int(rect[0][1])
                    if not self.roi_refle[0][0] < cx < self.roi_refle[1][0] \
                            or not \
                            self.roi_refle[0][1] < cy < self.roi_refle[1][1]:
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

        cnt_list = self.find_refle(roi)

        if len(cnt_list) > 0:
            cnt = cnt_list[index]
        else:
            raise AttributeError('No reflections found.')

        rect = cv2.minAreaRect(cnt)

        # rect center
        self.cx_refle = int(rect[0][0])
        self.cy_refle = int(rect[0][1])

        # reset roi
        roi_size = 30
        scaled_roi_size = int(roi_size / self.scale)
        self.roi_refle = [(self.cx_refle - roi_size, self.cy_refle - roi_size),
                          (self.cx_refle + roi_size, self.cy_refle + roi_size)]

        # scale for drawing
        scaled_cx = int(self.cx_refle / self.scale)
        scaled_cy = int(self.cy_refle / self.scale)
        scaled_cnt = np.rint(cnt / self.scale)
        scaled_cnt = scaled_cnt.astype(int)
        scaled_rect = cv2.minAreaRect(scaled_cnt)
        box = cv2.boxPoints(scaled_rect)
        box = np.int0(box)

        # draw
        cv2.circle(self.display_frame, (scaled_cx, scaled_cy), 2, (100, 100, 100))
        if verbose:
            cv2.rectangle(self.display_frame,
                          (scaled_cx - scaled_roi_size, scaled_cy - scaled_roi_size),
                          (scaled_cx + scaled_roi_size, scaled_cy + scaled_roi_size),
                          (255, 255, 255))
            cv2.drawContours(self.display_frame, scaled_cnt, -1, (0, 0, 255), 2)
            cv2.drawContours(self.display_frame, [box], 0, (0, 255, 100), 1)

    def track_refle(self):
        """
        Makes call to draw reflection with proper roi and handles errors.
        """
        if self.roi_refle is not None:
            try:
                self.draw_refle(roi=self.roi_refle)
                self.data[1][self.frame_num] = [self.cx_refle, self.cy_refle]
            except IndexError as e:
                # print e
                pass
            except AttributeError as e:
                # print e
                pass
        else:
            pass


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
        self.t = None

        self.SetDoubleBuffered(True)
        self.fps = 1000
        self.fps_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.draw, self.fps_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def start_timer(self):
        """
        Starts timer for draw timing.
        """
        self.fps_timer.Start(1000 // self.fps)
        self.t = MonotonicClock()

    def stop_timer(self):
        """
        Stops timer.
        """
        self.fps_timer.Stop()
        # try:
        #     t = self.t.getTime()
        #     f = self.app.tracker.num_frames
        #     print t, f,
        #     print f/t
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
        self.Refresh()

    def draw(self, evt=None):
        """
        Draws frame passed from tracking class.
        """
        if self.app.playing:
            try:
                if self.app.plot_toggle and self.app.tracker.frame_num % 2 == 0:
                    self.app.plots_panel.on_draw()
                self.app.tracker.next_frame()
            except EOFError as e:
                print e
                self.app.playing = False
                self.stop_timer()
                return
            except IOError as e:
                print e
                self.app.playing = False
                self.stop_timer()
                return

            self.app.tracker.track_pupil()
            self.app.tracker.track_refle()
            self.app.tracker.write_out()

        self.image_bmp.CopyFromBuffer(self.app.tracker.get_frame())
        self.Refresh()

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
        self.Refresh()


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
        self.pupil_index = -1
        self.refle_index = -1

        # buttons
        self.find_pupil_button = wx.Button(self, label='Find pupil')
        self.find_refle_button = wx.Button(self, label='Find refle')
        self.clear_button = wx.Button(self, label='Clear')
        self.load_button = wx.Button(self, label='Load')
        self.play_button = wx.Button(self, label='Play')
        self.stop_button = wx.Button(self, label='Stop')

        # toggles
        self.plot_toggle = wx.CheckBox(self, label='Plot')
        self.plot_toggle.SetValue(True)
        self.pip_toggle = wx.CheckBox(self, label='PIP')
        self.pip_toggle.SetValue(False)
        self.save_video_toggle = wx.CheckBox(self, label='Save video')
        self.save_video_toggle.SetValue(False)

        # button sizer
        button_sizer = wx.BoxSizer(wx.VERTICAL)

        # add buttons to sizer
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
        button_sizer.Add(self.stop_button,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.plot_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.pip_toggle,
                         flag=wx.LEFT | wx.RIGHT | wx.TOP,
                         border=5)
        button_sizer.Add(self.save_video_toggle,
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
                  self.on_stop_button,
                  self.stop_button)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_plot_toggle,
                  self.plot_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_pip_toggle,
                  self.pip_toggle)
        self.Bind(wx.EVT_CHECKBOX,
                  self.on_save_video_toggle,
                  self.save_video_toggle)

        # set sizer
        self.SetSizer(button_sizer)

    def on_find_pupil_button(self, evt):
        try:
            self.pupil_index += 1
            self.app.draw_pupil(self.pupil_index)
        except IndexError:
            self.pupil_index = -1
            self.on_find_pupil_button(evt)
        except AttributeError as e:
            self.pupil_index = -1
            print e

    def on_find_refle_button(self, evt):
        try:
            self.refle_index += 1
            self.app.draw_refle(self.pupil_index, self.refle_index)
        except IndexError:
            self.refle_index = -1
            self.on_find_refle_button(evt)
        except AttributeError as e:
            print e

    def on_clear_button(self, evt):
        try:
            self.app.clear(draw=True)
        except AttributeError as e:
            print e
            return
        self.pupil_index = -1
        self.refle_index = -1

    def on_load_button(self, evt):
        self.pupil_index = -1
        self.refle_index = -1

        self.app.load_dialog()

    def on_play_button(self, evt):
        self.app.play()

    def on_stop_button(self, evt):
        self.app.stop()

    def on_plot_toggle(self, evt):
        self.app.toggle_plot()

    def on_pip_toggle(self, evt):
        self.app.toggle_pip()

    def on_save_video_toggle(self, evt):
        self.app.toggle_save_video()


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
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.line4 = None
        self.line5 = None
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
        self.line1 = self.plot(self.frames, self.pupil_x,
                               ymin=0-guess_dif,
                               ymax=0+guess_dif,
                               color='red',
                               label='x',
                               markersize=5,
                               labelfontsize=5,
                               show_legend=True,
                               legend_loc='ul',
                               legendfontsize=5,
                               linewidth=1,
                               xlabel='frame number',
                               ylabel='delta (pixels)')[0]

        self.line2 = self.oplot(self.frames, self.pupil_y,
                                color='blue',
                                label='y',
                                linewidth=1)[0]

        self.line3 = self.oplot(self.frames, self.x_norm,
                                color='orange',
                                label='x pos',
                                linewidth=1)[0]

        self.line4 = self.oplot(self.frames, self.y_norm,
                                color='purple',
                                label='y pos',
                                linewidth=1)[0]

        self.line5 = self.oplot(self.frames, self.angle_data,
                                color='green',
                                label='angle',
                                linewidth=1,
                                side='right',
                                ymin=0,
                                ymax=180,
                                # ylabel='angle (deg)'
                                )[0]

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

    def on_draw(self):
        self.calc()
        self.fig.canvas.restore_region(self.background)
        self.line1.set_ydata(self.pupil_x)
        self.line2.set_ydata(self.pupil_y)
        self.line3.set_ydata(self.x_norm)
        self.line4.set_ydata(self.y_norm)
        self.line5.set_ydata(self.angle_data)
        self.axes.draw_artist(self.line1)
        self.axes.draw_artist(self.line2)
        self.axes.draw_artist(self.line3)
        self.axes.draw_artist(self.line4)
        self.axes.draw_artist(self.line5)
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
        super(MyFrame, self).__init__(None, title='PupilTracker', size=(-1,
                                                                        -1))
        # instance attributes
        self.playing = False
        self.plot_toggle = True
        self.pip_toggle = False
        self.save_video_toggle = False
        self.save_video_name = None

        # instantiate tracker
        self.tracker = PupilTracker(self)

        self.image_panel = ImagePanel(self)
        self.tools_panel = ToolsPanel(self)
        self.plots_panel = PlotPanel(self)

        # sizer for image and tools panels
        image_tools_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # add panels to sizer
        image_tools_sizer.Add(self.image_panel,
                              flag=wx.EXPAND,
                              proportion=1)
        image_tools_sizer.Add(self.tools_panel)

        # sizer for image/tools and plot
        panel_plot_sizer = wx.BoxSizer(wx.VERTICAL)

        # add to sizer
        panel_plot_sizer.Add(image_tools_sizer,
                             flag=wx.EXPAND,
                             proportion=1)
        panel_plot_sizer.Add(self.plots_panel,
                             flag=wx.EXPAND)

        # set sizer
        # self.plots_panel.Hide()
        self.SetSizer(panel_plot_sizer)
        panel_plot_sizer.Fit(self)

        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_MAXIMIZE, self.on_size)

        # draw frame
        self.Show()

    def draw(self):
        self.image_panel.draw()

    def load_dialog(self):
        default_dir = os.path.abspath(
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
        self.stop()
        self.open_video(video_file)

    def open_video(self, video_file):
        w = self.image_panel.GetClientRect()[2]
        self.tracker.load_video(video_file, width=w)
        self.plots_panel.init_plot(self.tracker.data, self.tracker.angle_data)

    def load_video(self, img):
        self.image_panel.load_image(img)

    def draw_pupil(self, pupil_index):
        self.clear()
        if pupil_index != -1:
            self.tracker.draw_pupil(index=pupil_index, verbose=True)
        self.image_panel.draw()

    def save_dialog(self):
        default_dir = os.path.abspath(
            r'C:\Users\Alex\PycharmProjects\EyeTracker\vids\saved')

        # popup save dialog
        save_dialog = wx.FileDialog(self,
                                    message='File path',
                                    defaultDir=default_dir,
                                    wildcard='*.mov',
                                    style=wx.FD_SAVE)

        # to exit out of popup on cancel button
        if save_dialog.ShowModal() == wx.ID_CANCEL:
            self.toggle_save_video(False)
            return

        # get path from save dialog and open
        video_file = save_dialog.GetPath()
        self.save_video_name = video_file

    def draw_refle(self, pupil_index, refle_index):
        self.clear()
        if pupil_index != -1:
            self.tracker.draw_pupil(index=pupil_index, verbose=True)
            self.image_panel.draw()
        self.tracker.draw_refle(index=refle_index, verbose=True)
        self.image_panel.draw()

    def clear(self, draw=False):
        self.tracker.get_orig_frame()
        if draw:
            self.draw()

    def play(self):
        self.image_panel.start_timer()
        self.playing = True

    def stop(self):
        self.image_panel.stop_timer()
        self.playing = False
        if self.save_video_toggle:
            self.toggle_save_video(False)

    def on_size(self, evt):
        w = self.image_panel.GetClientRect()[2]
        size = self.tracker.set_scaled_size(w)
        try:
            img = self.tracker.on_size()
            self.image_panel.on_size(size, img)
        except IOError:
            pass

        evt.Skip()

    def on_maximize(self, evt):
        self.on_size(evt)

    def toggle_plot(self):
        if self.plot_toggle:
            self.plot_toggle = False
            size = self.Size
            self.SetSize((size[0], size[1]-self.plots_panel.plot_height))
            self.plots_panel.Hide()
            self.Layout()
        else:
            self.plot_toggle = True
            size = self.Size
            self.SetSize((size[0], size[1]+self.plots_panel.plot_height))
            self.plots_panel.Show()
            self.Layout()

    def toggle_pip(self):
        if self.pip_toggle:
            self.pip_toggle = False
        else:
            self.pip_toggle = True

    def toggle_save_video(self, set_to=None):

        if self.save_video_toggle or set_to is False:
            self.tracker.release_out()
            self.save_video_toggle = False
            self.tools_panel.save_video_toggle.SetValue(False)

        elif not self.save_video_toggle or set_to:
            was_playing = False
            if self.playing:
                was_playing = True
                self.stop()

            self.save_video_toggle = True
            self.save_dialog()
            self.tracker.init_out()

            if was_playing:
                self.play()


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