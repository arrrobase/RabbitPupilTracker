#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
GUI for PupilTracker
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

# from sys import platform
import wx
import os
import cv2
import numpy as np

class PupilTracker(object):
    """
    Image processing class.
    """
    def __init__(self, app):
        """
        Constructor.
        """
        self.app = app
        self.cap = None
        self.num_frames = None
        self.frame = None
        self.dx = None
        self.dy = None
        self.orig_image = None
        self.noise_kernel = None
        self.roi_pupil = None
        self.roi_refle = None

    def load_video(self, video_file):
        """
        Creates capture object for video
        :param video_file: video path
        """
        self.cap = cv2.VideoCapture(video_file)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.noise_kernel = np.ones((3, 3), np.uint8)

        # draw first frame
        self.next_frame()
        self.app.load_video(self.frame)

        # go back to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def next_frame(self):
        """
        Gets next frame.

        :return: next frame
        :raise EOFError: if at end of video file
        """
        if self.cap is not None:
            ret, next_frame = self.cap.read()
            if ret:
                frame = cv2.resize(next_frame, (0, 0), fx=0.5, fy=0.5)
                self.orig_image = frame.copy()
                self.frame = frame
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raise EOFError('Video end.')
        else:
            raise IOError('No video loaded.')

    def get_frame(self):
        if self.frame is not None:
            return self.frame

    def get_orig_frame(self):
        if self.orig_image is not None:
            self.frame = self.orig_image.copy()
            self.roi_pupil = None
            self.roi_refle = None

            return self.orig_image

        else:
            raise AttributeError('Nothing to clear to.')

    def process_image(self, img, roi=None):
        # print roi

        if roi is not None:
            # roi
            self.dx = roi[0][0]
            self.dy = roi[0][1]
            roi_image = img[roi[0][1]:roi[1][1],
                            roi[0][0]:roi[1][0]]
            # gauss
            gauss = cv2.GaussianBlur(roi_image, (5, 5), 0)

        else:
            self.dx = 0
            self.dy = 0
            # gauss
            gauss = cv2.GaussianBlur(img, (5, 5), 0)

        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

        return gray

    def find_pupils(self, roi=None):
        # roi and gauss
        grayed = self.process_image(self.frame, roi)
        # threshold and remove noise
        _, thresh_pupil = cv2.threshold(grayed, 45, 255, cv2.THRESH_BINARY)
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

                if not 1000 < area < 5000:
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
        if index is None:
            index = 0

        cnt = self.find_pupils(roi)[index]

        # fit ellipse
        ellipse = cv2.fitEllipse(cnt)

        # centroid
        cx = int(ellipse[0][0])
        cy = int(ellipse[0][1])

        # draw
        cv2.circle(self.frame, (cx, cy), 2, (255,255,255))
        if verbose:
            cv2.drawContours(self.frame, cnt, -1, (0, 0, 255), 2)
            cv2.ellipse(self.frame, ellipse, (0, 255, 100), 1)
            cv2.rectangle(self.frame, (cx-100, cy-100), (cx+100, cy+100), (255, 255,
                                                                    255))

        self.roi_pupil = [(cx - 100, cy - 100), (cx + 100, cy + 100)]

    def track_pupil(self):
        if self.roi_pupil is not None:
            try:
                self.draw_pupil(roi=self.roi_pupil)
            except IndexError:
                pass
        else:
            pass

    def find_refle(self, roi=None):
        # roi and gauss
        grayed = self.process_image(self.frame, self.roi_pupil)
        # threshold and remove noise
        _, thresh_refle = cv2.threshold(grayed, 200, 255, cv2.THRESH_BINARY)
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

                if not 25 < area < 400:
                    continue

                # rescale to full image
                cnt[:, :, 0] += self.dx
                cnt[:, :, 1] += self.dy

                found_reflections.append(cnt)

        return found_reflections

    def draw_refle(self, index=None, roi=None, verbose=True):
        if index is None:
            index = 0

        cnt_list = self.find_refle(roi)

        if len(cnt_list) > 0:
            cnt = cnt_list[index]
        else:
            raise AttributeError('No reflections found.')

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # rect center
        cx = int(rect[0][0])
        cy = int(rect[0][1])

        # draw
        cv2.circle(self.frame, (cx, cy), 2, (100, 100, 100))
        if verbose:
            cv2.rectangle(self.frame, (cx-15, cy-15), (cx+15, cy+15), (255, 255, 255))
            cv2.drawContours(self.frame, cnt, -1, (0, 0, 255), 2)
            cv2.drawContours(self.frame, [box], 0, (0, 255, 100), 1)

    def track_refle(self):
        if self.roi_refle is not None:
            try:
                self.draw_refle(roi=self.roi_pupil)
            except IndexError:
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
        Constructor
        """
        # super instantiation
        super(ImagePanel, self).__init__(parent, size=(960, 540))

        # instance attributes
        self.app = parent
        self.image_ctrl = wx.StaticBitmap(self)
        self.image_bmp = None
        self.orig_image = None

    def load_image(self, img):
        """
        Creates buffer loader and loads first image
        """
        self.image_bmp = wx.BitmapFromBuffer(960, 540, img)
        self.image_ctrl.SetBitmap(self.image_bmp)

    def draw(self, img):
        """
        Draws drawings.
        """
        self.image_bmp.CopyFromBuffer(img)
        self.image_ctrl.SetBitmap(self.image_bmp)


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

        # find buttons
        self.find_pupil_button = wx.Button(self, label='Find pupil')
        self.find_refle_button = wx.Button(self, label='Find refle')
        self.clear_button = wx.Button(self, label='Clear')
        self.load_button = wx.Button(self, label='Load')
        self.play_button = wx.Button(self, label='Play')

        # button sizer
        button_sizer = wx.BoxSizer(wx.VERTICAL)

        # add buttons to sizer
        button_sizer.Add(self.find_pupil_button,
                         flag=wx.LEFT | wx.RIGHT,
                         border=5)
        button_sizer.Add(self.find_refle_button,
                         flag=wx.LEFT | wx.RIGHT,
                         border=5)
        button_sizer.Add(self.clear_button,
                         flag=wx.LEFT | wx.RIGHT,
                         border=5)
        button_sizer.Add(self.load_button,
                         flag=wx.LEFT | wx.RIGHT,
                         border=5)
        button_sizer.Add(self.play_button,
                         flag=wx.LEFT | wx.RIGHT,
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
            self.app.clear()
            self.app.draw()
        except AttributeError as e:
            print e
            return
        self.pupil_index = -1
        self.refle_index = -1

    def on_load_button(self, evt):
        video_file = os.path.abspath(
            r'C:\Users\Alex\PycharmProjects\EyeTracker\vids\00091_short.mov')
        self.app.open_video(video_file)
        self.pupil_index = -1
        self.relfe_index = -1

    def on_play_button(self, evt):
        while True:
            try:
                self.app.tracker.next_frame()
            except EOFError as e:
                print e
                break
            except IOError as e:
                print e
                break

            self.app.tracker.track_pupil()
            self.app.tracker.track_refle()
            self.app.draw()

        self.pupil_index = -1
        self.refle_index = -1


class MyFrame(wx.Frame):
    """
    Class for generating main frame. Holds other panels.
    """
    def __init__(self):
        """
        Constructor
        """
        # super instantiation
        super(MyFrame, self).__init__(None, title='PupilTracker', size=(960, 540))

        # instantiate tracker
        self.tracker = PupilTracker(self)

        self.image_panel = ImagePanel(self)
        self.tools_panel = ToolsPanel(self)

        # sizer for panels
        panel_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # add panels to sizer
        panel_sizer.Add(self.image_panel,
                        flag=wx.EXPAND)
        panel_sizer.Add(self.tools_panel,
                        flag=wx.EXPAND)

        # set sizer
        self.SetSizer(panel_sizer)
        panel_sizer.Fit(self)

        # draw frame
        self.Show()

    def draw(self):
        self.image_panel.draw(self.tracker.get_frame())

    def load_video(self, img):
        self.image_panel.load_image(img)

    def open_video(self, video_file):
        self.tracker.load_video(video_file)

    def draw_pupil(self, index):
        self.clear()
        if index != -1:
            self.tracker.draw_pupil(index=index, verbose=True)
        self.draw()

    def draw_refle(self, pupil_index, refle_index):
        self.clear()
        if pupil_index != -1:
            self.tracker.draw_pupil(index=pupil_index, verbose=True)
        self.tracker.draw_refle(index=refle_index, verbose=True)
        self.draw()

    def clear(self):
        self.tracker.get_orig_frame()


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