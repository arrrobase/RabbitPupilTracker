#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
GUI for StimProgram
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

# from sys import platform
import wx
import os
import cv2
import time
import numpy as np


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
        self.imageCtrl = wx.StaticBitmap(self)

        # set image
        self.video_file = os.path.abspath(r'C:\Users\Alex\PycharmProjects\EyeTracker\vids'
                             r'\00091_short.mov')
        self.roi_pupil = [(450-100, 260-100), (450+100, 260+100)] # 91
        self.roi_refl  = [(411, 224), (461, 274)] # 91
        self.kernel = np.ones((3, 3), np.uint8)

        self.cap = cv2.VideoCapture(self.video_file)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = 960, 540

        ret, self.frame = self.cap.read()
        self.ori_img = cv2.resize(self.frame, (0, 0), fx=0.5, fy=0.5)
        self.img = self.ori_img.copy()
        self.image = wx.BitmapFromBuffer(w, h, self.img)

        self.imageCtrl.SetBitmap(self.image)

    def next_frame(self):
        ret , frame = self.cap.read()
        # rescale
        self.img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        self.ori_img = self.img.copy()

    def process_img(self):
        # roi
        self.dx = self.roi_pupil[0][0]
        self.dy = self.roi_pupil[0][1]
        roi = self.img[self.roi_pupil[0][1]:self.roi_pupil[1][1],
              self.roi_pupil[0][0]:self.roi_pupil[1][0]]
        # gauss
        gauss = cv2.GaussianBlur(roi, (5,5), 0)
        # make grayscale
        self.gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

    def find_pupil(self):
        self.process_img()
        # threshold
        _, thresh_pupil = cv2.threshold(self.gray, 40, 255, cv2.THRESH_BINARY)

        # remove noise
        filt_pupil = cv2.morphologyEx(thresh_pupil, cv2.MORPH_CLOSE,
                                      self.kernel, iterations=4)
        # find contours
        cont_pupil = filt_pupil.copy()
        _, contours_pupil, _ = cv2.findContours(cont_pupil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

                hull[:, :, 0] += self.dx
                hull[:, :, 1] += self.dy

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
                # data[0][i] = [cx, cy]

                # draw ellipse, contours, centroid
                # if verbose:
                cv2.drawContours(self.img, hull, -1, (0, 0, 255), 2)
                cv2.ellipse(self.img, ellipse, (0, 255, 100), 1)
                cv2.circle(self.img, (cx, cy), 2, (255,255,255))

                # reset roi
                if cx < 100:
                    cx = 100
                if cy < 100:
                    cy = 100
                roi_pupil = [(cx - 100, cy -100), (cx + 100, cy + 100)]

                # draw roi
                # if verbose:
                cv2.rectangle(self.img, (cx-100, cy-100), (cx+100, cy+100),
                              (255, 255, 255))

    def find_refle(self):
        self.process_img()
        # threshold
        _, thresh_refl = cv2.threshold(self.gray, 200, 255,
                                       cv2.THRESH_BINARY_INV)
        # remove noise
        filt_refl = cv2.morphologyEx(thresh_refl, cv2.MORPH_CLOSE, self.kernel,
                                     iterations=2)
        # find contours
        cont_refl = filt_refl.copy()
        _, contours_refl, _ = cv2.findContours(cont_refl, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if len(contours_refl) != 0:
            for cnt in contours_refl:
                # drop small and large
                area = cv2.contourArea(cnt)
                # print area
                if area == 0:
                    continue
                if not 25 < area < 400:
                    continue

                cnt[:, :, 0] += self.dx
                cnt[:, :, 1] += self.dy

                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # rect center
                cx = int(rect[0][0])
                cy = int(rect[0][1])

                # see if center in roi
                if not self.roi_refl[0][0] < cx < self.roi_refl[1][0] or not self.roi_refl[0][1] <\
                        cy < self.roi_refl[1][1]:
                    continue
                # print 'refl:', (cx, cy)

                # add to data
                # data[1][i] = [cx, cy]

                # reset roi
                self.roi_refl = [(cx - 15, cy - 15), (cx + 15, cy + 15)]

                # draw
                # if verbose:
                cv2.rectangle(self.img, (cx-15, cy-15), (cx+15, cy+15), (255, 255, 255))
                cv2.drawContours(self.img, cnt, -1, (0, 0, 255), 2)
                cv2.drawContours(self.img, [box], 0, (0, 255, 100), 1)
                cv2.circle(self.img, (cx, cy), 2, (100, 100, 100))

    def draw(self):
        self.image.CopyFromBuffer(self.img)
        self.imageCtrl.SetBitmap(self.image)
        self.Refresh()

    def clear(self):
        self.img = self.ori_img.copy()
        self.image.CopyFromBuffer(self.img)
        self.imageCtrl.SetBitmap(self.image)
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
        self.frame = parent

        # find buttons
        self.find_pupil_button = wx.Button(self, label='Find pupil')
        self.find_refle_button = wx.Button(self, label='Find reflection')
        self.clear_button = wx.Button(self, label='Clear')
        self.advance_button = wx.Button(self, label='Advance')

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
        button_sizer.Add(self.advance_button,
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
                  self.on_advance_button,
                  self.advance_button)

        # set sizer
        self.SetSizer(button_sizer)

    def on_find_pupil_button(self, evt):
        self.frame.image_panel.find_pupil()
        self.frame.image_panel.draw()

    def on_find_refle_button(self, evt):
        self.frame.image_panel.find_refle()
        self.frame.image_panel.draw()

    def on_clear_button(self, evt):
        self.frame.image_panel.clear()

    def on_advance_button(self, evt):
        # self.frame.image_panel.next_frame()
        # self.frame.image_panel.draw()
        for i in range(60):
            self.frame.image_panel.next_frame()
            self.frame.image_panel.find_pupil()
            self.frame.image_panel.find_refle()
            self.frame.image_panel.draw()


class DataPanel(wx.Panel):
    """
    Class for panel with output graphs of pupil locations.

    :param parent: parent window of panel
    """
    def __init__(self, parent):
        """
        Constructor
        """
        # super instantiation
        super(DataPanel, self).__init__(parent)


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

        self.image_panel = ImagePanel(self)
        self.tools_panel = ToolsPanel(self)

        # sizer for panels
        panel_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # add panels to sizer
        panel_sizer.Add(self.image_panel,
                        proportion=1,
                        flag=wx.EXPAND)
        panel_sizer.Add(self.tools_panel,
                        proportion=1,
                        flag=wx.EXPAND)

        # set sizer
        self.SetSizer(panel_sizer)
        panel_sizer.Fit(self)

        # draw frame
        self.Show()


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