#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
GUI for StimProgram
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

# from sys import platform
import wx
import os


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
        super(ImagePanel, self).__init__(parent)


class MyFrame(wx.Frame):
    """
    Class for generating main frame. Holds other panels.
    """
    def __init__(self):
        """
        Constructor
        """
        # super instantiation
        super(MyFrame, self).__init__(None, title='PupilTracker')

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