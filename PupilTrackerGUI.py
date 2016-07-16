#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python

"""
Pupil tracking software.
"""

# Copyright (C) 2016 Alexander Tomlinson
# Distributed under the terms of the GNU General Public License (GPL).

from __future__ import division, print_function
import wx
import wxmplot  # wx matplotlib library
import numpy as np
from os import path
from sys import platform
from PupilTracker import PupilTracker
# from psychopy.core import MonotonicClock  # for getting display fps


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

    def draw(self, evt=None, img=None, step=False, direction='forward'):
        """
        Draws frame passed from tracking class.

        :param evt: Required event parameter
        """
        if self.app.playing or step:
            try:
                if direction == 'forward':
                    self.app.next_frame()
                elif direction == 'backward':
                    try:
                        self.app.prev_frame()
                    except EOFError:
                        # print e
                        return

                self.app.SetStatusText(str(self.app.tracker.frame_num+1) +
                                       '/' + str(
                    self.app.tracker.num_frames), 1)
            except EOFError as e:
                print(e)
                self.app.toggle_playing(set_to=False)
                self.stop_timer()
                self.app.clear_rois()
                self.app.clear_indices()
                self.load_image(self.app.get_frame())
                self.app.SetStatusText(str(self.app.tracker.frame_num+1) +
                                       '/' + str(
                    self.app.tracker.num_frames), 1)
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
        except IndexError:
            self.pupil_index = 0
            self.on_find_pupil_button(evt)

        # no pupils found, so draw blank (or with reflection if found)
        except AttributeError:
            self.pupil_index = None
            self.app.draw()
            # print(e)

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
            self.app.clear(draw=True, keep_roi=False)
        except IOError as e:
            print(e)

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
        except IOError:
            return

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError:
                pass

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError:
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
        except IOError:
            return

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError:
                pass

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError:
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
        except AttributeError:
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
        except IOError:
            return

        # redraw pupil if present
        if self.pupil_index is not None:
            try:
                self.app.redraw_pupil()
            except AttributeError:
                pass

        # redraw reflection if present
        if self.refle_index is not None:
            try:
                self.app.redraw_refle()
            except AttributeError:
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

        # setup menus
        file_menu = wx.Menu()
        file_open = file_menu.Append(wx.ID_OPEN,
                                     'Open',
                                     'Open a movie to analyze')
        file_camera = file_menu.Append(wx.ID_CANCEL,
                                       'Webcam',
                                       'Use webcam as video stream')

        help_menu = wx.Menu()
        help_about = help_menu.Append(wx.ID_ABOUT,
                                      'About',
                                      'Information about this application')

        # create menu bar
        menu_bar = wx.MenuBar()
        menu_bar.Append(file_menu, 'File')
        menu_bar.Append(help_menu, 'Help')

        # set menu bar
        self.SetMenuBar(menu_bar)

        # event binders
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_MAXIMIZE, self.on_maximize)
        self.Bind(wx.EVT_CLOSE, self.on_close)

        self.Bind(wx.EVT_MENU, self.on_file_open, file_open)
        self.Bind(wx.EVT_MENU, self.on_file_camera, file_camera)
        self.Bind(wx.EVT_MENU, self.on_help_about, help_about)

        # keyboard binders
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)

        # change background color to match panels on win32
        if platform == 'win32':
            self.SetBackgroundColour(wx.NullColour)

        # draw frame
        self.Show()

    def draw(self, img=None, step=False, direction='forward'):
        """
        Draws frame.

        :param img: image to draw, will override getting frame
        """
        self.image_panel.draw(img=img, step=step, direction=direction)

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
        Clears indices in tools panel.
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

    def prev_frame(self):
        """
        Seeks to next frame.
        """
        self.tracker.prev_frame()

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
            except IOError:
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
                try:
                    self.tracker.dump_data(self.dump_file_name)
                # no filename passed
                except TypeError:
                    # print(e)
                    pass
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

    def on_file_open(self, evt):
        """
        Menu event for file, open.

        :param evt: required event parameter
        :return:
        """
        self.tools_panel.clear_indices()
        self.load_dialog()

    def on_file_camera(self, evt):
        """
        Menu event for streaming webcam.

        :param evt: required event parameter
        :return:
        """
        self.tools_panel.clear_indices()
        self.open_video('webcam')

    def on_help_about(self, evt):
        """
        Menu event for help, about.

        :param evt: required event parameter
        :return:
        """
        message = 'Rabbit pupil tracking software.' \
                  '\nCopyright (C) 2016 Alexander Tomlinson' \
                  '\nDistributed under the terms of the GNU General Public' \
                  'License (GPL)'

        wx.MessageBox(message, 'About',
                      style=wx.OK)

    def on_close(self, evt):
        """
        Catches close event. Exits gracefully.

        :param evt: required event parameter
        """
        self.pause()
        try:
            self.tracker.release_cap()
        except IOError:
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
            self.redraw_pupil()
            self.redraw_refle()
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

    def on_key_down(self, evt):
        """
        Catches key down event.

        :param evt: required event parameter
        """
        key = evt.GetKeyCode()

        if key == wx.WXK_RIGHT:
            self.draw(step=True, direction='forward')

        if key == wx.WXK_LEFT:
            self.draw(step=True, direction='backward')
            # print('sorry, can\'t yet')

        else:
            evt.Skip()


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
