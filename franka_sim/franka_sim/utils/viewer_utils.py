import mujoco.viewer
import threading
import platform

class MujocoViewer:
    def __init__(self, model, data, single_window=True, key_callback=None):
        self.model = model
        self.data = data
        self.single_window = single_window
        self.viewer_1 = None
        self.viewer_2 = None
        self.key_callback = key_callback

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def launch(self):
        if platform.system() == 'Darwin':
            self._launch_on_main_thread()
        else:
            self._launch_viewers()

    def _launch_on_main_thread(self):
        if threading.current_thread() is threading.main_thread():
            self._launch_viewers()
        else:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._launch_viewers)

    def _launch_viewers(self):
        self.viewer_1 = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self.key_callback
        )
        if not self.single_window:
            self.viewer_2 = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback
            )

    def is_running(self):
        if self.single_window:
            return self.viewer_1.is_running()
        else:
            return self.viewer_1.is_running() and self.viewer_2.is_running()

    def sync(self):
        if self.viewer_1:
            self.viewer_1.sync()
        if self.viewer_2:
            self.viewer_2.sync()

    def close(self):
        import glfw
        if self.viewer_1:
            self.viewer_1.close()
        if self.viewer_2:
            self.viewer_2.close()
        glfw.terminate()
