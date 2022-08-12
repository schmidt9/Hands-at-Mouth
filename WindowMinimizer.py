import pyautogui
from HandsAtMouthListener import HandsAtMouthListener


class WindowMinimizer(HandsAtMouthListener):

    def __init__(self, window_title):
        self.window_title = window_title

    def execute_action(self):
        try:
            # getWindowsWithTitle() not implemented on macOS
            for window in pyautogui.getWindowsWithTitle(self.window_title):
                window.minimize()
        except (pyautogui.PyAutoGUIException, AttributeError) as ex:
            print(ex)
