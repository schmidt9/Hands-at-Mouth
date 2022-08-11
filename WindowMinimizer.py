import pyautogui
from HandsAtMouthListener import HandsAtMouthListener


class WindowMinimizer(HandsAtMouthListener):

    def execute_action(self):
        try:
            # getActiveWindow() not found on macOS
            pyautogui.getActiveWindow().minimize()
        except (pyautogui.PyAutoGUIException, AttributeError) as ex:
            print(ex)
