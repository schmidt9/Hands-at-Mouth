from typing import List
from HandsAtMouthListener import HandsAtMouthListener


class HandsAtMouthHandler:

    def __init__(self):
        self.listeners: List[HandsAtMouthListener] = []

    def add_listener(self, listener: HandsAtMouthListener):
        self.listeners.append(listener)

    def handle_hands_at_mouth(self):
        for listener in self.listeners:
            listener.execute_action()
