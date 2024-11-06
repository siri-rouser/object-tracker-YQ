from typing import List

from visionapi_YQ.messages_pb2 import SaeMessage

class MessageBuffer:
    def __init__(self, target_window_size_ms: int) -> None:
        self._messages: List[SaeMessage] = []
        self._target_window_size_ms = target_window_size_ms

    def append(self, msg: SaeMessage) -> None:
        self._messages.append(msg)
        self._messages.sort(key=lambda m: m.frame.timestamp_utc_ms) # sort the message based on the timestamp




    def is_healthy(self) -> bool:
        '''Check if the buffer has enough messages to satisfy the window size condition.'''
        return len(self._messages) > 0 and self._messages[-1].frame.timestamp_utc_ms - self._messages[0].frame.timestamp_utc_ms >= self._target_window_size_ms
        