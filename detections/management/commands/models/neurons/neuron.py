import cv2
from pubsub import pub

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.tools import get_param


class Neuron():
    def __init__(self, model_ext: str = 'pt'):
        self.model_ext = model_ext
        self.current_model_version = None
        self.model = None
        self.score_min = float(get_param('vision_score_min'))

    def log(self, event: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.NEURON, event=event, infos=infos, level=level)

    def check(self, reason: str):
        raise Exception('Check not implemented')

    def process(self, frame: cv2.typing.MatLike):
        raise Exception('Process not implemented')

    def release(self):
        return False
