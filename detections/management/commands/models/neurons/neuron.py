import cv2
from pubsub import pub

from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type


class Neuron():
    def __init__(self, model_ext: str = 'pt', score_min: float = 0.7):
        self.model_ext = model_ext
        self.current_model_version = None
        self.model = None
        self.score_min = score_min

    def log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.NEURON, action=action, infos=infos)

    def check(self, origin: str):
        raise Exception('Check not implemented')

    def process(self, frame: cv2.typing.MatLike):
        raise Exception('Process not implemented')

    def release(self):
        return False
