import os

import cv2
from loguru import logger

from configuration.models import Family, Zone, Parameter
from detections.models import Detection
from helpers.array import ArrayHelper


class Model:

    def __init__(self):
        self.model = None
        self.model_version = ''
        self.model_path = ''
        self.min_score = 0
        self.min_hour = 0
        self.max_hour = 0
        self.stop = False
        self.params_dict = {}

        self.save_time = 0
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.families = []
        self.zones = []
        self.families_dict = {}
        self.last_detections_dict = {}

    def log(self, message, channel=''):
        if os.getenv('VERBOSE') == 'True':
            print(message)
        else:
            match channel:
                case 'error':
                    logger.error(message)
                case _:
                    logger.info(message)

    def get_params(self):
        parameters = Parameter.objects.all()
        self.params_dict: dict[int, Parameter] = (
            ArrayHelper.object_list_to_dict(parameters, 'slug')
        )

    def get_param(self, param):
        if param in self.params_dict:
            return self.params_dict[param].value

        self.log(f'Param "{param}" not exist', "error")
        return None

    def fill_params(self):
        self.get_params()

        self.families = Family.objects.all()
        self.zones = Zone.objects.all()
        self.families_dict = ArrayHelper.object_list_to_dict(self.families, 'index')

        last_detections = Detection.objects.raw(
            'SELECT * FROM' +
            ' (SELECT * FROM detections_detection d' +
            ' LEFT JOIN configuration_family f ON d.family_id = f.id' +
            ' WHERE f.is_tracked = true'
            ' ORDER BY d.id DESC) d' +
            ' GROUP BY d.family_id')

        last_detections_dict: dict[int, Detection] = (
            ArrayHelper.object_list_to_dict(last_detections, 'family_id')
        )

        self.last_detections_dict: dict[int, Zone] = (
            dict(map(lambda kv: (kv[0], kv[1].zone), last_detections_dict.items())))

        self.min_score = float(self.get_param('vision_min_score'))
        self.min_hour = int(self.get_param('vision_min_hour'))
        self.max_hour = int(self.get_param('vision_max_hour'))

    def check_param(self, param, value) -> bool:
        return self.get_param(param) == value

    def reload(self):
        self.model_version = self.get_param('vision_model_version')

        self.log(f'Load model version "{self.model_version}"')

        self.model_path = (f"{os.getenv('MODEL_DIR')}/"
                           f"{os.getenv('MODEL_PREFIX')}{self.model_version}.{os.getenv('MODEL_EXT')}")

    def infer(self, frame: cv2.typing.MatLike):
        raise Exception('Infer not implemented')

    def check_model(self):
        raise Exception('Check model not implemented')
