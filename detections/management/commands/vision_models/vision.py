import os
import time
from datetime import datetime

import cv2

from configuration.models import Family, Zone
from detections.management.commands.vision_models.archi import Archi
from detections.management.commands.vision_models.capture_analyse import Capture_analyse
from detections.management.commands.vision_models.model_yolo_classify import Model_YOLO_Classify
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.supervisor import Supervisor
from detections.models import Detection
from utils.array import ArrayHelper
from utils.image import ImageHelper


class Vision:
    def __init__(self, supervisor: Supervisor):
        self.supervisor = supervisor
        self.families = []
        self.zones = []
        self.families_index_dict = {}
        self.families_slug_dict = {}
        self.last_detections_dict = {}

        self.save_time = 0

        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.families = Family.objects.all()
        self.families_index_dict = ArrayHelper.object_list_to_dict(self.families, 'index')
        self.families_slug_dict = ArrayHelper.object_list_to_dict(self.families, 'slug')
        self.model_classify = None

        if self.supervisor.source == Source.VISION:
            last_detections = Detection.objects.raw(
                'SELECT * FROM (' +
                '    SELECT * FROM detections_detection d' +
                '    LEFT JOIN configuration_family f ON d.family_id = f.id' +
                '    WHERE f.is_tracked = true'
                '    ORDER BY d.id DESC'
                ' ) d' +
                ' GROUP BY d.family_id')

            last_detections_dict: dict[int, Detection] = (
                ArrayHelper.object_list_to_dict(last_detections, 'family_id')
            )

            self.last_detections_dict: dict[int, (float, float)] = (
                dict(
                    map(
                        lambda kv: (kv[0], (kv[1].center_x, kv[1].center_y)),
                        last_detections_dict.items()
                    )
                )
            )

        self.fill_objects()

        if self.supervisor.source == Source.VISION or os.getenv('ENABLE_CLASSIFY'):
            self.model_classify = Model_YOLO_Classify(supervisor)

        if self.supervisor.archi == Archi.HAILO:
            from detections.management.commands.vision_models.model_hailo_detect import Model_Hailo_Detect
            self.model_detect = Model_Hailo_Detect(supervisor, self.supervisor.source)
        else:
            from detections.management.commands.vision_models.model_yolo_detect import Model_YOLO_Detect
            self.model_detect = Model_YOLO_Detect(supervisor, self.supervisor.source)

    def release(self):
        if self.supervisor.archi == Archi.HAILO:
            self.model_detect.release()

    def check_model(self, origin):
        if self.model_detect is not None:
            self.model_detect.check_model(origin)

        if self.model_classify is not None:
            self.model_classify.check_model(origin)

    def fill_objects(self):
        if self.supervisor.source == Source.VISION:
            self.zones = Zone.objects.all().filter(is_enabled=True).order_by('id')

    def filter(self, yolo_all_results, detect_safes, detect_unsafes):
        detect_unsafes_bis = list()

        try_ok = False

        for detect in detect_unsafes:
            infers = list(
                filter(
                    lambda infer: self.families_slug_dict[infer[0]].index not in detect_safes,
                    detect['infers']
                )
            )

            if len(infers) > 0:
                if len(infers) == 1 or detect['try'] > 0:
                    cls = self.families_slug_dict[infers[0][0]].index

                    yolo_all_results.append(detect['result'].clone(cls, infers[0][1]))

                    detect_safes.append(cls)
                else:
                    if not try_ok:
                        detect['try'] = 1
                        try_ok = True

                    detect_unsafes_bis.append(detect)

        return yolo_all_results, detect_safes, detect_unsafes_bis

    def infer(self, frame: cv2.typing.MatLike, frame_count, capture_date: datetime):
        saved = False

        yolo_results = self.model_detect.infer(frame)
        yolo_all_results = list()

        detect_safes = list()
        detect_unsafes = list()

        if len(yolo_results) > 0:
            if self.model_classify is not None:
                for yolo_result in yolo_results:
                    image_result = frame[
                                   yolo_result.ortho_tl_y:yolo_result.ortho_br_y,
                                   yolo_result.ortho_tl_x:yolo_result.ortho_br_x
                                   ]

                    classify_results = self.model_classify.infer(image_result)

                    if len(classify_results) > 0:
                        if len(classify_results) == 1:
                            cls = self.families_slug_dict[classify_results[0][0]].index

                            if cls > 0:
                                yolo_all_results.append(yolo_result.clone(cls, classify_results[0][1]))

                                detect_safes.append(cls)
                        else:
                            detect_unsafes.append({
                                'result': yolo_result,
                                'infers': classify_results,
                                'try': 0
                            })

                    yolo_all_results.append(yolo_result)

                if len(detect_unsafes) > 0:
                    while True:
                        if len(detect_unsafes) == 0:
                            break

                        yolo_all_results, detect_safes, detect_unsafes = self.filter(yolo_all_results, detect_safes,
                                                                                     detect_unsafes)
            else:
                yolo_all_results = yolo_results

            if len(yolo_all_results) > 0:
                analyse = Capture_analyse(
                    self.model_detect.current_model_version,
                    self.model_classify.current_model_version if self.model_classify is not None else None,
                    frame, capture_date, frame_count,
                    self.last_detections_dict, self.families_index_dict, self.zones,
                    self.supervisor
                )

                frame = analyse.detect(yolo_all_results)

                if analyse.is_triggered:
                    if os.getenv('ENABLE_SAVE'):
                        analyse.save()

                    self.save_time = time.time()
                    saved = True

        return ImageHelper.resize_with_ratio(frame, self.capture_width, None), saved
