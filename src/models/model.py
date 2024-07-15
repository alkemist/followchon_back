from ultralytics import YOLO
import time

from ..data.zones import data_zones
from ..helpers.image import ImageHelper
from .annotation import Annotation
from .capture import Capture
from .zone import Zone
from .classes.noisette import Noisette
from .classes.stitch import Stitch


class Model:

    def __init__(self, model_path, capture_width, capture_height, capture_min_conf):
        self.model = YOLO(model_path, task='detect')
        Annotation.set_zones([
            Zone(
                zone['name'],
                zone['points'][0],
                zone['points'][1],
                zone['points'][2],
                zone['points'][3]
            ) for zone in data_zones
        ])

        self.capture_width = capture_width
        self.capture_height = capture_height

        self.noisette = Noisette()
        self.stitch = Stitch()

        Capture.set_classes(range(len(self.model.names)))
        Capture.set_capture_size(self.capture_width, self.capture_height)
        Capture.set_capture_min_conf(capture_min_conf)
        Annotation.set_labels(self.model.names)
        self.save_time = 0

    def detect(self, frame, captures_directory, save_enabled=False, verbose=False):
        results = self.model(frame, stream=True, verbose=False)

        save_time_elapsed = time.time() - self.save_time
        noisette_moved = False
        stitch_moved = False

        capture = Capture(frame)
        frame = capture.detect(results)

        if capture.cls_counts[Noisette.cls] == 1\
                and capture.cls_zones[Noisette.cls] is not None:
            noisette_moved = self.noisette.set_zone(capture.cls_zones[Noisette.cls])

        if capture.cls_counts[Stitch.cls] == 1\
                and capture.cls_zones[Stitch.cls] is not None:
            stitch_moved = self.stitch.set_zone(capture.cls_zones[Stitch.cls])

        if save_enabled and capture.valid and save_time_elapsed > 1:
            self.save_time = time.time()
            capture.save(captures_directory)

        # if verbose and len(capture.log) > 0:
        #     print(capture.log)

            # if noisette_moved:
            #     print(f"{Noisette.name} => {self.noisette.zone.name.value}")
            #
            # if stitch_moved:
            #     print(f"{Stitch.name} => {self.stitch.zone.name.value}")

        return ImageHelper.resize_with_ratio(frame, self.capture_width, self.capture_height)
