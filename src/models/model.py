import time

from ultralytics import YOLO

from helpers.image import ImageHelper
from .frame import Frame


class Model:

    def __init__(self, model_path, capture_width, capture_height, capture_min_conf):
        self.model = YOLO(model_path, task='detect')

        self.capture_width = capture_width
        self.capture_height = capture_height

        Frame.set_classes(range(len(self.model.names)))
        Frame.set_capture_size(self.capture_width, self.capture_height)
        Frame.set_capture_min_conf(capture_min_conf)
        self.save_time = 0

    def detect(self, frame, save_enabled=False, verbose=False):
        results = self.model(frame, stream=True, verbose=False)

        save_time_elapsed = time.time() - self.save_time
        noisette_moved = False
        stitch_moved = False

        capture = Frame(frame)
        frame = capture.detect(results)

        # if capture.cls_counts[Noisette.cls] == 1 \
        #         and capture.cls_zones[Noisette.cls] is not None:
        #     noisette_moved = self.noisette.set_zone(capture.cls_zones[Noisette.cls])
        #
        # if capture.cls_counts[Stitch.cls] == 1 \
        #         and capture.cls_zones[Stitch.cls] is not None:
        #     stitch_moved = self.stitch.set_zone(capture.cls_zones[Stitch.cls])

        if save_enabled and capture.valid and save_time_elapsed > 1:
            self.save_time = time.time()
            capture.save()

        # if verbose and len(capture.log) > 0:
        #     print(capture.log)

        # if noisette_moved:
        #     print(f"{Noisette.name} => {self.noisette.zone.name.value}")
        #
        # if stitch_moved:
        #     print(f"{Stitch.name} => {self.stitch.zone.name.value}")

        return ImageHelper.resize_with_ratio(frame, self.capture_width, self.capture_height)
