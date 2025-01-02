import os
import pathlib

from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from detections.models import Capture


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        captures = Capture.objects.all()

        for capture in captures:
            if not os.path.exists(capture.photo_path(None, True)) or not os.path.exists(capture.label_path(None, True)):
                print(f"[{capture.id} : ")
                print(os.path.exists(capture.photo_path(None, True)), capture.photo_path(None, True))
                print(os.path.exists(capture.label_path(None, True)), capture.label_path(None, True))

                capture.remove_files()
                capture.delete()
            else:
                detections = capture.detections.all()
                detections_error = False

                for detection in detections:
                    if detection.width < 0 or detection.height < 0 or detection.center_x < 0 or detection.center_y < 0:
                        detections_error = True

                        detection.width = max(0, detection.width)
                        detection.height = max(0, detection.height)
                        detection.center_x = max(0, detection.center_x)
                        detection.center_y = max(0, detection.center_y)
                        detection.save()

                if detections_error:
                    label_path = capture.label_path(None, True)
                    content = "\n".join([detection.line() for detection in detections])
                    file = pathlib.Path(label_path)
                    file.write_text(content)

                    print(f"[{capture.id} : ", label_path)
                    print(content)

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
