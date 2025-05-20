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
            file = pathlib.Path(capture.label_path(None, True))
            file.write_text("\n".join([detection.line() for detection in capture.detections.all()]))

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
