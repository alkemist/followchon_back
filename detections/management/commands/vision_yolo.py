from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from detections.management.commands.vision_models.model_yolo import Model_YOLO
from detections.management.commands.vision_models.streamer import Streamer


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        model = Model_YOLO()

        streamer = Streamer(model)
        streamer.start()

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
