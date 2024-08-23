from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from detections.management.commands.vision_models.model_hailo_cmd import Model_Hailo_cmd
from detections.management.commands.vision_models.streamer import Streamer


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        model = Model_Hailo_cmd()

        streamer = Streamer(model)
        streamer.start()

        model.destruct()

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
