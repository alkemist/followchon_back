import os

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger

from detections.management.commands.vision_models.model_hailo import Model_Hailo
from detections.management.commands.vision_models.streamer import Streamer


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        logger.add(f"{os.getenv('LOG_DIRECTORY')}vision_hailo.log", format="{time} | {level} | {message}",
                   rotation="1 days", retention=7)

        model = None

        try:
            model = Model_Hailo()

            streamer = Streamer(model)
            streamer.start()

        except Exception as error:
            if model is not None:
                model.release()

            logger.error(error)
