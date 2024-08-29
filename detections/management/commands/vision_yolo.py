import os

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger

from detections.management.commands.vision_models.model_yolo import Model_YOLO
from detections.management.commands.vision_models.streamer import Streamer


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        logger.add(f"{os.getenv('LOG_DIRECTORY')}vision_yolo.log", format="{time} | {level} | {message}",
                   rotation="1 days", retention=7)

        model = Model_YOLO()

        streamer = Streamer(model)
        streamer.start()
