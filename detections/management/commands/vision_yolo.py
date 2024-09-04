import os

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger

from detections.management.commands.vision_models.model_yolo import Model_YOLO
from detections.management.commands.vision_models.streamer import Streamer
from detections.management.commands.vision_models.supervisor import Supervisor


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        logger.add(f"{os.getenv('LOG_DIRECTORY')}vision_hailo.log",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   rotation="1 day", retention=7)

        supervisor = Supervisor()
        model = Model_YOLO(supervisor)

        streamer = Streamer(model)
        streamer.start()
