import os
import sys

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger
from vcgencmd import Vcgencmd

from configuration.models import Log
from detections.management.commands.vision_models.model_hailo import Model_Hailo
from detections.management.commands.vision_models.streamer import Streamer


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        logger.add(f"{os.getenv('LOG_DIRECTORY')}vision_hailo.log",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   rotation="1 day", retention=7)

        model = None
        streamer = None

        try:
            model = Model_Hailo()

            streamer = Streamer(model)
            streamer.start()

        except Exception as error:
            if model is not None:
                model.release()

            logger.error(repr(error))

            ex_type, ex_value, ex_traceback = sys.exc_info()
            Log().create(
                model.current_model_version,
                'vision',
                'error',
                type(error).__name__,
                ex_value,
                Vcgencmd().measure_temp()
            )
