import os
import sys

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger
from vcgencmd import Vcgencmd

from configuration.models import Log
from detections.management.commands.vision_models.model_hailo import Model_Hailo
from detections.management.commands.vision_models.streamer import Streamer
from detections.management.commands.vision_models.supervisor import Supervisor


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        logger.add(f"{os.getenv('LOG_DIRECTORY')}vision_hailo.log",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   rotation="1 day", retention=7)

        supervisor = Supervisor(Vcgencmd())
        model = None
        streamer = None

        try:
            model = Model_Hailo(supervisor)

            streamer = Streamer(model)
            streamer.start()

        except Exception as error:
            if model is not None:
                model.release()

            logger.error(repr(error))

            ex_type, ex_value, ex_traceback = sys.exc_info()
            Log().create(
                supervisor.current_model_version,
                'vision',
                'error',
                type(error).__name__,
                f"{ex_value} : {str(error)}",
                Vcgencmd().measure_temp()
            )
