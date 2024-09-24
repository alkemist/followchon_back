import os

from django.core.management.base import BaseCommand
from django.db import OperationalError
from dotenv import load_dotenv
from loguru import logger
from vcgencmd import Vcgencmd

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
        error = None

        try:
            model = Model_Hailo(supervisor)

            streamer = Streamer(model)
            streamer.start()

        except Exception as ex:
            error = ex
            trace = ""

            tb = ex.__traceback__
            while tb is not None:
                file = tb.tb_frame.f_code.co_filename.replace(
                    'home/jaden/Projects/followchon_back/', '')
                line = str(tb.tb_lineno)
                trace = f" in file {file} on line {line}"
                
                tb = tb.tb_next

            if model is not None:
                model.release()

            supervisor.log(
                type(error).__name__,
                f"{str(error)}{trace}",
                'fail'
            )

        finally:
            if error is OperationalError:
                self.handle()
