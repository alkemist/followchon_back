import os

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger

from detections.management.commands.vision_models.archi import Archi
from detections.management.commands.vision_models.levels import Levels
from detections.management.commands.vision_models.source import Source
from detections.management.commands.vision_models.streamer import Streamer
from detections.management.commands.vision_models.supervisor import Supervisor


class Command(BaseCommand):
    help = ""

    def add_arguments(self, parser):
        parser.add_argument(
            "--hailo",
            action="store_true",
            help="Use hailo model",
        )
        parser.add_argument(
            "--video",
            action="store_true",
            help="Sources videos",
        )
        parser.add_argument(
            "--photo",
            action="store_true",
            help="Sources videos",
        )

    def handle(self, *args, **options):
        load_dotenv()

        if options["hailo"]:
            from vcgencmd import Vcgencmd

            model_name = 'hailo'
            archi = Archi.HAILO
            log_temp = Vcgencmd()
        else:
            model_name = 'yolo'
            archi = Archi.NATIF
            log_temp = None

        if options["video"]:
            source = Source.VIDEO
        elif options["photo"]:
            source = Source.PHOTO
        else:
            source = Source.VISION

        logger.add(f"{os.getenv('LOG_DIRECTORY')}{source}_{model_name}.log",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   rotation="1 day", retention=7)

        supervisor = Supervisor(log_temp, source, archi)
        error = None
        streamer = None

        try:
            streamer = Streamer(supervisor)

            if options["photo"]:
                streamer.read()
            else:
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

            if streamer is not None:
                streamer.release()

            supervisor.log(
                type(error).__name__,
                f"{str(error)}{trace}",
                Levels.FAIL
            )

        finally:
            # if error is OperationalError:
            if type(error).__name__ == 'OperationalError':
                self.handle(*args, **options)
