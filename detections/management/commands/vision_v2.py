import os

from django.core.management.base import BaseCommand
from dotenv import load_dotenv
from loguru import logger

from detections.management.commands.models.agent import Agent
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.architecture import Architecture


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
        error = None
        agent = None

        if options["hailo"]:
            from vcgencmd import Vcgencmd

            archi = Architecture.HAILO
            log_temp = Vcgencmd()
        else:
            archi = Architecture.NATIF
            log_temp = None

        if options["video"]:
            source = Agent_Source.VIDEO
        elif options["photo"]:
            source = Agent_Source.PHOTO
        else:
            source = Agent_Source.VISION

        try:
            logger.add(f"{os.getenv('LOG_DIRECTORY')}{source}_{archi}.log",
                       format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                       rotation="1 day", retention=7)

            agent = Agent(archi, source)
            agent.start()
            agent.finish()


        except Exception as ex:
            error = ex
            trace = ""

            print(error)

            tb = ex.__traceback__
            while tb is not None:
                file = tb.tb_frame.f_code.co_filename.replace(
                    'home/jaden/Projects/followchon_back/', '')
                line = str(tb.tb_lineno)
                trace = f" in file {file} on line {line}"
                print(trace)

                tb = tb.tb_next

        finally:
            # if error is OperationalError:
            if type(error).__name__ == 'OperationalError':
                if agent is not None:
                    agent.finish()
                self.handle(*args, **options)
