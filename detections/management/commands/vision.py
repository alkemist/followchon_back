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
            help="Source video",
        )
        parser.add_argument(
            "--chons",
            action="store_true",
            help="Classification",
        )
        parser.add_argument(
            "--photo",
            action="store_true",
            help="Source photo",
        )

    def handle(self, *args, **options):
        load_dotenv()
        error = None
        agent = None

        if options["hailo"]:
            archi = Architecture.HAILO
        else:
            archi = Architecture.NATIF

        if options["video"]:
            source = Agent_Source.VIDEO
        elif options["photo"]:
            source = Agent_Source.PHOTO
        else:
            source = Agent_Source.VISION

        logger.add(f"{os.getenv('LOG_DIRECTORY')}{source}_{archi}.log",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                   rotation="00:00", retention="1 month")

        try:
            agent = Agent(archi, source, options["chons"])
            agent.start()
            agent.end()


        except Exception as ex:
            error = ex

            print(error)

            tb = ex.__traceback__
            while tb is not None:
                file = tb.tb_frame.f_code.co_filename.replace(
                    'home/jaden/Projects/followchon_back/', '')
                line = str(tb.tb_lineno)
                print(f" in file {file} on line {line}")

                tb = tb.tb_next

        finally:
            # if error is OperationalError:
            if type(error).__name__ == 'OperationalError':
                if agent is not None:
                    agent.end()
                self.handle(*args, **options)
