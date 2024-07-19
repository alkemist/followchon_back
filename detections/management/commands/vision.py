from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from detections.management.commands.vision_models.streamer import Streamer


# from detections.models import Detection


class Command(BaseCommand):
    help = ""

    # def add_arguments(self, parser):
    #     parser.add_argument("poll_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        # for poll_id in options["poll_ids"]:
        #     try:
        #         poll = Poll.objects.get(pk=poll_id)
        #     except Poll.DoesNotExist:
        #         raise CommandError('Poll "%s" does not exist' % poll_id)
        #
        #     poll.opened = False
        #     poll.save()
        #

        load_dotenv()

        streamer = Streamer()
        streamer.start()

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
