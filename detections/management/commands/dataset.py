import os
import pathlib
import random
import shutil

from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from helpers.file import FileHelper


def extract(items, count):
    extracts = list()

    for i in range(count):
        choice = random.choice(items)
        extracts.append(pathlib.Path(choice).stem)

        items.remove(choice)

    return extracts


def copy_to(files, ext, src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for file in files:
        if os.path.exists(f"{src_dir}/{file}.{ext}"):
            shutil.copy(f"{src_dir}/{file}.{ext}",
                        f"{dst_dir}/{file}.{ext}")


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

        dataset_source_labels_path = './live/saved/labels'
        dataset_source_images_path = './live/saved/images'

        dataset_test_labels_path = './live/dataset/test/labels'
        dataset_test_images_path = './live/dataset/test/images'
        dataset_val_labels_path = './live/dataset/val/labels'
        dataset_val_images_path = './live/dataset/val/images'

        dataset_train_labels_path = './live/dataset/train/labels'
        dataset_train_images_path = './live/dataset/train/images'

        dataset_test_percent = float(os.getenv('DATASET_TEST_PERCENT'))
        dataset_val_percent = float(os.getenv('DATASET_VAL_PERCENT'))

        annotations = FileHelper.list_files(dataset_source_labels_path, r'.*\.(txt)$').tolist()

        backup_count = len(annotations)
        test_count = int(backup_count * dataset_test_percent)
        val_count = int(backup_count * dataset_val_percent)
        train_count = backup_count - test_count - val_count

        tests = extract(annotations, test_count)
        copy_to(tests, 'txt', dataset_source_labels_path, dataset_test_labels_path)
        copy_to(tests, 'jpg', dataset_source_images_path, dataset_test_images_path)

        vals = extract(annotations, val_count)
        copy_to(vals, 'txt', dataset_source_labels_path, dataset_val_labels_path)
        copy_to(vals, 'jpg', dataset_source_images_path, dataset_val_images_path)

        trains = extract(annotations, train_count)
        copy_to(trains, 'txt', dataset_source_labels_path, dataset_train_labels_path)
        copy_to(trains, 'jpg', dataset_source_images_path, dataset_train_images_path)

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
