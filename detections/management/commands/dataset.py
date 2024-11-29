import os
import pathlib
import random
import shutil
from math import isnan

from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from configuration.models import Parameter
from utils.array import ArrayHelper
from utils.file import FileHelper


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


def chunk_list(lst, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        dataset_source_path = os.getenv('DATASET_SOURCE_DIR')
        chunk_number = int(os.getenv('DATASET_CHUNK_FIRST'))
        dataset_chunk = int(os.getenv('DATASET_CHUNK'))
        dataset_source_labels_path = f'{dataset_source_path}/labels'
        dataset_source_images_path = f'{dataset_source_path}/images'
        dataset_base_dir = os.getenv('DATASET_RESULT_DIR')

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version'].value) + 1

        dataset_test_labels_path = f'{dataset_dir}/test/labels'
        dataset_test_images_path = f'{dataset_dir}/test/images'
        dataset_val_labels_path = f'{dataset_dir}/val/labels'
        dataset_val_images_path = f'{dataset_dir}/val/images'
        dataset_train_labels_path = f'{dataset_dir}/train/labels'
        dataset_train_images_path = f'{dataset_dir}/train/images'

        dataset_test_percent = float(os.getenv('DATASET_TEST_PERCENT'))
        dataset_val_percent = float(os.getenv('DATASET_VAL_PERCENT'))

        annotations = FileHelper.list_files(dataset_source_labels_path, r'.*\.(txt)$').tolist()

        chunks = chunk_list(annotations, dataset_chunk)

        for chunk in chunks:
            dataset_dir = f'{dataset_base_dir}{chunk_number}'

            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)

            backup_count = len(chunk)

            test_count = int(backup_count * dataset_test_percent)
            val_count = int(backup_count * dataset_val_percent)
            train_count = backup_count - test_count - val_count

            tests = extract(chunk, test_count)
            copy_to(tests, 'txt', dataset_source_labels_path, dataset_test_labels_path)
            copy_to(tests, 'jpg', dataset_source_images_path, dataset_test_images_path)

            vals = extract(chunk, val_count)
            copy_to(vals, 'txt', dataset_source_labels_path, dataset_val_labels_path)
            copy_to(vals, 'jpg', dataset_source_images_path, dataset_val_images_path)

            trains = extract(chunk, train_count)
            copy_to(trains, 'txt', dataset_source_labels_path, dataset_train_labels_path)
            copy_to(trains, 'jpg', dataset_source_images_path, dataset_train_images_path)

            shutil.copy(f"{dataset_source_path}/data.yaml",
                        f"{dataset_dir}/data.yaml")

            chunk_number = chunk_number + 1

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
