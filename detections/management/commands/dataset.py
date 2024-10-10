import os
import pathlib
import random
import shutil

from django.core.management.base import BaseCommand
from dotenv import load_dotenv

from configuration.models import Parameter
from helpers.array import ArrayHelper
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

    def get_param(self, param: str):
        if param in self.params_dict:
            return self.params_dict[param].value

        return None

    def handle(self, *args, **options):
        load_dotenv()

        dataset_source_path = os.getenv('DATASET_SOURCE_DIR')
        dataset_source_labels_path = f'{dataset_source_path}/labels'
        dataset_source_images_path = f'{dataset_source_path}/images'

        dataset_dir = os.getenv('DATASET_RESULT_DIR')

        if not dataset_dir:
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            new_model_version = int(params_dict['vision_model_version'].value) + 1
            dataset_dir = f'./static/captures/chons-v{new_model_version}'

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        dataset_test_labels_path = f'{dataset_dir}/test/labels'
        dataset_test_images_path = f'{dataset_dir}/test/images'
        dataset_val_labels_path = f'{dataset_dir}/val/labels'
        dataset_val_images_path = f'{dataset_dir}/val/images'
        dataset_train_labels_path = f'{dataset_dir}/train/labels'
        dataset_train_images_path = f'{dataset_dir}/train/images'

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

        shutil.copy(f"{dataset_source_path}/data.yaml",
                    f"{dataset_dir}/data.yaml")

        self.stdout.write(
            self.style.SUCCESS('Successfully finished')
        )
