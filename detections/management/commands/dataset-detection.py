import math
import os
import pathlib
import random
import shutil
from math import isnan

import pandas as pd
from django.core.management.base import BaseCommand
from django.db import connection
from dotenv import load_dotenv

from configuration.models import Parameter
from detections.models import Capture
from utils.array import ArrayHelper


def extract(items, count):
    extracts = list()

    for i in range(count):
        choice = random.choice(items)
        extracts.append(pathlib.Path(choice).stem)

        items.remove(choice)

    return extracts


def filter_yolo_file(input_file_path, output_file_path, min_width, min_height):
    filtered_lines = []

    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            elements = line.split()
            if len(elements) == 5:
                try:
                    cls = float(elements[0])
                    width = float(elements[3])
                    height = float(elements[4])
                    if cls == 0 and width >= min_width and height >= min_height:
                        filtered_lines.append(line)
                except ValueError:
                    print(f"Invalid line values: {line}")
            else:
                print(f"Invalid line format: {line}")

    with open(output_file_path, 'w') as output_file:
        output_file.writelines(filtered_lines)


def copy_to(df, dist_dir, min_width, min_height):
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.exists(f"{dist_dir}/images"):
        os.makedirs(f"{dist_dir}/images")
    if not os.path.exists(f"{dist_dir}/labels"):
        os.makedirs(f"{dist_dir}/labels")

    for index in df.index:
        shutil.copy(
            str(df.loc[index, 'image_path']),
            f"{dist_dir}/images/" + df.loc[index, 'filename'] + "." + df.loc[index, 'ext']
        )

        filter_yolo_file(
            str(df.loc[index, 'label_path']),
            f"{dist_dir}/labels/" + df.loc[index, 'filename'] + f".txt",
            min_width,
            min_height,
        )


def chunk_list(lst, chunk_size):
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def query_to_dataframe(query):
    """
    Exécute une requête SQL brute et retourne un DataFrame Pandas
    avec les colonnes automatiquement définies.
    """
    with connection.cursor() as cursor:
        # Exécuter la requête SQL
        cursor.execute(query)

        # Extraire les noms des colonnes
        columns = [col[0] for col in cursor.description]

        # Extraire les données
        rows = cursor.fetchall()

    # Créer le DataFrame
    return pd.DataFrame(rows, columns=columns)


def replace_extension(path, new_ext=""):
    base, _ = os.path.splitext(path)
    return base + new_ext


def get_extension(path):
    return os.path.splitext(path)[1][1:]


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        chunk_number = int(os.getenv('DATASET_CHUNK_FIRST')) if os.getenv('DATASET_CHUNK_FIRST') else None
        dataset_base_result_dir = os.getenv('DATASET_RESULT_DIR')
        active = os.getenv('DATASET_DETECTION_ACTIVE', False)

        min_width = float(os.getenv('DETECTION_MIN_WIDTH_NORM', 0.02))
        min_height = float(os.getenv('DETECTION_MIN_HEIGHT_NORM', 0.04))
        test_percent = float(os.getenv('DATASET_TEST_PERCENT'))
        val_percent = float(os.getenv('DATASET_VAL_PERCENT'))
        video_percent = float(os.getenv('DATASET_VIDEO_PERCENT'))
        changed_percent = float(os.getenv('DATASET_CHANGED_PERCENT'))
        dataset_min_count = float(os.getenv('DATASET_MIN_COUNT'))
        dataset_first = os.getenv('DATASET_DETECTION_FIRST', False)
        train_percent = 1 - test_percent - val_percent

        vision_percent = 1 - video_percent
        unchanged_percent = 1 - changed_percent

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version_detect'].value) + 1

        capture_statuses = ['archived', 'verified']
        capture_version = None
        family_type = 'all'
        family_type_db = f'train_{family_type}'

        df = query_to_dataframe(
            'SELECT c.id'
            + f' ,c.source'
            + f' ,c.status'
            + f' ,c.photo_file'
            + f' ,c.date'
            + f' ,c.changed'
            + f' FROM detections_capture c'
            + f' WHERE c.status IN ("' + '","'.join(capture_statuses) + '")'
            + (f' AND {family_type_db} = False' if not dataset_first else '')
            + (f' AND c.version_detect = {capture_version}' if capture_version else '')
        )

        df['filename'] = df['photo_file'].apply(replace_extension)
        df['ext'] = df['photo_file'].apply(get_extension)
        df['image_path'] = './static/captures/' + df['status'] + '/images/' + df['photo_file']
        df['label_path'] = './static/captures/' + df['status'] + '/labels/' + df['photo_file'].apply(
            replace_extension,
            new_ext=".txt")

        df = df.sample(frac=1, random_state=42)

        df_video = df[df['source'] == 'video']
        df_vision = df[df['source'] == 'vision']
        df_changed = df_vision[df_vision['changed']]
        df_unchanged = df_vision[~df_vision['changed']]

        video_count = df_video.shape[0]
        vision_count = df_vision.shape[0]

        if video_percent > 0:
            vision_count = math.ceil(vision_percent * video_count / video_percent)

        if video_count + vision_count > df.shape[0]:
            print("/!\ Too many video items")
            vision_count = df_vision.shape[0]
            video_count = math.ceil(video_percent * vision_count / vision_percent)

        if video_count + vision_count > dataset_min_count:
            changed_count = min(df_changed.shape[0], math.ceil(vision_count * changed_percent))
            unchanged_count = min(
                math.ceil(unchanged_percent * changed_count / changed_percent),
                vision_count - changed_count
            )

            if video_count + changed_count + unchanged_count > df.shape[0]:
                print("/!\ Too many changed items")
                unchanged_count = min(df_unchanged.shape[0], math.ceil(vision_count * unchanged_percent))
                changed_count = min(
                    math.ceil(changed_percent * unchanged_count / unchanged_percent),
                    unchanged_count - unchanged_count
                )

            print("[All] Video count : ", video_count, '/', df_video.shape[0])
            print("[All] Vision count : ", vision_count, '/', df_vision.shape[0])
            print("[All] Changed count : ", changed_count, '/', df_changed.shape[0])
            print("[All] Unchanged count : ", unchanged_count, '/', df_unchanged.shape[0])
            print("[All] Vision (Un)changed count : ", changed_count + unchanged_count, '/', df_vision.shape[0])
            print("[All] Total : ", video_count + changed_count + unchanged_count, '/', df.shape[0])

            video_val_count = math.ceil(video_count * val_percent)
            vision_changed_val_count = math.ceil(changed_count * val_percent)
            vision_unchanged_val_count = math.ceil(unchanged_count * val_percent)

            video_test_count = math.ceil(video_count * test_percent)
            vision_changed_test_count = math.ceil(changed_count * test_percent)
            vision_unchanged_test_count = math.ceil(unchanged_count * test_percent)

            video_train_count = math.ceil(video_count * train_percent)
            vision_changed_train_count = math.ceil(changed_count * train_percent)
            vision_unchanged_train_count = math.ceil(unchanged_count * train_percent)

            df_val = pd.concat([
                df_video.iloc[:video_val_count],
                df_changed.iloc[:vision_changed_val_count],
                df_unchanged.iloc[:vision_unchanged_val_count],
            ]).reset_index(drop=True)
            df_val['type'] = 'val'

            df_test = pd.concat([
                df_video.iloc[video_val_count:video_val_count + video_test_count],
                df_changed.iloc[vision_changed_val_count:vision_changed_val_count + vision_changed_test_count],
                df_unchanged.iloc[vision_unchanged_val_count:vision_unchanged_val_count + vision_unchanged_test_count],
            ]).reset_index(drop=True)
            df_test['type'] = 'test'

            df_train = pd.concat([
                df_video.iloc[
                video_val_count + video_test_count:video_val_count + video_test_count + video_train_count],
                df_changed.iloc[
                vision_changed_val_count + vision_changed_test_count:vision_changed_val_count + vision_changed_test_count + vision_changed_train_count],
                df_unchanged.iloc[
                vision_unchanged_val_count + vision_unchanged_test_count:vision_unchanged_val_count + vision_unchanged_test_count + vision_unchanged_train_count],
            ]).reset_index(drop=True)
            df_train['type'] = 'train'

            df_all = pd.concat([df_val, df_test, df_train])

            print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=True))
            print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=False))
            # print("\n[ALL] Répartition des TYPES pour VIDEO dans:\n",
            #      df_all[df_all['source'] == 'video']['type'].value_counts(normalize=True))
            # print("\n[ALL] Répartition des TYPES pour VIDEO dans:\n",
            #      df_all[df_all['source'] == 'video']['type'].value_counts(normalize=False))
            # print("\n[ALL] Répartition des TYPES pour CHANGED dans:\n",
            #      df_all[df_all['changed']]['type'].value_counts(normalize=True))
            # print("\n[ALL] Répartition des TYPES pour CHANGED dans:\n",
            #      df_all[df_all['changed']]['type'].value_counts(normalize=False))
            # print("\n[ALL] Répartition des SOURCES dans:\n", df_all['source'].value_counts(normalize=True))
            # print("\n[ALL] Répartition des SOURCES dans:\n", df_all['source'].value_counts(normalize=False))
            # print("\n[ALL] Répartition des CHANGED dans:\n", df_all['changed'].value_counts(normalize=True))
            # print("\n[ALL] Répartition des CHANGED dans:\n", df_all['changed'].value_counts(normalize=False))

            print("\n[VAL] Répartition des SOURCES dans:\n", df_val['source'].value_counts(normalize=True))
            # print("\n[VAL] Répartition des SOURCES dans:\n", df_val['source'].value_counts(normalize=False))
            print("\n[VAL] Répartition des CHANGED dans:\n", df_val['changed'].value_counts(normalize=True))
            # print("\n[VAL] Répartition des CHANGED dans:\n", df_val['changed'].value_counts(normalize=False))

            print("\n[TEST] Répartition des SOURCES dans:\n", df_test['source'].value_counts(normalize=True))
            # print("\n[TEST] Répartition des SOURCES dans:\n", df_test['source'].value_counts(normalize=False))
            print("\n[TEST] Répartition des CHANGED dans:\n", df_test['changed'].value_counts(normalize=True))
            # print("\n[TEST] Répartition des CHANGED dans:\n", df_test['changed'].value_counts(normalize=False))

            print("\n[TRAIN] Répartition des SOURCES dans:\n", df_train['source'].value_counts(normalize=True))
            # print("\n[TRAIN] Répartition des SOURCES dans:\n", df_train['source'].value_counts(normalize=False))
            print("\n[TRAIN] Répartition des CHANGED dans:\n", df_train['changed'].value_counts(normalize=True))
            # print("\n[TRAIN] Répartition des CHANGED dans:\n", df_train['changed'].value_counts(normalize=False))

            dataset_dir = f'{dataset_base_result_dir}{chunk_number}-{family_type}'
            print(f"\n=> Save to {dataset_dir}\n")

            if active:
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                shutil.copy("./static/captures/data.yaml",
                            f"{dataset_dir}/data.yaml")

                copy_to(df_val, f'{dataset_dir}/val', min_width, min_height)
                copy_to(df_test, f'{dataset_dir}/test', min_width, min_height)
                copy_to(df_train, f'{dataset_dir}/train', min_width, min_height)

                if dataset_first:
                    Capture.objects.update(**{family_type_db: False})

                Capture.objects.filter(id__in=df_all['id'].to_list()) \
                    .update(**{family_type_db: True})

                self.stdout.write(
                    self.style.SUCCESS(
                        (
                            '[DEMO]' if not active else '') + f'[{chunk_number}] Successfully finished')
                )
        else:
            print("[All] Video count : ", video_count, '/', df_video.shape[0])
            print("[All] Vision count : ", vision_count, '/', df_vision.shape[0])
            self.stdout.write((
                                  '[DEMO]' if not active else '')
                              + f' Not enough items '
                              )
