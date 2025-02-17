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
from utils.array import ArrayHelper


def extract(items, count):
    extracts = list()

    for i in range(count):
        choice = random.choice(items)
        extracts.append(pathlib.Path(choice).stem)

        items.remove(choice)

    return extracts


def copy_to(df, dist_dir):
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
        shutil.copy(
            str(df.loc[index, 'label_path']),
            f"{dist_dir}/labels/" + df.loc[index, 'filename'] + f".txt"
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
        active = os.getenv('DATASET_ACTIVE', False)

        dataset_test_percent = float(os.getenv('DATASET_TEST_PERCENT'))
        dataset_val_percent = float(os.getenv('DATASET_VAL_PERCENT'))

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version_detect'].value) + 1

        capture_statuses = ['archived', 'verified']
        capture_version = None
        family_type = 'all'
        family_indexes = [0]

        type_video_percent = 0.2
        type_vision_percent = 1 - type_video_percent
        dataset_min_count = 1000

        family_type_db = f'train_{family_type}'

        df = query_to_dataframe(
            'SELECT c.id'
            + f' ,c.source'
            + f' ,c.status'
            + f' ,c.photo_file'
            + f' ,c.date'
            + f' ,c.changed'
            + f' FROM detections_capture c'
            + f' WHERE {family_type_db} = False'
            + f' AND c.status IN ("' + '","'.join(capture_statuses) + '")'
            + (f' AND c.version_detect = {capture_version}' if capture_version else '')
        )

        df['filename'] = df['photo_file'].apply(replace_extension)
        df['ext'] = df['photo_file'].apply(get_extension)
        df['image_path'] = './static/captures/' + df['status'] + '/images/' + df['photo_file']
        df['label_path'] = './static/captures/' + df['status'] + '/labels/' + df['photo_file'].apply(
            replace_extension,
            new_ext=".txt")

        df = df.sample(frac=1, random_state=42)  # random_state pour la reproductibilité

        total_size = len(df)
        val_size = int(0.2 * total_size)
        test_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size

        # Fonction pour créer un sous-ensemble avec les contraintes spécifiées
        def create_subset(df, size, video_percentage=0.2, changed_false_percentage=0.2):
            video_size = int(size * video_percentage)
            changed_false_size = int(size * changed_false_percentage)

            # Sélectionner les lignes avec source = 'video'
            video_rows = df[df['source'] == 'video']
            selected_video_rows = video_rows.sample(n=min(video_size, len(video_rows)), random_state=42)

            # Sélectionner les lignes avec changed = False
            changed_false_rows = df[df['changed'] == False]
            selected_changed_false_rows = changed_false_rows.sample(n=min(changed_false_size, len(changed_false_rows)),
                                                                    random_state=42)

            # Sélectionner le reste des lignes
            remaining_rows = df.drop(selected_video_rows.index).drop(selected_changed_false_rows.index)
            selected_remaining_rows = remaining_rows.sample(
                n=size - len(selected_video_rows) - len(selected_changed_false_rows), random_state=42)

            # Combiner les lignes sélectionnées
            subset = pd.concat([selected_video_rows, selected_changed_false_rows, selected_remaining_rows])
            return subset

        # Créer les sous-ensembles val, test et train
        df_val = create_subset(df, val_size)
        df_test = create_subset(df.drop(df_val.index), test_size)
        df_train = create_subset(df.drop(df_val.index).drop(df_test.index), train_size)

        df_val['type'] = 'val'
        df_test['type'] = 'test'
        df_train['type'] = 'train'

        df_all = pd.concat([df_train, df_val, df_test])

        print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=True))
        print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=False))
        print("\n[ALL] Répartition des SOURCES dans:\n", df_all['source'].value_counts(normalize=True))
        print("\n[ALL] Répartition des SOURCES dans:\n", df_all['source'].value_counts(normalize=False))
        print("\n[ALL] Répartition des CHANGED dans:\n", df_all['changed'].value_counts(normalize=True))
        print("\n[ALL] Répartition des CHANGED dans:\n", df_all['changed'].value_counts(normalize=False))

        print("\n[VAL] Répartition des SOURCES dans:\n", df_val['source'].value_counts(normalize=True))
        print("\n[VAL] Répartition des SOURCES dans:\n", df_val['source'].value_counts(normalize=False))
        print("\n[VAL] Répartition des CHANGED dans:\n", df_val['changed'].value_counts(normalize=True))
        print("\n[VAL] Répartition des CHANGED dans:\n", df_val['changed'].value_counts(normalize=False))

        print("\n[TEST] Répartition des SOURCES dans:\n", df_test['source'].value_counts(normalize=True))
        print("\n[TEST] Répartition des SOURCES dans:\n", df_test['source'].value_counts(normalize=False))
        print("\n[TEST] Répartition des CHANGED dans:\n", df_test['changed'].value_counts(normalize=True))
        print("\n[TEST] Répartition des CHANGED dans:\n", df_test['changed'].value_counts(normalize=False))

        print("\n[TRAIN] Répartition des SOURCES dans:\n", df_train['source'].value_counts(normalize=True))
        print("\n[TRAIN] Répartition des SOURCES dans:\n", df_train['source'].value_counts(normalize=False))
        print("\n[TRAIN] Répartition des CHANGED dans:\n", df_train['changed'].value_counts(normalize=True))
        print("\n[TRAIN] Répartition des CHANGED dans:\n", df_train['changed'].value_counts(normalize=False))

        # df_vision = df[(df['source'] == 'vision') & (df['changed'])].sample(frac=1, random_state=42) \
        #     .reset_index(drop=True)
        # df_video = df[df['source'] == 'video'] \
        #     .sample(frac=1, random_state=42).reset_index(drop=True)
        #
        # if (df_vision.shape[0] > 0) & (df_video.shape[0] > 0):
        #     if df['source'].value_counts(normalize=True).loc['video'] < type_video_percent:
        #         df_vision = df_vision.sample(
        #             n=int(df_video.shape[0] * (1 - type_video_percent) / type_video_percent),
        #             random_state=42
        #         )
        #     else:
        #         df_video = df_video.sample(
        #             n=int(df_vision.shape[0] * (1 - type_vision_percent) / type_vision_percent),
        #             random_state=42
        #         )
        #
        #     if (df_video.shape[0] + df_vision.shape[0]) > dataset_min_count:
        #         video_val_count = floor(df_video.shape[0] * dataset_val_percent)
        #         vision_val_count = floor(df_vision.shape[0] * dataset_val_percent)
        #         video_test_count = floor(df_video.shape[0] * dataset_test_percent)
        #         vision_test_count = floor(df_vision.shape[0] * dataset_test_percent)
        #
        #         df_val = pd.concat([
        #             df_vision.iloc[:vision_val_count],
        #             df_video.iloc[:video_val_count],
        #         ]).reset_index(drop=True)
        #         df_val['type'] = 'val'
        #
        #         df_test = pd.concat([
        #             df_vision.iloc[vision_val_count:vision_val_count + vision_test_count],
        #             df_video.iloc[video_val_count:video_val_count + video_test_count],
        #         ]).reset_index(drop=True)
        #         df_test['type'] = 'test'
        #
        #         df_train = pd.concat([
        #             df_vision.iloc[vision_val_count + vision_test_count:],
        #             df_video.iloc[video_val_count + video_test_count:],
        #         ]).reset_index(drop=True)
        #         df_train['type'] = 'train'
        #
        #         dataset_dir = f'{dataset_base_result_dir}{chunk_number}-{family_type}'
        #
        #         if active:
        #             if not os.path.exists(dataset_dir):
        #                 os.makedirs(dataset_dir)
        #
        #             shutil.copy("./static/captures/data.yaml",
        #                         f"{dataset_dir}/data.yaml")
        #
        #             copy_to(df_val, f'{dataset_dir}/val')
        #             copy_to(df_test, f'{dataset_dir}/test')
        #             copy_to(df_train, f'{dataset_dir}/train')
        #
        #             capture_ids = df_vision['id'].to_list() + df_video['id'].to_list()
        #             Capture.objects.filter(id__in=capture_ids) \
        #                 .update(**{family_type_db: True})
        #
        #         print('TRAIN : ')
        #         print(df_train['source'].value_counts(normalize=True))
        #         print(df_train['changed'].value_counts(normalize=True))
        #         print('VAL : ')
        #         print(df_val['source'].value_counts(normalize=True))
        #         print(df_val['changed'].value_counts(normalize=True))
        #         print('TEST : ')
        #         print(df_test['source'].value_counts(normalize=True))
        #         print(df_test['changed'].value_counts(normalize=True))
        #         print('ALL : ')
        #         print(pd.concat([df_train, df_val, df_test])['type'].value_counts(normalize=True))
        #         print(pd.concat([df_train, df_val, df_test])['type'].value_counts(normalize=False))
        #         print(pd.concat([df_train, df_val, df_test])['source'].value_counts(normalize=True))
        #         print(pd.concat([df_train, df_val, df_test])['source'].value_counts(normalize=False))
        #         print(pd.concat([df_train, df_val, df_test])['changed'].value_counts(normalize=True))
        #         print(pd.concat([df_train, df_val, df_test])['changed'].value_counts(normalize=False))
        #
        #         self.stdout.write(
        #             self.style.SUCCESS(
        #                 (
        #                     '[DEMO]' if not active else '') + f'[{chunk_number}] Successfully finished with {df_vision.shape[0] + df_video.shape[0]} items')
        #         )
        #     else:
        #         print(df['source'].value_counts(normalize=False))
        #         print(df['changed'].value_counts(normalize=False))
        #         self.stdout.write(
        #             self.style.ERROR(
        #                 (
        #                     '[DEMO]' if not active else '') + f'Not enough items: [vision] {df_vision.shape[0]} / [video] {df_video.shape[0]} ')
        #         )
        # else:
        #     self.stdout.write(
        #         self.style.ERROR(('[DEMO]' if not active else '') + 'No "video" items ')
        #     )
