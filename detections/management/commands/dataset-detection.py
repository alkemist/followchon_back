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


def repartir_echantillon(df, taille_cible, proportions_internes):
    echantillon = []
    for (source, changed), proportion in proportions_internes.items():
        df_filtre = df[(df['source'] == source) & (df['changed'] == changed)]
        taille_groupe = len(df_filtre)

        # Calcul de la taille de l'échantillon pour ce groupe, en tenant compte des proportions globales
        taille_echantillon_groupe = int(taille_cible * proportion)

        # Ajustement si le groupe est trop petit
        if taille_echantillon_groupe > taille_groupe:
            taille_echantillon_groupe = taille_groupe

        if taille_echantillon_groupe > 0:
            echantillon.append(df_filtre.sample(n=taille_echantillon_groupe, random_state=42))

    return pd.concat(echantillon)


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

        # Calcul des tailles des échantillons
        taille_val = int(len(df) * 0.2)
        taille_test = int(len(df) * 0.1)
        taille_train = len(df) - taille_val - taille_test

        # Proportions pour chaque DataFrame
        proportions_val = {
            ('video', True): 0.04,
            ('video', False): 0.04,
            ('vision', True): 0.04,
            ('vision', False): 0.04
        }

        proportions_test = {
            ('video', True): 0.02,
            ('video', False): 0.02,
            ('vision', True): 0.02,
            ('vision', False): 0.02
        }

        proportions_train = {
            ('video', True): 0.14,
            ('video', False): 0.14,
            ('vision', True): 0.14,
            ('vision', False): 0.14
        }

        # Création des DataFrames
        df_val = repartir_echantillon(df, taille_val, proportions_val)
        df_test = repartir_echantillon(df, taille_test, proportions_test)
        df_train = repartir_echantillon(df, taille_train, proportions_train)

        print("Taille DataFrame val:", len(df_val))
        print("Taille DataFrame test:", len(df_test))
        print("Taille DataFrame train:", len(df_train))

        print("\nRépartition des sources dans df_val:\n", df_val['source'].value_counts(normalize=True))
        print("\nRépartition des sources dans df_val:\n", df_val['source'].value_counts(normalize=False))
        print("\nRépartition des 'changed' dans df_val:\n", df_val['changed'].value_counts(normalize=True))
        print("\nRépartition des 'changed' dans df_val:\n", df_val['changed'].value_counts(normalize=False))

        print("\nRépartition des sources dans df_test:\n", df_test['source'].value_counts(normalize=True))
        print("\nRépartition des sources dans df_test:\n", df_test['source'].value_counts(normalize=False))
        print("\nRépartition des 'changed' dans df_test:\n", df_test['changed'].value_counts(normalize=True))
        print("\nRépartition des 'changed' dans df_test:\n", df_test['changed'].value_counts(normalize=False))

        print("\nRépartition des sources dans df_train:\n", df_train['source'].value_counts(normalize=True))
        print("\nRépartition des sources dans df_train:\n", df_train['source'].value_counts(normalize=False))
        print("\nRépartition des 'changed' dans df_train:\n", df_train['changed'].value_counts(normalize=True))
        print("\nRépartition des 'changed' dans df_train:\n", df_train['changed'].value_counts(normalize=False))

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
