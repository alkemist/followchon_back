import os
import pathlib
import random
import shutil
from math import isnan

import numpy as np
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


def create_stratified_split(df, source_ratio=0.2, changed_ratio=0.8, train_size=0.7, val_size=0.2, test_size=0.1):
    """
    Crée des splits stratifiés d'un DataFrame avec des ratios spécifiques pour les colonnes 'source' et 'changed'.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame d'entrée avec les colonnes 'source' et 'changed'
    source_ratio : float
        Proportion désirée de 'source' = 'video' (par défaut 0.2)
    changed_ratio : float
        Proportion désirée de 'changed' = True (par défaut 0.8)
    train_size : float
        Proportion pour le set d'entraînement (par défaut 0.7)
    val_size : float
        Proportion pour le set de validation (par défaut 0.2)
    test_size : float
        Proportion pour le set de test (par défaut 0.1)

    Returns:
    --------
    tuple(pandas.DataFrame)
        df_sub, df_train, df_val, df_test
    """
    # Vérification que les proportions somment à 1
    assert np.isclose(train_size + val_size + test_size, 1.0)

    # Création de df_sub avec les ratios souhaités
    # Sélection pour source
    video_samples = df[df['source'] == 'video']
    vision_samples = df[df['source'] == 'vision']

    total_desired_size = len(df)
    video_size = int(total_desired_size * source_ratio)
    vision_size = total_desired_size - video_size

    video_sub = video_samples.sample(n=min(video_size, len(video_samples)))
    vision_sub = vision_samples.sample(n=min(vision_size, len(vision_samples)))

    df_sub = pd.concat([video_sub, vision_sub])

    # Ajustement pour changed
    true_samples = df_sub[df_sub['changed'] == True]
    false_samples = df_sub[df_sub['changed'] == False]

    true_size = int(len(df_sub) * changed_ratio)
    false_size = len(df_sub) - true_size

    true_sub = true_samples.sample(n=min(true_size, len(true_samples)))
    false_sub = false_samples.sample(n=min(false_size, len(false_samples)))

    df_sub = pd.concat([true_sub, false_sub])

    # Création des splits train/val/test avec les mêmes ratios
    def create_split(data, size):
        total_size = int(len(df_sub) * size)
        video_size = int(total_size * source_ratio)
        true_size = int(total_size * changed_ratio)

        # Split par source
        split_video = data[data['source'] == 'video'].sample(n=video_size)
        split_vision = data[data['source'] == 'vision'].sample(n=total_size - video_size)

        # Combiner et ajuster pour changed
        split_combined = pd.concat([split_video, split_vision])
        split_true = split_combined[split_combined['changed'] == True].sample(n=true_size)
        split_false = split_combined[split_combined['changed'] == False].sample(n=total_size - true_size)

        return pd.concat([split_true, split_false])

    # Création des différents sets
    df_train = create_split(df_sub, train_size)
    remaining = df_sub.drop(df_train.index)

    # Ajuster les proportions pour les ensembles restants
    remaining_ratio = val_size / (val_size + test_size)
    df_val = create_split(remaining, remaining_ratio)
    df_test = df_sub.drop(pd.concat([df_train, df_val]).index)

    return df_sub, df_train, df_val, df_test


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
        df['type'] = ''

        df = df.sample(frac=1, random_state=42)  # random_state pour la reproductibilité

        df_sub, df_train, df_val, df_test = create_stratified_split(
            df,
            source_ratio=0.2,  # 20% video, 80% vision
            changed_ratio=0.8,  # 80% True, 20% False
            train_size=0.7,  # 70% pour l'entraînement
            val_size=0.2,  # 20% pour la validation
            test_size=0.1  # 10% pour le test
        )

        df_all = pd.concat([df_train, df_val, df_test])
        df_val['type'] = 'val'
        df_test['type'] = 'test'
        df_train['type'] = 'train'

        # df_all = df  # pd.concat([df_train, df_val, df_test])
        # df_vision = df_all[df_all['source'] == 'vision']
        # df_video = df_all[df_all['source'] == 'video']
        # df_changed = df_vision[df_vision['changed']]
        # df_unchanged = df_vision[~df_vision['changed']]

        print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=True))
        print("\n[ALL] Répartition des TYPES dans:\n", df_all['type'].value_counts(normalize=False))
        print("\n[ALL] Répartition des TYPES pour VIDEO dans:\n", df_all['type'].value_counts(normalize=True))
        print("\n[ALL] Répartition des TYPES pour VIDEO dans:\n", df_all['type'].value_counts(normalize=False))
        print("\n[ALL] Répartition des TYPES pour CHANGED dans:\n", df_all['type'].value_counts(normalize=True))
        print("\n[ALL] Répartition des TYPES pour CHANGED dans:\n", df_all['type'].value_counts(normalize=False))
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
