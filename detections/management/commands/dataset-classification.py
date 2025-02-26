import math
import os
from math import isnan

import pandas as pd
from PIL import Image
from django.core.management.base import BaseCommand
from django.db import connection
from dotenv import load_dotenv

from configuration.models import Parameter
from detections.models import Capture
from utils.array import ArrayHelper


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


def crop_and_save(image_path, output_path, center_x_norm, center_y_norm, width_norm, height_norm):
    # Ouvrir l'image
    image = Image.open(image_path)
    img_width, img_height = image.size

    # Convertir les coordonnées normalisées en pixels
    center_x = int(center_x_norm * img_width)
    center_y = int(center_y_norm * img_height)
    width = int(width_norm * img_width)
    height = int(height_norm * img_height)

    # Calculer les bords de la boîte de découpe
    left = center_x - width // 2
    top = center_y - height // 2
    right = center_x + width // 2
    bottom = center_y + height // 2

    # Découper l'image
    crop_box = (left, top, right, bottom)

    if right - top > 10 and bottom - top > 10:
        cropped_image = image.crop(crop_box)

        # Sauvegarder l'image découpée
        cropped_image.save(output_path)


def generate_dir(df, basedir, subdir, cls):
    if not os.path.exists(f"{basedir}/{subdir}"):
        os.makedirs(f"{basedir}/{subdir}")
    if not os.path.exists(f"{basedir}/{subdir}/{cls}"):
        os.makedirs(f"{basedir}/{subdir}/{cls}")

    for index in df.index:
        image_path = df.loc[index, 'image_path']
        output_path = f"{basedir}/{subdir}/{cls}/{cls}_" + df.loc[index, 'photo_file']

        crop_and_save(
            image_path,
            output_path,
            df.loc[index, 'center_x'],
            df.loc[index, 'center_y'],
            df.loc[index, 'width'],
            df.loc[index, 'height'],
        )


def get_extension(path):
    return os.path.splitext(path)[1][1:]


class Command(BaseCommand):
    help = ""

    def handle(self, *args, **options):
        load_dotenv()

        chunk_number = int(os.getenv('DATASET_CHUNK_FIRST')) if os.getenv('DATASET_CHUNK_FIRST') else None
        dataset_base_result_dir = os.getenv('DATASET_RESULT_DIR')
        dataset_min_count = float(os.getenv('DATASET_MIN_COUNT'))
        changed_percent = float(os.getenv('DATASET_CHANGED_PERCENT'))
        active = os.getenv('DATASET_ACTIVE', False)

        val_percent = float(os.getenv('DATASET_VAL_PERCENT'))
        test_percent = float(os.getenv('DATASET_TEST_PERCENT'))
        train_percent = 1 - test_percent - val_percent
        unchanged_percent = 1 - changed_percent

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version_classify'].value) + 1

        family_indexes = [1, 6]
        family_classes = ['noisette', 'sundae']
        capture_statuses = ['archived', 'verified']
        capture_version = None
        family_type_db = 'train_chons'

        df = query_to_dataframe(
            'SELECT c.id'
            + f' ,c.status'
            + f' ,c.photo_file'
            + f' ,c.changed'
            + f' ,f."index"'
            + f' ,d.center_x'
            + f' ,d.center_y'
            + f' ,d.width'
            + f' ,d.height'
            + f' FROM detections_detection d'
            + f' LEFT JOIN detections_capture c ON d.capture_id = c.id'
            + f' LEFT JOIN configuration_family f ON d.family_id = f.id'
            + f' WHERE c.status IN ("' + '","'.join(capture_statuses) + '")'
            + f' AND {family_type_db} = False'
            + f' AND ('
            + f' ' + ' OR '.join([f"f.'index' = {i}" for i in family_indexes])
            + f' )'
            + (f' AND c.version = {capture_version}' if capture_version else '')
        ).sample(frac=1, random_state=42).reset_index(drop=True)

        df['image_path'] = './static/captures/' + df['status'] + '/images/' + df['photo_file']

        df_changed = df[df['changed']]
        df_unchanged = df[~df['changed']]

        df = df.sample(frac=1, random_state=42)

        counts_changed = []
        counts_unchanged = []

        for index in family_indexes:
            counts_changed.append(df_changed[df_changed['index'] == index].shape[0])
            counts_unchanged.append(df_unchanged[df_unchanged['index'] == index].shape[0])

        changed_min_count = min(counts_changed)
        unchanged_min_count = min(counts_unchanged)

        print("Min count changed : ", changed_min_count, '/', counts_changed)
        print("Min count unchanged : ", unchanged_min_count, '/', counts_unchanged)

        changed_count = changed_min_count
        unchanged_count = math.ceil(unchanged_percent * changed_min_count / changed_percent)

        if unchanged_count > unchanged_min_count:
            unchanged_count = unchanged_min_count
            changed_count = math.ceil(changed_percent * unchanged_count / unchanged_percent),

        print("Count changed/unchanged : ", changed_count, '/', unchanged_count)

        val_changed_count = math.ceil(changed_count * val_percent)
        val_unchanged_count = math.ceil(unchanged_count * val_percent)
        test_changed_count = math.ceil(changed_count * test_percent)
        test_unchanged_count = math.ceil(unchanged_count * test_percent)
        train_changed_count = math.ceil(changed_count * train_percent)
        train_unchanged_count = math.ceil(unchanged_count * train_percent)

        for i, family_index in enumerate(family_indexes):
            df_family = df[df['index'] == family_index] \
                .sample(frac=1, random_state=42) \
                .reset_index(drop=True)

            df_family_changed = df_family[df_family['changed']]
            df_family_unchanged = df_family[~df_family['changed']]

            family_cls = family_classes[i]

            dataset_dir = f'{dataset_base_result_dir}{chunk_number}-chons'

            df_val = pd.concat([
                df_family_changed.iloc[:val_changed_count],
                df_family_unchanged.iloc[:val_unchanged_count],
            ]).reset_index(drop=True)
            df_val['type'] = 'val'

            df_test = pd.concat([
                df_family_changed.iloc[val_changed_count:val_changed_count + test_changed_count],
                df_family_unchanged.iloc[val_unchanged_count:val_unchanged_count + test_unchanged_count],
            ]).reset_index(drop=True)
            df_test['type'] = 'test'

            df_train = pd.concat([
                df_family_changed.iloc[
                val_changed_count + test_changed_count:val_changed_count + test_changed_count + train_changed_count],
                df_family_unchanged.iloc[
                val_unchanged_count + test_unchanged_count:val_unchanged_count + test_unchanged_count + train_unchanged_count],
            ]).reset_index(drop=True)
            df_train['type'] = 'train'

            df_all = pd.concat([df_val, df_test, df_train])

            print(f"\n[{family_cls}] {df_all.shape[0]} items in {dataset_dir}")

            print("\nRépartition des TYPES dans:\n", df_all['type'].value_counts(normalize=True))
            print("\nRépartition des TYPES dans:\n", df_all['type'].value_counts(normalize=False))

            print("\n[VAL] Répartition des CHANGED dans:\n", df_val['changed'].value_counts(normalize=True))
            print("\n[TEST] Répartition des CHANGED dans:\n", df_test['changed'].value_counts(normalize=True))
            print("\n[TRAIN] Répartition des CHANGED dans:\n", df_train['changed'].value_counts(normalize=True))

            if active:
                if not os.path.exists(dataset_dir):
                    os.makedirs(dataset_dir)

                generate_dir(
                    df_val,
                    dataset_dir,
                    'val',
                    family_cls
                )

                generate_dir(
                    df_test,
                    dataset_dir,
                    'test',
                    family_cls
                )

                generate_dir(
                    df_train,
                    dataset_dir,
                    'train',
                    family_cls
                )

                Capture.objects.filter(id__in=df_all['id'].to_list()) \
                    .update(**{family_type_db: True})

        if active:
            self.stdout.write(
                self.style.SUCCESS(
                    ('[DEMO]' if not active else '') + f'[{chunk_number}] Successfully finished')
            )
