import os
from math import isnan, floor

import pandas as pd
from PIL import Image
from django.core.management.base import BaseCommand
from django.db import connection
from dotenv import load_dotenv

from configuration.models import Parameter
from detections.models import Capture
from utils.array import ArrayHelper


def query_to_dataframe(query):
    print(query)

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
        active = os.getenv('DATASET_ACTIVE', False)

        dataset_val_percent = float(os.getenv('DATASET_VAL_PERCENT'))
        dataset_test_percent = float(os.getenv('DATASET_TEST_PERCENT'))

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version_classify'].value) + 1

        family_indexes_all = [0, 1, 2]
        family_indexes = [1, 2]
        family_classes = ['guinea-pig', 'noisette', 'stitch']
        capture_statuses = ['archived', 'verified']
        capture_version = None
        family_type_db = 'train_chons'

        type_others_percent = 0.2
        type_chons_percent = 1 - type_others_percent
        dataset_min_count = 1000

        df_chons = query_to_dataframe(
            'SELECT c.id'
            + f' ,c.status'
            + f' ,c.photo_file'
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

        df_others = query_to_dataframe(
            'SELECT c.id'
            + f' ,c.status'
            + f' ,c.photo_file'
            + f' ,f."index"'
            + f' ,d.center_x'
            + f' ,d.center_y'
            + f' ,d.width'
            + f' ,d.height'
            + f' FROM detections_detection d'
            + f' LEFT JOIN detections_capture c ON d.capture_id = c.id'
            + f' LEFT JOIN configuration_family f ON d.family_id = f.id'
            + f' WHERE NOT EXISTS ('
            + f'    SELECT d2.id'
            + f'    FROM detections_detection d2'
            + f'    LEFT JOIN configuration_family f2 ON d2.family_id = f2.id'
            + f'    WHERE d2.capture_id = c.id'
            + f'    AND ('
            + f'    ' + ' OR '.join([f"f2.'index' = {i}" for i in family_indexes])
            + f'    )'
            + f' )'
            + f' AND f."index" = {family_indexes_all[0]}'
            + f' AND {family_type_db} = False'
            + f' AND c.status IN ("' + '","'.join(capture_statuses) + '")'
            + (f' AND c.version = {capture_version}' if capture_version else '')
        ).sample(frac=1, random_state=42).reset_index(drop=True)

        df_chons['image_path'] = './static/captures/' + df_chons['status'] + '/images/' + df_chons['photo_file']
        df_others['image_path'] = './static/captures/' + df_others['status'] + '/images/' + df_others['photo_file']
        df_chons['source'] = 'chons'
        df_others['source'] = 'others'

        df = pd.concat([
            df_chons,
            df_others,
        ]).reset_index(drop=True)

        if (df_chons.shape[0] > 0) & (df_others.shape[0] > 0):
            if df['source'].value_counts(normalize=True).loc['chons'] < type_chons_percent:
                df_others = df_others.sample(
                    n=int(df_chons.shape[0] * (1 - type_chons_percent) / type_chons_percent),
                    random_state=42
                )
            else:
                df_chons = df_chons.sample(
                    n=int(df_others.shape[0] * (1 - type_others_percent) / type_others_percent),
                    random_state=42
                )

            if (df_chons.shape[0] + df_others.shape[0]) > dataset_min_count:
                for i, family_index in enumerate(family_indexes_all):
                    df_family = df[df['index'] == family_index] \
                        .sample(frac=1, random_state=42) \
                        .reset_index(drop=True)

                    family_cls = family_classes[i]

                    val_count = floor(df_family.shape[0] * dataset_val_percent)
                    test_count = floor(df_family.shape[0] * dataset_test_percent)

                    dataset_dir = f'{dataset_base_result_dir}{chunk_number}-chons'

                    if not os.path.exists(dataset_dir):
                        os.makedirs(dataset_dir)

                    generate_dir(
                        df_family.iloc[:val_count],
                        dataset_dir,
                        'val',
                        family_cls
                    )

                    generate_dir(
                        df_family.iloc[val_count:val_count + test_count],
                        dataset_dir,
                        'test',
                        family_cls
                    )

                    generate_dir(
                        df_family.iloc[val_count + test_count:],
                        dataset_dir,
                        'train',
                        family_cls
                    )

                Capture.objects.filter(id__in=df_chons['id'].to_list()) \
                    .update(**{family_type_db: True})

                Capture.objects.filter(id__in=df_others['id'].to_list()) \
                    .update(**{family_type_db: True})

                self.stdout.write(
                    self.style.SUCCESS(
                        (
                            '[DEMO]' if not active else '') + f'[{chunk_number}] Successfully finished with {df.shape[0]} items')
                )

            else:
                self.stdout.write(
                    self.style.ERROR(
                        (
                            '[DEMO]' if not active else '') + f'Not enough items: chons {df_chons.shape[0]} / others {df_others.shape[0]} ')
                )
        else:
            self.stdout.write(
                self.style.ERROR(
                    (
                        '[DEMO]' if not active else '') + f'No items : chons {df_chons.shape[0]} / others {df_others.shape[0]}')
            )
