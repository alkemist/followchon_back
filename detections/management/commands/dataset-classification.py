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

        dataset_val_percent = float(os.getenv('DATASET_VAL_PERCENT'))
        dataset_test_percent = float(os.getenv('DATASET_TEST_PERCENT'))

        if not chunk_number or chunk_number == 0 or isnan(chunk_number):
            parameters = Parameter.objects.all()
            params_dict: dict[str, Parameter] = (
                ArrayHelper.object_list_to_dict(parameters, 'slug')
            )
            chunk_number = int(params_dict['vision_model_version'].value) + 1

        family_indexes = [1, 2]
        family_classes = ['noisette', 'stitch']
        capture_statuses = ['archived', 'verifed']
        capture_version = None
        family_type_db = 'train_chons'

        df = query_to_dataframe(
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
            + f' ORDER BY c.photo_file'
        )

        df['image_path'] = './static/captures/' + df['status'] + '/images/' + df['photo_file']

        for i, family_index in enumerate(family_indexes):
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

        Capture.objects.filter(id__in=df['id'].to_list()) \
            .update(**{family_type_db: True})
