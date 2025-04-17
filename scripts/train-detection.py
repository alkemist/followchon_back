import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.helpers import log_files_counts, train, log_version, export, parse, build, \
    metrics_dir, commit_files, purge_cache, end, models_dir

train_previous_path = os.getenv('TRAIN_DETECT_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_DETECT_DATASET_PATH')
train_name = os.getenv('TRAIN_DETECT_DATASET_NAME')

if __name__ == '__main__':
    train_classes = [0]
    classes_count = 6
    is_ok = False
    task = 'detect'

    log_version()
    log_files_counts(train_name, train_dataset_path,
                     ['train', 'val', 'test'],
                     ['labels'], "*.txt")

    is_ok = train(
        task,
        1024,
        train_previous_path,
        f"{train_dataset_path}/data.yaml",
        train_name,
        [
            ['confusion_matrix', 'png'],
            ['confusion_matrix_normalized', 'png'],
            ['F1_curve', 'png'],
            ['P_curve', 'png'],
            ['R_curve', 'png'],
            ['PR_curve', 'png'],
            ['results', 'csv'],
        ],
        train_classes
    )

    export(train_name, task)

    parse(train_name, classes_count)
    is_ok = build(train_name, train_dataset_path, classes_count) or is_ok

    model_pt = f"{models_dir}/{train_name}.pt"
    model_hef = f"{models_dir}/{train_name}.hef"
    best_hyperparameters_path = f'{models_dir}/{task}_best_hyperparameters.yaml'
    commit_files(train_name, [model_pt, model_hef, best_hyperparameters_path, f'{metrics_dir}/*'])

    purge_cache(train_dataset_path, ['train', 'val', 'test'], ['images'])

    end(is_ok)
