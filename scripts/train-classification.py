import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.helpers import log_files_counts, train, log_version, \
    metrics_dir, commit_files, purge_cache, end, models_dir

train_previous_path = os.getenv('TRAIN_CLASSIFY_MODEL_PATH')
train_dataset_path = os.getenv('TRAIN_CLASSIFY_DATASET_PATH')
train_name = os.getenv('TRAIN_CLASSIFY_DATASET_NAME')

if __name__ == '__main__':
    task = 'classify'

    log_version()
    log_files_counts(train_name, train_dataset_path,
                     ['train', 'val', 'test'],
                     ['noisette', 'sundae'], "*.*")

    is_ok = train(
        task,
        416,
        train_previous_path,
        train_dataset_path,
        train_name,
        [
            ['confusion_matrix', 'png'],
            ['confusion_matrix_normalized', 'png'],
            ['results', 'csv'],
        ],
    )

    model_pt = f"{models_dir}/{train_name}.pt"
    best_hyperparameters_path = f'{models_dir}/{task}_best_hyperparameters.yaml'
    commit_files(train_name, [model_pt, best_hyperparameters_path, f'{metrics_dir}/*'])

    purge_cache(train_dataset_path, ['train', 'val', 'test'], ['noisette', 'sundae'])

    end(os.getenv('TRAIN_CLASSIFY_SHUTDOWN'),is_ok)
