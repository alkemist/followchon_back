from dotenv import load_dotenv

from scripts.helpers import log_files_counts, tune, train, move_metrics, log_version, model_pt, \
    metrics_dir, commit_files, purge_cache, end

load_dotenv()

if __name__ == '__main__':
    log_version()
    log_files_counts(['train', 'val', 'test'], ['noisette', 'sundae'], "*.*")

    model = tune()

    train(
        model,
        'classify',
        416
    )

    move_metrics([
        ['confusion_matrix', 'png'],
        ['confusion_matrix_normalized', 'png'],
        ['results', 'csv'],
    ])

    commit_files([model_pt, f'{metrics_dir}/*'])

    purge_cache(['train', 'val', 'test'], ['noisette', 'sundae'])

    end()
