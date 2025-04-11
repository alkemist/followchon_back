from dotenv import load_dotenv

from scripts.helpers import log_files_counts, tune, train, move_metrics, log_version, export, parse, build, model_pt, \
    model_hef, metrics_dir, commit_files, purge_cache, end

load_dotenv()

if __name__ == '__main__':
    train_classes = [0]
    classes_count = 6

    log_version()
    log_files_counts(['train', 'val', 'test'], ['labels'], "*.txt")

    model = tune()

    train(
        model,
        'detect',
        416,
        train_classes
    )

    move_metrics([
        ['confusion_matrix', 'png'],
        ['confusion_matrix_normalized', 'png'],
        ['F1_curve', 'png'],
        ['P_curve', 'png'],
        ['R_curve', 'png'],
        ['PR_curve', 'png'],
        ['results', 'csv'],
    ])

    export()

    parse(classes_count)
    build(classes_count)

    commit_files([model_pt, model_hef, f'{metrics_dir}/*'])

    purge_cache(['train', 'val', 'test'], ['images'])

    end()
