# venv/bin/python -m src.vision
import os

from .models.streamer import Streamer
from dotenv import load_dotenv

load_dotenv()

check_all_records = os.getenv('CHECK_ALL_RECORDS') == 'True'
verbose = os.getenv('VERBOSE') == 'True'
save_enabled = os.getenv('SAVE_ENABLED') == 'True'
track_enabled = os.getenv('TRACK_ENABLED') == 'True'
loop_enabled = os.getenv('LOOP_ENABLED') == 'True'
delete_record = os.getenv('DELETE_RECORD') == 'True'
show_stream = os.getenv('SHOW_STREAM') == 'True'
frame_time_seconds = float(os.getenv('FRAME_TIME_SECONDS'))  # 0.03 < > 0.02
capture_min_conf = float(os.getenv('CAPTURE_MIN_CONF'))  # 0.03 < > 0.02

streamer = Streamer(
    os.getenv('LIVE_STREAM_PATH'),
    os.getenv('MODEL_PATH'),
    './live/records',
    './live/captures',
    capture_width=1024,
    capture_height=768,
    frame_time_seconds=0.025,  # 0.03 < > 0.02
    capture_min_conf=capture_min_conf,
    check_all_records=check_all_records,
    show_stream=show_stream,
    verbose=verbose,
    save_enabled=save_enabled,
    track_enabled=save_enabled,
    loop_enabled=loop_enabled,
    delete_record=delete_record,
)

streamer.start()
