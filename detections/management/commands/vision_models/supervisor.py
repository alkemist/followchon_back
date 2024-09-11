import os
import statistics
import time
from datetime import datetime
from math import floor

import psutil
from loguru import logger

from configuration.models import Parameter, Log
from helpers.array import ArrayHelper
from helpers.date import DateHelper
from helpers.file import FileHelper


class Supervisor:

    def __init__(self, vcgencmd=None):
        self.records_directory = './records'

        self.current_model_version = 0
        self.model_version = ''
        self.model_path = ''
        self.score_min = 0
        self.hour_min = 0
        self.hour_max = 0
        self.records_max = 0
        self.temp_alert = 0
        self.pause_records_minutes = 0
        self.pause_capture_seconds = 0
        self.temp_pause = 0
        self.frame_seconds = 0
        self.popcorn_frame_count = 0
        self.stats_hourly = False
        self.enabled = True
        self.params_dict = {}

        self.detection_near_margin_norm = 1
        self.detection_move_margin_norm = 1

        self.records_count = 0
        self.record_time = 60
        self.record_time_delay = 50
        self.min_records_capture = 1
        self.min_records_recording = 2
        self.is_recording = False
        self.last_frame_seconds = 0
        self.last_capture_seconds = time.time()

        self.vcgm = vcgencmd
        self.temp = 0
        self.temps: dict[str, float] = {}

        self.delay_start = 0
        self.delays = []

        self.analyses_by_record = []

        self.check_disk_free()

    def get_model_path(self):
        return (f"{os.getenv('MODEL_DIR')}/"
                f"{os.getenv('MODEL_PREFIX')}{self.current_model_version}.{os.getenv('MODEL_EXT')}")

    def get_params(self):
        parameters = Parameter.objects.all()
        self.params_dict: dict[int, Parameter] = (
            ArrayHelper.object_list_to_dict(parameters, 'slug')
        )

    def get_param(self, param: str):
        if param in self.params_dict:
            return self.params_dict[param].value

        logger.error(f'Param "{param}" not exist')
        return None

    def get_records(self):
        records = FileHelper.list_files(self.records_directory, r'.*\.(mkv)$')
        self.records_count = len(records)
        return records

    def get_records_count(self):
        self.get_records()
        return self.records_count

    def add_temperature(self, force: bool = False):
        hour = datetime.now().hour
        minute = floor(datetime.now().minute / 10)

        key = f"{hour}:{minute}"

        self.temp = self.vcgm.measure_temp() if self.vcgm is not None else 0

        if force or key not in self.temps:
            self.temps[key] = round(self.temp, 1)

    def check_disk_free(self):
        disk_info = psutil.disk_usage(os.getenv('WATCH_DISK'))

        if disk_info.percent >= 80:
            self.log(
                'Disk',
                self.get_log_time_ave() +
                f"disk_used={FileHelper.convert_size(disk_info.used)} "
                f"disk_free={FileHelper.convert_size(disk_info.free)} "
                f"disk_usage={disk_info.percent}% "
                , 'warning'
            )

    def add_processing_delay(self):
        self.delays.append(
            round(
                time.time() - self.delay_start
            )
        )

    def is_recording_ok(self):
        return (
                self.hour_min <= datetime.now().hour < self.hour_max and
                (self.is_recording and self.records_count <= self.records_max or
                 not self.is_recording and self.records_count <= self.min_records_recording)
        )

    def begin_recording(self):
        self.is_recording = True
        self.last_capture_seconds = time.time()

    def stop_recording(self):
        self.is_recording = False

    def fill_params(self):
        self.get_params()

        self.model_version = int(self.get_param('vision_model_version'))
        self.score_min = float(self.get_param('vision_score_min'))
        self.hour_min = int(self.get_param('vision_hour_min'))
        self.hour_max = int(self.get_param('vision_hour_max'))
        self.records_max = int(self.get_param('vision_records_max'))
        self.pause_records_minutes = int(self.get_param('vision_pause_records_minutes'))
        self.pause_capture_seconds = float(self.get_param('vision_pause_capture_seconds'))
        self.temp_alert = int(self.get_param('vision_temp_alert'))
        self.temp_pause = int(self.get_param('vision_temp_pause_minutes'))
        self.popcorn_frame_count = int(self.get_param('vision_popcorn_frame_count'))
        self.frame_seconds = float(self.get_param('vision_frame_seconds'))  # 0.03 < > 0.02
        self.enabled = self.get_param('vision_enabled') == '1'
        self.stats_hourly = self.get_param('vision_stats_hourly') == '1'

        self.detection_near_margin_norm = float(self.get_param('vision_detection_near_margin_norm'))
        self.detection_move_margin_norm = float(self.get_param('vision_detection_move_margin_norm'))

        self.temp = self.vcgm.measure_temp() if self.vcgm is not None else 0

    def local_log(self, event: str, info: str, level: str = 'info'):
        message = f"{event} : {info} temp={self.temp}°"

        match level:
            case 'warning':
                logger.warning(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
            case 'error':
                logger.error(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
            case _:
                logger.info(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    def log(self, event: str, info: str = '', level: str = 'info'):
        self.local_log(event, info, level)

        Log().create(self.current_model_version, 'vision', level, event, info, self.temp)

    def get_log_records(self):
        return f"records={self.records_count}/{self.records_max} "

    def get_log_hour(self):
        return f"hour={datetime.now().hour}h/{self.hour_min}h-{self.hour_max}h "

    def get_log_pause_records(self):
        return f"pause_records={self.pause_records_minutes}m "

    def get_log_disk_free(self):
        disk_info = psutil.disk_usage(os.getenv('WATCH_DISK'))
        return f"disk_usage={disk_info.percent}% "

    def get_log_fps(self):
        return (f"frame_seconds={self.frame_seconds}s "
                + f"pause_capture={self.pause_capture_seconds}s "
                + self.get_log_pause_records())

    def get_log_time_ave(self):
        return f"time_ave={DateHelper.secondsToMMSS(statistics.fmean(self.delays))} " if len(self.delays) > 0 else ""

    def get_log_temp_ave(self):
        return f"temp_ave={round(statistics.fmean(self.temps.values()), 2)}° " if len(self.temps.values()) > 0 else ""

    def get_log_fpm_ave(self):
        return f"fpm_ave={round(statistics.fmean(self.analyses_by_record))} " if len(
            self.analyses_by_record) > 0 else ""

    def get_log_enabled(self):
        return f"enabled={self.enabled} "

    def get_log_temp_pause(self):
        return f"enabled={self.temp_pause} "

    def get_log_delay(self, capture_time_elapsed: int):
        return f"capture_time_elapsed={capture_time_elapsed}/{self.record_time + self.record_time_delay} " if capture_time_elapsed else ""

    def log_start(self):
        self.records_count = self.get_records_count()
        self.log(
            'Started',
            self.get_log_records()
        )

    def log_start_recording(self):
        self.log(
            'Start recording',
            self.get_log_records()
            + self.get_log_hour()
        )

    def log_warning_temperature(self):
        self.log(
            'Temperature',
            self.get_log_records()
            + self.get_log_fps()
            + self.get_log_temp_pause()
            , "warning"
        )

    def log_stop_recording(self):
        self.log(
            'Stop recording',
            self.get_log_records()
        )

    def log_sleeping(self):
        self.local_log(
            'Sleeping',
            self.get_log_records()
            + self.get_log_fps()
            + self.get_log_time_ave()
        )

    def log_awakened(self):
        self.local_log(
            'Awakened',
            self.get_log_records()
            + self.get_log_fps()
        )

    def log_restart_recording(self, capture_time_elapsed: int | None = None):
        self.log(
            'Restart recording',
            self.get_log_records()
            + self.get_log_pause_records()
            + self.get_log_delay(capture_time_elapsed)
            + self.get_log_time_ave()
            , 'warning'
        )

    def log_waiting(self):
        self.local_log(
            'Waiting',
            self.get_log_records()
            + self.get_log_pause_records()
        )

    def log_stopped(self):
        self.log(
            'Stopped',
            self.get_log_records()
            + self.get_log_fps()
            + self.get_log_disk_free()
            + self.get_log_hour()
        )

    def log_popcorn(self, capture_date, frame_count):
        self.log(
            'Popcorn',
            f"date={capture_date.strftime('%Y-%m-%d %H-%M')} "
            f"count={frame_count} "
        )

    def log_stat_processing(self):
        if len(self.delays) > 0:
            self.log(
                'Processing',
                self.get_log_time_ave() +
                f"time_med={DateHelper.secondsToMMSS(statistics.median(self.delays))} "
                f"time_min={DateHelper.secondsToMMSS(min(self.delays))} "
                f"time_max={DateHelper.secondsToMMSS(max(self.delays))} "
                , 'statistic'
            )

    def log_stat_fpm(self):
        if len(self.analyses_by_record) > 0:
            self.log(
                'Analyses',
                self.get_log_fpm_ave() +
                f"fpm_med={statistics.median(self.analyses_by_record)} "
                f"fpm_min={min(self.analyses_by_record)} "
                f"fpm_max={max(self.analyses_by_record)} "
                , 'statistic'
            )

    def log_stat_temperature(self):
        if len(self.temps.values()) > 0:
            self.log(
                'Temperature',
                self.get_log_temp_ave() +
                f"temp_med={statistics.median(self.temps.values())}° "
                f"temp_min={min(self.temps.values())}° "
                f"temp_max={max(self.temps.values())}° "
                , 'statistic'
            )

    def log_hourly(self):
        if self.stats_hourly:
            self.log('Hourly',
                     self.get_log_records()
                     + self.get_log_fps()
                     + self.get_log_time_ave()
                     + self.get_log_fpm_ave()
                     + self.get_log_temp_ave()
                     , 'statistic'
                     )
        else:
            self.local_log('Hourly',
                           self.get_log_records()
                           + self.get_log_fps()
                           + self.get_log_time_ave()
                           + self.get_log_fpm_ave()
                           + self.get_log_temp_ave()
                           , 'statistic'
                           )
