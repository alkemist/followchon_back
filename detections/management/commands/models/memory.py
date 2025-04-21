import math
import os
import statistics
import time
from datetime import datetime

import psutil
from pubsub import pub

from configuration.models import Zone, Family
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.architecture import Architecture
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.tools import get_param, exec_command
from detections.models import Detection
from utils.array import ArrayHelper
from utils.file import FileHelper


# Il s'occupe de l'accès aux fichiers et l'enregistrement des vidéos
class Memory():
    def __init__(
            self,
            architecture: Architecture,
            source: Agent_Source,
    ):
        self.perception = None

        self.source = source
        self.stream_path = os.getenv('LIVE_STREAM_PATH')
        self.show_stream = os.getenv('SHOW_STREAM') == 1
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.records_directory = f"./records/{source}"
        self.record_exts = 'jpg|png' if source == Agent_Source.PHOTO else 'mkv|mp4'

        self.date = datetime.now()
        self.eye_start = None
        self.queue = 0
        self.record_time = 60
        self.record_time_delay = 50
        self.last_record_seconds = time.time()
        self.recording = False
        self.vcgm = None
        self.temperature = 0

        if architecture == Architecture.HAILO:
            from vcgencmd import Vcgencmd
            self.vcgm = Vcgencmd()

        self.move_tolerance_margin_norm = float(get_param('vision_detection_move_margin_norm', 0.06))
        self.temp_alert = int(get_param('vision_temp_alert'))
        self.popcorn_frame_count = int(get_param('vision_popcorn_frame_count'))

        self.hour_min = 0
        self.hour_max = 0
        self.frames_classified_step = 0
        self.frames_saved_step = 0
        self.pause_capture_seconds = 0
        self.records_max = 0
        self.brain_enabled = False

        self.temperatures: dict[str, float] = {}
        self.durations = []
        self.fpm_counts = []
        self.queues = []

        self.classify_families = Family.objects.filter(is_listed=True, is_unique=True)
        self.classify_families_dict = ArrayHelper.object_list_to_dict(
            self.classify_families,
            'slug'
        )
        self.families_dict = ArrayHelper.object_list_to_dict(
            Family.objects.all(),
            'index'
        )

        if self.source == Agent_Source.VISION:
            last_detections = Detection.objects.raw('''
                SELECT * FROM (
                    SELECT d.*
                    FROM detections_detection d
                    LEFT JOIN configuration_family f ON d.family_id = f.id
                    WHERE f.is_tracked = true
                    ORDER BY d.id DESC
                 ) d 
                 GROUP BY d.family_id
            ''')

            last_detections_dict: dict[int, Detection] = (
                ArrayHelper.object_list_to_dict(last_detections, 'family_id')
            )

            self.last_detections: dict[int, (float, float)] = (
                dict(
                    map(
                        lambda kv: (kv[0], (kv[1].center_x, kv[1].center_y)),
                        last_detections_dict.items()
                    )
                )
            )

            self.zones = Zone.objects.filter(is_enabled=True).order_by('id')
        else:
            self.last_detections = {}
            self.zones = []

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.MEMORY, event=event, infos=infos, level=level)

    def check(self, reason):
        self.hour_min = int(get_param('vision_hour_min', 7))
        self.hour_max = int(get_param('vision_hour_max', 20))
        self.frames_classified_step = float(get_param('vision_frames_classified_step', 2))
        self.frames_saved_step = int(get_param('vision_frames_saved_step', 10))
        self.pause_capture_seconds = float(get_param('vision_pause_capture_seconds', 0.1))
        self.records_max = int(get_param('vision_records_max', 20))

        self.brain_enabled = int(get_param('vision_enabled', 0)) == 1

        if not self.brain_enabled:
            self.terminate('check')

    def get_memories(self):
        return FileHelper.list_files(self.records_directory, r'.*\.(' + self.record_exts + ')$')

    def get_last_memory(self):
        memories = self.get_memories()

        if len(memories) > 0:
            return f"{self.records_directory}/{memories[0]}"

        return None

    def has_memory(self):
        return self.queue > 0

    def is_full(self):
        return self.queue >= self.records_max

    def is_low(self):
        return self.source == Agent_Source.VISION and self.queue < 2

    def is_empty(self):
        return self.queue == 0

    def is_started(self):
        return (self.hour_max > self.hour_min and self.hour_min <= self.date.hour) \
            or (self.hour_max < self.hour_min and (self.date.hour < self.hour_min))

    def is_awake(self):
        return (self.hour_max > self.hour_min and self.hour_min <= self.date.hour < self.hour_max) \
            or (self.hour_max < self.hour_min and (self.date.hour >= self.hour_max or self.date.hour < self.hour_min))

    def is_lost(self):
        capture_time_elapsed = round(time.time() - self.last_record_seconds)
        return capture_time_elapsed >= self.record_time + self.record_time_delay

    def add_temperature(self, force: bool = False):
        if self.vcgm is not None:
            minute = math.floor(self.date.minute / 10)

            key = f"{self.date.hour}:{minute}"

            if force or key not in self.temperatures:
                temperature = self.vcgm.measure_temp()
                self.temperatures[key] = round(temperature, 1)

                if temperature > self.temp_alert:
                    self.log_warning_temperature()

    def record(self, reason: str = ''):
        self.send_log('record', reason)

        if self.recording:
            self.log_restart_recording()
        else:
            self.log_start_recording()

        exec_command((f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                      f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                      f"-segment_time {self.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                      f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv"))

        self.last_record_seconds = time.time()
        self.recording = True

    def terminate(self, reason: str):
        self.send_log('terminate', reason, Log_Level.LOCAL)
        self.brain_enabled = False

    def stop(self, reason: str = ''):
        self.log_stop_recording(reason)

        exec_command("pkill ffmpeg")

        self.recording = False

    def log_hour(self):
        infos = (f"records={self.queue}/{self.records_max} " +
                 f"frame_seconds={self.frames_classified_step}s " +
                 f"pause_capture={self.pause_capture_seconds}s ")

        self.send_log('Hour', infos, Log_Level.LOCAL)

        self.log_statistics(is_hour=True)

    def log_popcorn(self, capture_date, frame_saved_count):
        self.send_log(
            'Popcorn',
            f"count={frame_saved_count}/{self.popcorn_frame_count} "
            f"time={capture_date.strftime('%H:%M')} "
            , Log_Level.EVENT
        )

    def log_warning_temperature(self):
        self.send_log(
            'Temperature',
            f"records={self.queue}/{self.records_max} " +
            f"frame_seconds={self.frames_classified_step}s " +
            f"pause_capture={self.pause_capture_seconds}s " +
            f"temp_ave={round(statistics.fmean(self.temperatures.values()), 2)}° " +
            f"temp_max={max(self.temperatures.values())}° ",
            Log_Level.HOT
        )

    def log_start(self):
        self.queue = len(self.get_memories())

        self.send_log(
            'Started',
            f"records={self.queue}/{self.records_max} ",
            Log_Level.INFO
        )

    def log_start_recording(self):
        self.send_log(
            'Start recording',
            "",
            Log_Level.INFO
        )

    def log_stop_recording(self, reason: str = ''):
        infos = ''

        if reason:
            infos = (
                    f"records={self.queue}/{self.records_max} " +
                    f"frame_seconds={self.frames_classified_step}s " +
                    f"pause_capture={self.pause_capture_seconds}s "
            )

        self.send_log(
            'Stop recording',
            infos,
            Log_Level.WARNING if reason else Log_Level.INFO
        )

    def log_restart_recording(self):
        self.send_log(
            'Restart recording',
            f"records={self.queue}/{self.records_max} ",
            Log_Level.WARNING
        )

    def log_end(self):
        self.queue = len(self.get_memories())

        infos = (
                f"records={self.queue}/{self.records_max} "
                f"frame_seconds={self.frames_classified_step}s " +
                f"pause_capture={self.pause_capture_seconds}s "
        )

        disk = os.getenv('WATCH_DISK')
        if disk:
            disk_info = psutil.disk_usage(disk)
            infos = infos + f"disk_usage={disk_info.percent}% "

        self.send_log(
            'Stopped',
            infos,
            Log_Level.INFO
        )

    def log_statistics(self, is_hour=True):
        if len(self.fpm_counts) > 0:
            self.send_log(
                'Analyses',
                f"fpm_min={min(self.fpm_counts)} " + \
                f"fpm_max={max(self.fpm_counts)} " \
                f"fpm_ave={round(statistics.fmean(self.fpm_counts), 2)} "
                f"fpm_med={round(statistics.median(self.fpm_counts), 2)} ",
                Log_Level.LOCAL if is_hour else Log_Level.STATISTIC
            )

        if len(self.durations) > 0:
            self.send_log(
                'Processing',
                f"duration_min={min(self.durations)} " + \
                f"duration_max={max(self.durations)} " \
                f"duration_ave={round(statistics.fmean(self.durations), 2)} "
                f"duration_med={round(statistics.median(self.durations), 2)} ",
                Log_Level.LOCAL if is_hour else Log_Level.STATISTIC
            )

        if len(self.queues) > 0:
            self.send_log(
                'Queue',
                f"queue_min={min(self.queues)} " + \
                f"queue_max={max(self.queues)} " \
                f"queue_ave={round(statistics.fmean(self.queues), 2)} "
                f"queue_med={round(statistics.median(self.queues), 2)} ",
                Log_Level.LOCAL if is_hour else Log_Level.STATISTIC
            )

        if len(self.temperatures.values()) > 0:
            self.send_log(
                'Temperature',
                f"temp_min={min(self.temperatures.values())}° " + \
                f"temp_max={max(self.temperatures.values())}° " + \
                f"temp_ave={round(statistics.fmean(self.temperatures.values()), 2)}° "
                f"temp_med={round(statistics.median(self.temperatures.values()), 2)}° ",
                Log_Level.LOCAL if is_hour else Log_Level.STATISTIC
            )

    def check_disk_free(self):
        disk = os.getenv('WATCH_DISK')

        if disk:
            disk_info = psutil.disk_usage(disk)

            if disk_info.percent >= 80:
                self.send_log(
                    'Disk',
                    f"disk_used={FileHelper.convert_size(disk_info.used)} "
                    f"disk_free={FileHelper.convert_size(disk_info.free)} "
                    f"disk_usage={disk_info.percent}% "
                    , Log_Level.WARNING
                )
