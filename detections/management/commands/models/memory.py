import os
import time

from pubsub import pub

from configuration.models import Zone, Family
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.tools import get_param, exec_command
from detections.models import Detection
from utils.array import ArrayHelper
from utils.file import FileHelper


# Il s'occupe de l'accès aux fichiers et l'enregistrement des vidéos
class Memory():
    def __init__(
            self,
            source: Agent_Source,
    ):
        self.send_log('init')

        self.source = source
        self.stream_path = os.getenv('LIVE_STREAM_PATH')
        self.show_stream = os.getenv('SHOW_STREAM') == 'True'
        self.capture_width = int(os.getenv('CAPTURE_WIDTH'))

        self.records_directory = f"./records/{source}"
        self.record_exts = 'jpg|png' if source == Agent_Source.PHOTO else 'mkv|mp4'

        self.hour = None
        self.size = 0
        self.record_time = 60
        self.record_time_delay = 50
        self.last_record_seconds = time.time()
        self.frame_saved_count = 0
        self.memory_recording = False

        self.score_min = float(get_param('vision_score_min', 0.7))
        self.move_tolerance_margin_norm = float(get_param('vision_detection_move_margin_norm', 0.06))
        self.temp_alert = int(get_param('vision_temp_alert'))
        self.popcorn_frame_count = int(get_param('vision_popcorn_frame_count'))

        self.hour_min = 0
        self.hour_max = 0
        self.frame_seconds = 0
        self.pause_capture_seconds = 0
        self.records_max = 0
        self.brain_enabled = False
        self.perception = None

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

    def send_log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.MEMORY, action=action, infos=infos)

    def check(self, origin):
        self.send_log('check', origin)

        self.hour_min = int(get_param('vision_hour_min', 7))
        self.hour_max = int(get_param('vision_hour_max', 20))
        self.frame_seconds = float(get_param('vision_frame_seconds', 0.1))
        self.pause_capture_seconds = float(get_param('vision_pause_capture_seconds', 0.1))
        self.records_max = int(get_param('vision_records_max', 20))

        self.brain_enabled = get_param('vision_enabled', False)

        if not self.brain_enabled:
            self.disable('check')

    def get_memories(self):
        return FileHelper.list_files(self.records_directory, r'.*\.(' + self.record_exts + ')$')

    def get_last_memory(self):
        memories = self.get_memories()

        if len(memories) > 0:
            return f"{self.records_directory}/{memories[0]}"

        return None

    def has_memory(self):
        return self.size > 0

    def is_full(self):
        return self.size >= self.records_max

    def is_low(self):
        return self.source == Agent_Source.VISION and self.size < 2

    def is_empty(self):
        return self.size == 0

    def is_awake(self):
        return (self.hour_max > self.hour_min and self.hour_min <= self.hour < self.hour_max) \
            or (self.hour_max < self.hour_min and (self.hour >= self.hour_max or self.hour < self.hour_min))

    def is_lost(self):
        capture_time_elapsed = round(time.time() - self.last_record_seconds)
        return capture_time_elapsed >= self.record_time + self.record_time_delay

    def record(self):
        self.send_log('record')

        exec_command((f"ffmpeg -hide_banner -y -loglevel error -rtsp_transport tcp -use_wallclock_as_timestamps "
                      f"1 -i {self.stream_path} -vcodec copy -acodec copy -f segment -reset_timestamps 1 "
                      f"-segment_time {self.record_time} -segment_format mkv -segment_atclocktime 1 -strftime 1 "
                      f"{self.records_directory}/%Y-%m-%d_%H-%M-%S.mkv"))

        self.memory_recording = True

    def disable(self, origin: str):
        self.send_log('disable', origin)
        self.brain_enabled = False

    def stop(self):
        self.send_log('stop')

        exec_command("pkill ffmpeg")

        self.memory_recording = False
