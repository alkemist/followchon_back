from datetime import datetime

from loguru import logger
from pubsub import pub

from configuration.models import Log
from detections.management.commands.models.brain import Brain
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.architecture import Architecture
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.enums.log_level import Log_Level
from detections.management.commands.models.eye import Eye
from detections.management.commands.models.memory import Memory


class Agent:

    def __init__(self, architecture: Architecture, source: Agent_Source):
        self.architecture = architecture
        self.source = source

        pub.subscribe(self.process_log, Event_Type.AGENT_LOG)

        self.send_log('init', f'{architecture} / {source}')

        self.memory = Memory(
            architecture,
            source,
        )

        self.eye = Eye(
            self.memory
        )

        self.brain = Brain(
            architecture,
            self.memory
        )

    def send_log(self, event: str, infos: str = '', level: Log_Level = None):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.AGENT, event=event, infos=infos, level=level)

    def process_log(self, source: Event_Source, event: str, infos: str = '', level: Log_Level | None = None):
        message = f'[{source}] {event}' + (f' : {infos}' if infos else '')

        if level is not None:
            match level:
                case Log_Level.WARNING, Log_Level.HOT:
                    logger.warning(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
                case Log_Level.ERROR, Log_Level.FAIL, Log_Level.HOT:
                    logger.error(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
                case _:
                    logger.info(message, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

            if level != Log_Level.LOCAL:
                Log().create(self.source, level, event, infos)
        else:
            print(message)

    def start(self):
        self.memory.log_start()

        self.brain.start()
        self.brain.check('start')

        while self.memory.brain_enabled:
            now = datetime.now()

            if self.memory.date is not None and self.memory.date.hour != now.hour:
                self.memory.log_hour()

            self.memory.date = now
            self.memory.size = len(self.memory.get_memories())
            self.memory.add_temperature()

            if self.source == Agent_Source.VISION:
                if not self.memory.is_awake() and self.memory.is_low():
                    self.brain.sleep(60)
                if self.memory.is_low() and self.memory.is_awake():
                    if not self.memory.memory_recording:
                        self.memory.record('start')
                    elif self.memory.is_lost():
                        self.memory.record('lost')
                if self.memory.memory_recording:
                    if self.memory.is_full():
                        self.memory.stop('full')
                    elif not self.memory.is_awake():
                        self.memory.stop()

            if not self.memory.is_low() and not self.memory.is_empty():
                self.eye.watch()

            if self.memory.is_empty() and (not self.memory.is_awake() or self.source != Agent_Source.VISION):
                self.memory.terminate('finish')

    def end(self):
        self.memory.log_end()

        if self.architecture == Architecture.HAILO:
            self.memory.check_disk_free()

        if self.source == Agent_Source.VISION:
            self.memory.log_statistics()

        self.memory.stop()
        self.brain.stop()
