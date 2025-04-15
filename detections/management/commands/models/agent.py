from datetime import datetime

from pubsub import pub

from detections.management.commands.models.brain import Brain
from detections.management.commands.models.enums.agent_source import Agent_Source
from detections.management.commands.models.enums.architecture import Architecture
from detections.management.commands.models.enums.event_source import Event_Source
from detections.management.commands.models.enums.event_type import Event_Type
from detections.management.commands.models.eye import Eye
from detections.management.commands.models.memory import Memory


class Agent:

    def __init__(self, archi: Architecture, source: Agent_Source):
        self.source = source

        pub.subscribe(self.process_log, Event_Type.AGENT_LOG)

        self.send_log('init', f'{archi} / {source}')

        self.memory = Memory(
            source
        )

        self.eye = Eye(
            self.memory
        )

        self.brain = Brain(
            archi,
            self.memory
        )

    def send_log(self, action: str, infos: str = ''):
        pub.sendMessage(Event_Type.AGENT_LOG, source=Event_Source.AGENT, action=action, infos=infos)

    def process_log(self, source: Event_Source, action: str, infos: str = ''):
        print(f'[{source}] {action}' + (f' : {infos}' if infos else ''))

    def start(self):
        self.send_log('start')

        self.brain.start()
        self.brain.check('start')

        while self.memory.brain_enabled:
            hour = datetime.now().hour

            if self.memory.hour is not None and self.memory.hour != hour:
                self.brain.check('hour')

            self.memory.hour = hour
            self.memory.size = len(self.memory.get_memories())

            if self.source == Agent_Source.VISION:
                if not self.memory.is_awake() and self.memory.is_low():
                    self.brain.sleep(60)
                elif self.memory.is_low() and self.memory.is_awake() and \
                        (not self.memory.memory_recording or self.memory.is_lost()):
                    self.memory.record()
                elif self.memory.memory_recording and self.memory.is_full() or not self.memory.is_awake():
                    self.memory.stop()

            if not self.memory.is_low() and not self.memory.is_empty():
                self.eye.watch()

            if self.source == Agent_Source.VISION:
                if self.memory.frame_saved_count > self.memory.popcorn_frame_count:
                    self.send_log('popcorn', self.memory.frame_saved_count)  # capture_date

            if self.memory.is_empty() and (not self.memory.is_awake() or self.source != Agent_Source.VISION):
                self.memory.disable('finish')

    def finish(self):
        self.memory.stop()
        self.brain.stop()
        self.send_log('finish')
        return True
