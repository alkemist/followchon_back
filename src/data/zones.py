from enum import Enum

class Zones(Enum):
    CLAPIER = 'clapier'
    CACHETTE = 'cachette'
    FOIN = 'foin'
    CENTER = 'center'
    TUNNEL = 'tunnel'
    FONTAINE = 'fontaine'
    BAS = 'bas'

data_zones = [
    {
        'name': Zones.CLAPIER,
        'points': [0.42699724517906334, 0.07601877892248965, 0.16528925619834722, 0.14507498169180263]
    },
    {
        'name': Zones.CACHETTE,
        'points': [0.5730027548209367, 0.07657928015551248, 0.11937557392102838, 0.14395397922575698]
    },
    {
        'name': Zones.FOIN,
        'points': [0.7213039485766758, 0.10529537802265078, 0.17355371900826433, 0.19100091827364551]
    },
    {
        'name': Zones.TUNNEL,
        'points': [0.30020655163071497, 0.2543300984633369, 0.11488250652741505, 0.20887728459530022]
    },
    {
        'name': Zones.FONTAINE,
        'points': [0.19800994244672007, 0.7072414676347746, 0.38083961923879117, 0.5623249922025152]
    },
    {
        'name': Zones.BAS,
        'points': [0.7676878564903726, 0.7904982781643124, 0.4554859058182106, 0.39581137114343945]
    }
]
