import subprocess

from django.core.exceptions import ObjectDoesNotExist
from loguru import logger

from configuration.models import Parameter


def exec_command(command_str: str):
    return subprocess.Popen(command_str.split(" "),
                            stdout=subprocess.PIPE,
                            universal_newlines=True)


def get_param(param_slug, default_value=None):
    try:
        parameter = Parameter.objects.get(slug=param_slug)
        return parameter.value

    except ObjectDoesNotExist:
        logger.error(f'Param "{param_slug}" not exist')
        
    return default_value
