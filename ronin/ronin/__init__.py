from importlib.metadata import entry_points

from loguru import logger

RONIN_PLUGINS_GROUP_NAME = "ronin.plugins"
RONIN_PLUGINS = "assistants",

for entry_point in entry_points(group=RONIN_PLUGINS_GROUP_NAME):
    if entry_point.name in RONIN_PLUGINS:
        logger.info(f"Loading plugin {entry_point.name}")
        foo = entry_point.load()
    else:
        logger.warning(f"Skipping unknown plugin {entry_point.name}")
