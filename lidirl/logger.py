################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'lidirl'


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

#################################### FUNCTIONALITY ####################################
from typing import Dict
import json

from . import __version__

class TrainingLogger():
    def __init__(self, stdout : bool = True,
                        wandb_config : Dict = None ):
        self.stdout = stdout
        if wandb_config is not None:
            try:
                global wandb
                import wandb

                tags = os.getenv("WANDB_RUN_TAGS", default=None)
                if tags is not None:
                    tags = tags.split(',')
                try:
                    wandb.login(key=os.environ.get("WANDB_API_KEY"), force=True)
                except:
                    logger.info("No wandb login key found. Using default if logged in.")

                wandb.init(
                    entity="rewicks",
                    project=wandb_config.get("project_name", None),
                    config = wandb_config.get("config", None),
                    group = os.getenv("WANDB_RUN_GROUP", default=None),
                    tags = tags,
                    name = os.getenv("WANDB_RUN_NAME", default=None),
                    notes = os.getenv("WANDB_RUN_NOTES", default=None),
                    resume="allow",
                    id=os.getenv("WANDB_RUN_ID", default=None)
                )
                self.use_wandb = True

            except:
                logger.info("wandb not installed. If you would like to log your training with this tool please `pip install wandb`")
                self.use_wandb = False
        else:
            logger.info("No wandb project passed. Not logging online.")
            self.use_wandb = False

    def log(self, out_json):
        if self.stdout:
            logger.info(json.dumps(out_json))
        if self.use_wandb:
            log_json = {}
            for key, value in out_json.items():
                log_json[f"{out_json['type']}/{key}"] = value
            wandb.log(log_json, commit=True)

    def finish_log(self, end_message=None):
        if end_message is not None:
            logger.info(end_message)
        if self.use_wandb:
            wandb.finish()
        


        

