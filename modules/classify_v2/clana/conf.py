# called functions each time
import sys
sys.path.append('..')
import logging.config
from typing import TypeVar
# import matplotlib
from modules.classify_v2.param_config import PREP_PARAMS, unwanted_chars, CLANA_FOLDER
from . import utils


# in clustering for apply_grouping function
T=TypeVar("T")


# config.yaml
save_cm_plot=False
save_score_plot=False
save_hierarchy_labels=False

config = utils.load_config(CLANA_FOLDER + 'config.yaml', save_cm_plot,
                           save_score_plot, save_hierarchy_labels)


# for logs 
logger = logging.getLogger(__name__)
#logging.config.dictConfig(config["LOGGING"])
#logging.getLogger("matplotlib").setLevel("CRITICAL")
#logging.getLogger("matplotlib.pyplot").setLevel("CRITICAL")


# for do not appear of plot output
# matplotlib.use("Agg") 



