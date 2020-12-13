"""Utility functions for clana."""

import os
import yaml
from typing import Any, Dict, List, Optional

from .conf import *


def load_config(yaml_filepath: str, save_cm_plot, 
		save_score_plot, save_hierarchy_labels, 
		verbose: bool = False) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str, optional (default: package config file)
    name: str
    save_cm_plot: bool or None
    save_opt_plot: bool or None
    save_hierarchy_labels: bool or None
    verbose: bool (show load config print)

    Returns
    -------
    cfg : Dict[str, Any]
    
    """
    import os
    path = os.getcwd()
    yaml_filepath = path+"/modules/classify_v2/clana/config.yaml"
    # Read YAML experiment definition file
    if verbose:
        print("## Load config from %s\n" % yaml_filepath)
    
    with open(yaml_filepath) as stream:
        cfg = yaml.safe_load(stream)
        
        if save_cm_plot:
            cfg['visualize']['output_cm_path'] = 'save/clana_cm('+DATA_NAME+').pdf'

        if save_score_plot:    
            cfg['visualize']['output_score_path'] = 'save/clana_opt('+DATA_NAME+').png'   
        
        if save_hierarchy_labels:
            cfg['visualize']['output_hierarchy_path'] = 'save/hierarchy_labels('+DATA_NAME+').json'
        
        if save_cm_plot or save_score_plot or save_hierarchy_labels:
            with open(yaml_filepath, 'w') as outfile:
                yaml.dump(cfg, outfile, default_flow_style=False)
    
    config = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    
    return config


def make_paths_absolute(dir_: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : Dict[str, Any]

    Returns
    -------
    cfg : Dict[str, Any]
    
    """
    
    for key in cfg.keys():
        if hasattr(key, "endswith") and key.endswith("_path"):
            if cfg[key].startswith("~"):
                cfg[key] = os.path.expanduser(cfg[key])
            else:
                cfg[key] = os.path.join(dir_, cfg[key])
            
            cfg[key] = os.path.abspath(cfg[key])
        
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    
    return cfg
