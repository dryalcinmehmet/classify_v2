"""
@author: funda
@author: uğuray

"""

# import json, codecs
from .clana import main
import pandas as pd
from .clana.conf import *


def Clana(test_labels, name, cls_reports):
    '''
    This method automatically clusters similar classes together, 
    i.e. the classes that are most confused with each other.
    
    Parameters
    ----------
    test_labels: pd.series
    name: str
    cls_reports: dict

    Returns
    -------
    extract_labels: list

    '''

    # labels
    set_labels = pd.DataFrame(test_labels.unique(), columns=['labels']).sort_values('labels')
    
    # saving labels
    # labels_name = "label("+str(len(set_labels))+"_"+name+").csv"
    # set_labels.to_csv(CLANA_FOLDER+"save/"+labels_name, index=False)
    #                  separators=(',', ':'), sort_keys=True, indent=4)

    # confusion matrix
    conf_name = "conf_matrix("+str(name)+").json"
    conf_matrix = cls_reports['confusion_matrix'].tolist() 
    
    # saving confusion matrix
    # json.dump(conf_matrix_float, 
    #           codecs.open(CLANA_FOLDER+"save/"+conf_name, 'w', encoding='utf-8'), 
    
    
    # extract labels from clana
    extract_labels = main.clanaMain(
                            cm_=conf_matrix,
                            cm_name=conf_name,
                            steps=1000,
                            labels_=set_labels,
                            save_cm_plot=save_cm_plot,
                            save_score_plot=save_score_plot,
                            save_hierarchy_labels=save_hierarchy_labels
                    )

    # extract_labels = visualize_cm.main(cm_file=CLANA_FOLDER+"save/"+conf_name, 
    #                                    cm_name=conf_name,
    #                                    steps=1000,
    #                                    labels_file=CLANA_FOLDER+"save/"+labels_name, 
    #                                    zero_diagonal=False                
    #                                    hierarchy_save='yes'
    #                  )

    return extract_labels    

