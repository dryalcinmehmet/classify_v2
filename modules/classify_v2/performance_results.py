"""
@author: funda
@author: uğuray

"""

import numpy as np

import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize

import scipy.stats.mstats as scipystats

import os, io
import base64
import matplotlib.pyplot as plt

FIGURES_FOLDER = 'figures'


def get_accuracy(ytrue, ypred):
    return metrics.accuracy_score(ytrue, ypred)


def get_f1score(ytrue, ypred, avg="macro"):
    return metrics.f1_score(ytrue, ypred, average=avg)


def get_precision(ytrue, ypred, avg="macro"):
    return metrics.precision_score(ytrue, ypred, average=avg)


def get_recall(ytrue, ypred, avg="macro"):
    return metrics.recall_score(ytrue, ypred, average=avg)


def get_gmean(ytrue, ypred, avg="macro"):
    pre = get_precision(ytrue, ypred, avg)
    rec = get_recall(ytrue, ypred, avg)
    return scipystats.gmean([pre, rec])


# return the number of TP (true positive) instances per class
def get_TP_perclass(ytrue, ypred):
    cm = metrics.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))
    nTPs = [cm[i,i] for i in range(len(classes))]
    return nTPs


# return the number of TN (true negative) instances per class
def get_TN_perclass(ytrue, ypred):
    cm = metrics.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))
    n_instances = np.sum(cm)
    
    nTNs = []
    for i,_ in enumerate(classes):
        tn_i = n_instances - (sum(cm[i,:]) + sum(cm[:,i]) - cm[i,i])
        nTNs.append(tn_i)
    return nTNs


# return the number of FP (false positive) instances per class
def get_FP_perclass(ytrue, ypred):
    cm = metrics.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))

    nFPs = []
    for i,c in enumerate(classes):
        fp_i = sum(cm[:,i]) - cm[i,i]
        nFPs.append(fp_i) 
    return nFPs


# return the number of FN (false negative) instances per class
def get_FN_perclass(ytrue, ypred):
    cm = metrics.confusion_matrix(ytrue, ypred)
    classes = sorted(list(set(ytrue)))

    nFNs = []
    for i,c in enumerate(classes):
        tn_i = sum(cm[i,:]) - cm[i,i]
        nFNs.append(tn_i)
    return nFNs


def get_FP_overall(ytrue, ypred):
    return sum(get_FP_perclass(ytrue, ypred))


def get_TN_overall(ytrue, ypred):
    cm = metrics.confusion_matrix(ytrue, ypred)
    return np.sum(cm) - np.sum(np.diag(cm))


def get_FN_overall(ytrue, ypred):
    return sum(get_FN_perclass(ytrue, ypred))


def get_TP_overall(ytrue, ypred):
    return sum(get_TP_perclass(ytrue, ypred))


def get_accuracy_perclass(ytrue, ypred):

    TPs = np.array(get_TP_perclass(ytrue, ypred))
    FPs = np.array(get_FP_perclass(ytrue, ypred))
    TNs = np.array(get_TN_perclass(ytrue, ypred))
    FNs = np.array(get_FN_perclass(ytrue, ypred))
    
    acc_perclass = (TPs + TNs) / (TPs + TNs + FPs + FNs)
    return acc_perclass.tolist()


def get_fscore_perclass(ytrue, ypred):
    return metrics.f1_score(ytrue, ypred, average=None)


def get_precision_perclass(ytrue, ypred):
    return metrics.precision_score(ytrue, ypred, average=None)


def get_recall_perclass(ytrue, ypred):
    return metrics.recall_score(ytrue, ypred, average=None)


def get_gmean2(precision, recall):
    return scipystats.gmean([precision, recall])


def binarize_labels(ytrue):
    classes_ = sorted(list(set(ytrue)))
    
    if len(classes_) < 3:
        bin_labels1 =  label_binarize(ytrue, classes=classes_)
        bin_labels = np.zeros((len(ytrue), 2), dtype=int)
        for i,label in enumerate(bin_labels1):
            bin_labels[i, label] = 1   # assuming label_binarize applies 0-1 encoding
    else:
        bin_labels =  label_binarize(ytrue, classes=classes_)
    return np.array(bin_labels, dtype=int)
 

def get_auc_perclass(ytrue, ypred):
    ''' 
    As defined in Sokolova and Lapalme, 2009, A systematic 
    analysis of performance measures for classification tasks 
    '''
    TPs = np.array(get_TP_perclass(ytrue, ypred))
    FPs = np.array(get_FP_perclass(ytrue, ypred))
    TNs = np.array(get_TN_perclass(ytrue, ypred))
    FNs = np.array(get_FN_perclass(ytrue, ypred))
    
    auc_perclass = 0.5 * ((TPs / (TPs + FNs)) + (TNs / (TNs + FPs)))
    return auc_perclass.tolist()


def get_auc_avg(ytrue, ypred):
    AUCs = get_auc_perclass(ytrue, ypred)
    return sum(AUCs) / len(AUCs)


def find_roc_values(classes, ytrue, yscore):
    '''
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    '''
    ytruebin = binarize_labels(ytrue)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds, roc_auc = {}, {}, {}, {}
    
    for i, c in enumerate(classes):
        fpr[c], tpr[c], thresholds[c] = metrics.roc_curve(ytruebin[:, i], yscore[:, i])
        roc_auc[c] = metrics.auc(fpr[c], tpr[c])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(ytruebin.ravel(), yscore.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, thresholds, roc_auc


def draw_roc_curve(fpr, tpr, roc_auc, class_name, figpath=None):
    '''
    Valindex is either class index to draw the roc curve for the specific 
    class or micro to draw the roc curve (micro-avg'd) for the model
    '''
    plt.figure()
    lw = 2
    plt.plot(fpr[class_name], tpr[class_name], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[class_name])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristics Curve')
    plt.legend(loc="lower right")
    # if figpath:
        # plt.savefig(figpath)
    #plt.show()
    strio_byte = io.BytesIO()
    plt.savefig(strio_byte, format='png')
    strio_byte.seek(0)
    return base64.b64encode(strio_byte.read())


def find_precision_recall_curve_values(ytrue, yscore):
    '''
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
    '''    
    ytruebin = binarize_labels(ytrue)
    pr_rec_curve_dict = {}
    n_classes = ytruebin.shape[1] 
    
    for i in range(n_classes):
        pr_rec_curve_dict[i] = metrics.precision_recall_curve(ytruebin[:, i], yscore[:, i])
    
    pr_rec_curve_dict["micro"] = metrics.precision_recall_curve(ytruebin.ravel(), yscore.ravel())
    return pr_rec_curve_dict


def draw_precision_recall_curve(precisions, recalls, figpath=None):
    '''
    Valindex is either class index to draw the curve for the specific 
    class or micro to draw the curve (micro-avg'd) for the model
    '''
    plt.figure()
    lw = 2
    plt.plot(precisions, recalls, color='darkorange', lw=lw, label='PR curve')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Precision values over thresholds')
    plt.ylabel('Recall values over thresholds')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower left")
    # if figpath:
        # plt.savefig(figpath)
    #plt.show()
    strio_byte = io.BytesIO()
    plt.savefig(strio_byte, format='png')
    strio_byte.seek(0)
    return base64.b64encode(strio_byte.read())


def find_average_precision_values(ytrue, yscore, avg):
    ytruebin = binarize_labels(ytrue)
    avg_pr_dict = dict()
    n_classes = ytruebin.shape[1]
    
    for i in range(n_classes):
        avg_pr_dict[i] = metrics.average_precision_score(ytruebin[:, i], yscore[:, i], average = avg)
    
    avg_pr_dict["micro"] = metrics.average_precision_score(ytruebin.ravel(), yscore.ravel(), average = avg)
    return avg_pr_dict


def make_equal_sizes(items, extra_element=None):
    '''
        Make array elements equal sized
    '''
    max_length = max(list(map(lambda x:len(x), items)))
    for item in items:
        item_length = len(item)
        if item_length == max_length:
            continue
        extra_length = max_length - item_length
        for i in range(extra_length):
            item.append(extra_element)
    return items


def get_confusion_matrices_perclass(ytrue, ypred):
    classes = sorted(list(set(ytrue)))
    TNs = get_TN_perclass(ytrue, ypred)
    FPs = get_FP_perclass(ytrue, ypred)
    FNs = get_FN_perclass(ytrue, ypred)
    TPs = get_TP_perclass(ytrue, ypred)
    
    conf_matrices = []
    
    # confusion matrix is of the form [[TN, FP], [FN, TP]]
    # index 0 is for other classes and index 1 is for the target class.
    for i,_ in enumerate(classes):
        cm = np.zeros((2,2), dtype=int)
        cm[0,0] = TNs[i]
        cm[1,0] = FNs[i]
        cm[0,1] = FPs[i]
        cm[1,1] = TPs[i]
        conf_matrices.append(cm)
    
    return conf_matrices


def get_performance_results(test_labels, ypred, yscore):
    '''
    Calculate the model performance metrics.

    Parameters
    ----------
    test_labels : pd.series
    ypred : np.array
    yscore : np.array

    Returns
    -------
    result_dict : dict

    '''

    ytrue = np.array(test_labels)
    ypred = np.array(ypred)
    yscore = np.array(yscore)
    
    # ----------------------------------------------
    # ------------------ data summary --------------
    # ----------------------------------------------
    
    data_summary = {}    
    data_summary['number_of_instance'] = len(ytrue)
    data_summary['number_of_classes'] = len(set(ytrue))
    data_summary['classes'] = sorted(list(set(ytrue)))
    
    # ----------------------------------------------
    # --------------- metrics for all data ---------
    # ----------------------------------------------
    
    # accuracy
    accuracy = {'overall':{}, 'classes':{}}
    accuracy['overall'] = get_accuracy(ytrue, ypred)
    
    # fscore, precision, recall, gmean 
    avgs = ["macro", "micro", "weighted"]
    fscore, precision, recall, gmean = {'overall':{}, 'classes':{}}, \
                                       {'overall':{}, 'classes':{}}, \
                                       {'overall':{}, 'classes':{}}, \
                                       {'overall':{}, 'classes':{}}
    for avg in avgs:
        fscore['overall'][avg] = get_f1score(ytrue, ypred, avg=avg)
        precision['overall'][avg] = get_precision(ytrue, ypred, avg=avg)
        recall['overall'][avg] = get_recall(ytrue, ypred, avg=avg)
        gmean['overall'][avg] = get_gmean(ytrue, ypred, avg=avg)
    
    # ----------------------------------------------
    # ---------------- metrics for classes ---------
    # ---------------------------------------------- 

    # accuracy, fscore, precision, recall, gmean per class
    for i, c in enumerate(data_summary['classes']):
        accuracy['classes'][c] = get_accuracy_perclass(ytrue, ypred)[i]
        fscore['classes'][c] = get_fscore_perclass(ytrue, ypred)[i]
        precision['classes'][c] = get_precision_perclass(ytrue, ypred)[i]
        recall['classes'][c] = get_recall_perclass(ytrue, ypred)[i]
        gmean['classes'][c] = get_gmean2(precision['classes'][c], recall['classes'][c])
        
        
    # ----------------------------------------------
    # ------------- confusion matrix ---------------
    # ----------------------------------------------
    
    confusion_matrix = metrics.confusion_matrix(ytrue, ypred)
    
    # numbers of TPs, FPs, TNs, FNs from confusion matrix
    conf_matrix_val = {'overall':{}, 'classes': {}}
    
    conf_matrix_val['overall']["TP"] = get_TP_overall(ytrue, ypred)
    conf_matrix_val['overall']["FP"] = get_FP_overall(ytrue, ypred)
    conf_matrix_val['overall']["FN"] = get_FN_overall(ytrue, ypred)
    conf_matrix_val['overall']["TN"] = get_TN_overall(ytrue, ypred)


    # -----------------------------------------------------------
    # ----- Receiver Operating Characteristics (ROC) Curve ------
    # -----------------------------------------------------------
    
    # result_dict["Average_AUC"]
    avg_auc = get_auc_avg(ytrue, ypred)
    
    # result_dict["fpr-(false_positive_rate)"], result_dict["tpr-(true-positive-rate)"], 
    # result_dict["thresholds"], result_dict["roc_auc"] 
    fpr, tpr, thresholds, roc_auc = find_roc_values(data_summary['classes'], ytrue, yscore)
    roc_values = {
                  "average_auc": avg_auc,
                  "fpr":fpr,
                  "tpr": tpr,
                  "thresholds": thresholds,
                  "roc_auc": roc_auc
                 }

    # ROC Curves
    roc_curves = {'overall':None, 'classes':{}}
    
    # ROC curve for the model
    figpath = os.path.join(FIGURES_FOLDER, "roc_curve_micro-avg.png")
    roc_curves['overall'] = draw_roc_curve(fpr, tpr, roc_auc, "micro", figpath)
    # plt.show()

    # ROC curves for the classes
    for i, c in enumerate(data_summary['classes']):
        figpath = os.path.join(FIGURES_FOLDER, "roc_curve-class_"+str(c)+".png")
        roc_curves['classes'][c] = draw_roc_curve(fpr, tpr, roc_auc, c, figpath)

    # -----------------------------------------------------------
    # ---------------- Precision - Recall Curve -----------------
    # -----------------------------------------------------------
    
    # PR curve values
    pre_rec_curve_vals = find_precision_recall_curve_values(ytrue, yscore)
    
    # PR Curves
    pr_curves = {'overall':None, 'classes':{}}

    # PR curve for the model 
    figpath = os.path.join(FIGURES_FOLDER, "pr-rec-curve.png")
    prs = pre_rec_curve_vals["micro"][0]
    recs = pre_rec_curve_vals["micro"][1]
    pr_curves['overall'] = draw_precision_recall_curve(prs, recs, figpath)
    # plt.show()

    # PR curves for the classes
    for i,c in enumerate(data_summary['classes']):
        figpath = os.path.join(FIGURES_FOLDER, "pr-rec-curve_class-"+str(c)+".png")
        prs = pre_rec_curve_vals[i][0]
        recs = pre_rec_curve_vals[i][1]
        pr_curves['classes'][c] = draw_precision_recall_curve(prs, recs, figpath)
    
    # ----------------------------------------
    # ------- create a result dictionary -----
    # ----------------------------------------
    
    cls_reports = {} 
    cls_reports['data_summary'] = data_summary
    cls_reports['accuracy'] = accuracy
    cls_reports['fscores'] = fscore
    cls_reports['precision'] = precision
    cls_reports['recall'] = recall
    cls_reports['gmean'] = gmean
    cls_reports['confusion_matrix'] = confusion_matrix
    
    # TP, TN, FP, FN 
    cls_reports['conf_matrix_val'] = conf_matrix_val
    
    cls_reports["ROC_values"] = roc_values
    cls_reports['ROC_curves'] = roc_curves
    cls_reports['PR_curves'] = pr_curves
    cls_reports['precision_recall_curve_values']= pre_rec_curve_vals
    
    # auprc --> average_precision_scores
    for avg in avgs:
        cls_reports["average_precision_scores"] = find_average_precision_values(ytrue, yscore, avg)
    
    
    # ----------------------------------------
    # --------- metrics per cls --------------
    # ----------------------------------------
    
    n_actuals = [sum(confusion_matrix[i,:]) for i,_ in enumerate(data_summary['classes'])]
    n_predicteds = [sum(confusion_matrix[:,i]) for i,_ in enumerate(data_summary['classes'])]
    cls_reports["n_actuals_per_class"] = n_actuals
    cls_reports["n_predicteds_per_class"] = n_predicteds

    cls_reports['confusion_matrix_per_class'] = get_confusion_matrices_perclass(ytrue, ypred)
    
    cls_reports["TPs_per_class"] = get_TP_perclass(ytrue, ypred)
    cls_reports["FPs_per_class"] = get_FP_perclass(ytrue, ypred)
    cls_reports["FNs_per_class"] = get_FN_perclass(ytrue, ypred)
    cls_reports["TNs_per_class"] = get_TN_perclass(ytrue, ypred)
   
    cls_reports["avg_aucs_per_class"] = get_auc_perclass(ytrue, ypred)
    
    #precision_recall_curve_values calculate
    precision_recall_value_micro = list(cls_reports['precision_recall_curve_values']['micro'])
    precision_recall_value_micro= [list(micro) for micro
                                    in precision_recall_value_micro]
    precision_recall_value_micro = make_equal_sizes(precision_recall_value_micro)
    
    cls_reports['precision_recall_curve'] = [list(micro) for micro
                                   in precision_recall_value_micro]
    
    # result_dict["avg_aucs_per_class"] = get_auc_perclass(ytrue, ypred)
    cls_reports['average_aucs'] = get_auc_perclass(ytrue, ypred)
    
    cls_reports['confusion_matrix_per_class'] = get_confusion_matrices_perclass(ytrue, ypred)
    
    cls_reports['roc_curves'] = roc_curves
    cls_reports['pr_curves'] = pr_curves
    
    return cls_reports
