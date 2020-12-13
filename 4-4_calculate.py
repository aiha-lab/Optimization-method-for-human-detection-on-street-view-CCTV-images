"""
IITP 04: evaluation.py
"""
import os
import sys
import json
import pandas as pd
import pdb

MINOVERLAP = 0.50
BASE_NP = 6554609
BASE_INF = 4690
BASE_AP = 0.29205789973041


def compute_final_score(num_params, inference_time):
    return num_params / BASE_NP + inference_time / BASE_INF


def voc_ap(rec, prec):

    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    #
    # pdb.set_trace()
    return ap, mrec, mpre


def compute_AP(tp, fp, gt_len, num):

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    rec = tp[:] # recall
    
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / num
    
    prec = tp[:] # precision
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    
    ap, mrec, mprec = voc_ap(rec[:], prec[:])
    
    return ap


def cal_IoU(ground_truth_data, dr_data):

    if len(ground_truth_data) == 0:
        if len(dr_data) == 0:
            return [], []
        else:
            return [0]*len(dr_data), [1]*len(dr_data)
    else:
        if len(dr_data) == 0:
            return [], []

    tp, fp = [0]*len(dr_data), [0]*len(dr_data)

    used = [0]*len(ground_truth_data)

    for idx, bb in enumerate(dr_data):
        ovmax = -1
        gt_match = -1

        for jdx, obj in enumerate(ground_truth_data):
            bbgt = obj[:4]
            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw > 0 and ih > 0:
                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                ov = iw * ih / ua

                if ov > ovmax:
                    ovmax = ov
                    gt_match = jdx

        min_overlap = MINOVERLAP
        if ovmax >= min_overlap:
            if gt_match != -1 and used[gt_match] == 0:
                tp[idx] = 1
                used[gt_match] = 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    return tp, fp


def track4_evaluation(gt, pred):

    # load ground truths
    with open(gt, 'r') as json_file:
        gt_data = json.load(json_file)

    ground_truths = []
    num = 0 # All Groung Truths for Recall
    for img in gt_data['annotations']:
        gt = {}
        gt['file_name'] = img['file_name']

        gt['positions'] = []
        
        for i, obj in enumerate(img['objects']):
            num = num + 1
            gt['positions'].append(obj['position'])
        
        ground_truths.append(gt)
    
    df_gt = pd.DataFrame(ground_truths)
    gt_len = df_gt.shape[0]

    # predictions
    with open(pred, 'r') as json_file:
        pred_data = json.load(json_file)

    img_annots = pred_data['annotations']
    param_num = float(str(pred_data['param_num']).replace(",",""))
    inference_time = float(str(pred_data['inference_time']).replace(",",""))

    predictions = []
    for img in img_annots:
        img_fn = img['file_name']
        img_index = len(predictions)

        img_dt = []
        for i, obj in enumerate(img['objects']):
            bbox = {}
            bbox['file_name'] = img_fn
            bbox['obj_id'] = i
            bbox['position'] = obj['position']
            img_dt.append(obj['position'])
            bbox['confidence_score'] = float(obj['confidence_score'])
            bbox['tp'] = 0
            bbox['fp'] = 0
            predictions.append(bbox)

        
        # get TP/FP
        img_gt = df_gt.loc[df_gt['file_name'] == img_fn]['positions']
        img_gt = img_gt.tolist()[0]
        
        tp, fp = cal_IoU(img_gt, img_dt)

        for jdx in range(len(img_dt)):
            predictions[img_index + jdx]['tp'] = tp[jdx]
            predictions[img_index + jdx]['fp'] = fp[jdx]

    
    #pdb.set_trace()
    df_pred = pd.DataFrame(predictions)
    df_pred = df_pred.sort_values(by=['confidence_score'], ascending=False, ignore_index=True)

    true_positives = list(df_pred['tp'])
    false_positives = list(df_pred['fp'])

    ap = compute_AP(true_positives, false_positives, gt_len, num)
    print("AP Score:",ap)
    if ap < BASE_AP * 0.95:
        ap = 999
        score = 999
    else:
        score = compute_final_score(param_num, inference_time)

#    print(f'score:{score}')

if __name__ == '__main__':
    gt = sys.argv[1]
    pred = sys.argv[2]
    #pb_key = sys.argv[3]
    track4_evaluation(gt, pred)

