import numpy as np


class WMAPE:
    def __init__(self, ):
        pass

    def __call__(self, y_true, y_pred, threshold=0.0):
        """ 加权绝对百分比误差，实际值与预测值差值的绝对值除以序列所有实际值的平均值 """
        gt, pred, delta = [], [], []
        for i, ele in enumerate(y_true):
            if ele > threshold:
                gt.append(ele)
                pred.append(y_pred[i])
                delta.append(ele - y_pred[i])
        diff = np.abs(delta)
        tm = np.nanmean(np.abs(gt))
        if tm == 0: return 0, 0
        diff = np.nanmean(diff / tm)
        return diff
