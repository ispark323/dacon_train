import torch
import numpy as np

from sklearn.metrics import f1_score


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def f_score(pr, gt, onehot=False, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    # if not onehot:
    #     pr = torch.argmax(pr, dim=1)
    #
    # pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    if not onehot:
        pr = torch.argmax(pr, dim=1)
    else:
        pr = torch.argmax(pr, dim=1)
        gt = torch.argmax(gt, dim=1)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    score = f1_score(
        gt.cpu().detach().numpy(), pr.cpu().detach().numpy(), average='macro'
    )
    score = torch.tensor(score)

    return score


def accuracy(pr, gt, onehot=False, threshold=0.5, ignore_channels=None):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)

    if not onehot:
        pr = torch.argmax(pr, dim=1)
    else:
        pr = torch.argmax(pr, dim=1)
        gt = torch.argmax(gt, dim=1)

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt == pr)
    # score = torch.true_divide(tp, gt.view(-1).shape[0])
    score = torch.div(tp, gt.view(-1).shape[0])
    return score


def precision(pr, gt, onehot=False, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    if not onehot:
        pr = torch.argmax(pr, dim=1)
    else:
        pr = torch.argmax(pr, dim=1)
        gt = torch.argmax(gt, dim=1)

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score


def recall(pr, gt, onehot=False, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    if not onehot:
        pr = torch.argmax(pr, dim=1)
    else:
        pr = torch.argmax(pr, dim=1)
        gt = torch.argmax(gt, dim=1)

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score


def confusionmatrix(pr, gt, num_classes=25,
                    onehot=False, eps=1e-7, threshold=None, ignore_channels=None):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    pr = _threshold(pr, threshold=threshold)
    if not onehot:
        pr = torch.argmax(pr, dim=1)

    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    stacked = torch.stack(
        (
            gt, pr
        ), dim=1
    )

    for p in stacked:
        tl, pl = p.tolist()
        conf[tl, pl] = conf[tl, pl] + 1
    # tp = torch.sum(gt * pr)
    # fn = torch.sum(gt) - tp
    #
    # score = (tp + eps) / (tp + fn + eps)
    return conf
