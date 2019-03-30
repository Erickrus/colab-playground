import numpy as np

class BayesianAverage:

  def average(self, x, C, m):
    """
    先估计一个值，然后不断用新的信息修正，使得它越来越接近正确的值，
    如果投票人数少，总分接近全局平均分，
    如果投票人数多，总分接近项目平均分

    n是该项目的投票人数；
    x是该项目的每张选票的值；
    m是平均投票得分；
    C是平均投票人数
    """
    xBar = (C*m + np.sum(x)) / (C + x.size)
    return xBar

