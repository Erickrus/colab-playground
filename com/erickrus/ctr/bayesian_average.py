import numpy as np

class BayesianAverage:
  # the script reference following site
  # http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_bayesian_average.html
  def average(self, x, C, m):
    """
    先估计一个值，然后不断用新的信息修正，使得它越来越接近正确的值，
    如果投票人数少，总分接近全局平均分，
    如果投票人数多，总分接近项目平均分

    n是该项目的投票人数；
    x是该项目的每张选票的值；
    m是平均投票得分；
    C是平均投票人数

    此处的公式是一种一般形式的贝叶斯平均公式，之前文中提到的imdb分值也是这个公式的一种变型
    m（总体平均分）是"先验概率"，每一次新的投票都是一个调整因子，使总体平均分不断向该项目的真实投票结果靠近。
    投票人数越多，该项目的"贝叶斯平均"就越接近算术平均，对排名的影响就越小。
    因此，这种方法可以给一些投票人数较少的项目，以相对公平的排名。

    C*m    /        C = 历史上所有得分 / 历史上所有投票次数
    sum(x) / count(x) = 当前所有得分   / 当前所有投票次数

    缺点，主要问题是它假设用户的投票是正态分布
    """
    xBar = (C*m + np.sum(x)) / (C + x.size)
    return xBar

