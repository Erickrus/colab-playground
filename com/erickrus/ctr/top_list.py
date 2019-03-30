import math
import numpy as np

class TopList:

  def gravity_score(self, post, currTime, gravity=1.8):
    """
    post.clickNum:            P表示物品得票数；
    post.postTime - currTime: T表示物品已发布时间（单位为小时），加上2是为了防止最新的物品导致分母过小；
    gravity:                  G表示“重力因子”，即将物品排名往下拉的力量。
    """
    return (post.clickNum - 1.0) / (currTime - post.postTime + 2.0) ** gravity


  # http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_newton_s_law_of_cooling.html
  def exponential_decay_score(self, post, currTime, alpha=0.192):
    """
    函数基于 Newton's Law of cooling
    本期得分 = 上一期得分 x exp(-(冷却系数) x 间隔的小时数) 
    冷却系数 是一个你自己决定的值。如果假定一篇新文章的初始分数是100分，24小时之后"冷却"为1分，那么可以计算得到"冷却系数"约等于0.192。如果你想放慢"热文排名"的更新率，"冷却系数"就取一个较小的值，否则就取一个较大的值。
    """
    return post.clickNum * math.exp(- alpha * (currTime - post.postTime))


  # the script reference following site
  # http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_bayesian_average.html
  def bayesian_average(self, x, C, m):
    """
    贝叶斯平均主要解决的是如果被评选的项目数据较少的问题
    用一个先验概率放在里面作为辅助，得到一个调整后的(加权)值

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



if __name__ == "__main__":
  from com.erickrus.ctr.post import Post
  p = Post(500)
  tList = TopList()
  print(tList.gravity_score(p, 20))
  print(tList.gravity_score(p, 21))

  print(tList.exponential_decay_score(p, 20))
  print(tList.exponential_decay_score(p, 21))

  ba = BayesianAverage()
  print(ba.average(np.array(list(range(20))),100,5))