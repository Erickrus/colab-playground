import math

class TopList:
  def gravity_score(self, post, currTime, gravity=1.8):
    """
    post.clickNum:            P表示物品得票数；
    post.postTime - currTime: T表示物品已发布时间（单位为小时），加上2是为了防止最新的物品导致分母过小；
    gravity:                  G表示“重力因子”，即将物品排名往下拉的力量。
    """
    return (post.clickNum - 1.0) / (currTime - post.postTime + 2.0) ** gravity

  # http://www.ruanyifeng.com/blog/2012/03/ranking_algorithm_newton_s_law_of_cooling.html
  def exponential_decay(self, post, currTime, alpha=0.192):
    """
    本期得分 = 上一期得分 x exp(-(冷却系数) x 间隔的小时数) 
    冷却系数 是一个你自己决定的值。如果假定一篇新文章的初始分数是100分，24小时之后"冷却"为1分，那么可以计算得到"冷却系数"约等于0.192。如果你想放慢"热文排名"的更新率，"冷却系数"就取一个较小的值，否则就取一个较大的值。
    """
    return post.clickNum * math.exp(- alpha * (currTime - post.postTime))

