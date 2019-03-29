#@title Mount google Drive

import sys
import os

from google.colab import drive

class ColabPlayground:
  def __init__(self):
    self.repoName = 'colab-playground'
    self.notebookPath = '/content/drive/My Drive/Colab Notebooks'
    self.githubUrl = 'https://github.com/Erickrus'
    self.localRepoPath = os.path.join(self.notebookPath, self.repoName)

  def pull(self):
    if not os.path.exists('%s' % self.localRepoPath):
      os.system('cd "%s" && git clone %s/%s' % (self.notebookPath, self.githubUrl, self.repoName))
    else:
      os.system('cd "%s" && git pull' % self.localRepoPath)
    return self

  def register_path(self):
    sys.path.append(self.localRepoPath)
    return self

  def mount(self):
    try:
      drive.mount('/content/drive', force_remount=True)
    except:
      pass
    return self

def init_playground():
  playground = ColabPlayground()
  playground.mount().pull().register_path()

