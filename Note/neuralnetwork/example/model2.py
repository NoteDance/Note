from Note import nn

class Model(nn.Model):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential()
    self.layers.add(nn.conv2d(32, 3, activation='relu'))
    self.layers.add(nn.max_pool2d())
    self.layers.add(nn.conv2D(64, 3, activation='relu'))
    self.layers.add(nn.max_pool2d())
    self.layers.add(nn.flatten())
    self.layers.add(nn.dense(64, activation='relu'))
    self.layers.add(nn.dense(10))

  def __call__(self, x):
    return self.layers(x)