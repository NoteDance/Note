"""
This example demonstrates how to use Note's adabatch function.
"""
from Note import nn

class Model(nn.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.conv2d(32, 3, activation='relu')
        self.flatten = nn.flatten()
        self.d1 = nn.dense(128, activation='relu')
        self.d2 = nn.dense(10)
    
    def batch_size_fn(self,train_ds):
        if self.batch_counter%777==0:
            return self.adabatch(train_ds,7)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)