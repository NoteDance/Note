from Note import nn
import multiprocessing
from functools import partial


def epoch_end_callback(epoch, logs, model, lock, callback_func):
    callback_func(epoch, logs, model, lock)
    

class ParallelFinder:
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        manager = multiprocessing.Manager()
        self.logs = manager.dict()
        self.logs['best_loss'] = 1e9
        self.logs['best_time'] = 1e9
        self.lock = multiprocessing.Lock()

    def on_epoch_end(self, epoch, logs, model=None, lock=None):
        lock.acquire()
        loss = logs['loss']
        
        if epoch+1 == self.epochs:
            if loss < self.logs['best_loss']:
                self.logs['best_loss_model'] = model
                self.logs['best_loss'] = loss
                self.logs['time'] = model.time
            if model.time < self.logs['best_time']:
                self.logs['best_time_model'] = model
                self.logs['best_time'] = model.time
                self.logs['loss'] = model.train_loss
        lock.release()

    def find(self, train_ds=None, loss_object=None, train_loss=None, strategy=None, batch_size=64, epochs=1, jit_compile=True):
        self.epochs = epochs

        process_list=[]
        for i in range(len(self.models)):
            partial_callback = partial(
                epoch_end_callback,
                model=self.models[i],
                lock=self.lock,
                callback_func=self.on_epoch_end
            )
            callback = nn.LambdaCallback(on_epoch_end=partial_callback)
            self.models[i].optimizer = self.optimizers[i]
            if strategy == None:
                process=multiprocessing.Process(target=self.models[i].train,kwargs={
                                                        'train_ds': train_ds,
                                                        'loss_object': loss_object,
                                                        'train_loss': train_loss,
                                                        'epochs': epochs,
                                                        'callbacks': [callback],
                                                        'jit_compile': jit_compile,
                                                        'p': 0
                                                    })
                process.start()
                process_list.append(process)
            else:
                callback = nn.LambdaCallback(on_epoch_end=lambda epoch, logs: self.on_epoch_end(epoch, logs, self.models[i], self.lock))
                self.models[i].optimizer = self.optimizers[i]
                process=multiprocessing.Process(target=self.models[i].distributed_training,kwargs={
                                                        'train_dataset': train_ds,
                                                        'loss_object': loss_object,
                                                        'global_batch_size': batch_size,
                                                        'epochs': epochs,
                                                        'strategy': strategy,
                                                        'callbacks': [callback],
                                                        'jit_compile': jit_compile,
                                                        'p': 0
                                                    })
                process.start()
                process_list.append(process)
        for process in process_list:
            process.join()
