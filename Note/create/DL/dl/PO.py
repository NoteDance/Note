import numpy as np


class PO:
    def __init__(self,stop_func=None,attenuate=None,thread=None,row=None,grad=None):
        self.stop_func=stop_func
        self.attenuate=attenuate
        self.thread=thread
        self.row=row
        self.ln_list=[]
        self.grad=grad
    
    
    def PO(self,fp,gradient,opt,data,labels,param,PO,thread_lock,t=None,platform='tensorflow',opt_counter=None,gradient_lock=None,gradient_list=None):
        if platform=='tensorflow':
            try:
                tape,output,loss=fp(data,labels)
            except:
                tape,output,loss=fp(data,labels,t)
            if self.attenuate!=None:
                opt_counter[t]=0
            if PO==1:
                thread_lock.acquire()
                if self.stop_func!=None and self.stop_func(thread_lock):
                    return 0,0
                gradient=gradient(tape,loss,param)
                if self.attenuate!=None:
                    gradient=self.attenuate(gradient,opt_counter[t])
                try:
                    opt(gradient,param)
                except:
                    opt(gradient,param,t)
                if self.attenuate!=None:
                    opt_counter+=1
                thread_lock.release()
            elif PO==2:
                thread_lock[0].acquire()
                if self.stop_func!=None and self.stop_func(thread_lock[0]):
                    return 0,0
                self.gradient=gradient(tape,loss,param)
                if self.attenuate!=None:
                    self.gradient=self.attenuate(self.gradient,opt_counter[t])
                thread_lock[0].release()
                thread_lock[1].acquire()
                if self.stop_func!=None and self.stop_func(thread_lock[1]):
                    return 0,0
                try:
                    opt(gradient,param)
                except:
                    opt(gradient,param,t)
                if self.attenuate!=None:
                    opt_counter+=1
                thread_lock[1].release()
            elif PO==3:
                if self.row==None and len(gradient_lock)==self.thread:
                    ln=t
                else:
                    if self.row!=None:
                        while True:
                            rank_index=np.random.choice(len(gradient_lock))
                            row_index=np.random.choice(len(gradient_lock[rank_index]))
                            if [rank_index,row_index] in self.ln_list:
                                continue
                            else:
                                break
                    else:
                        while True:
                            ln=np.random.choice(len(gradient_lock))
                            if ln in self.ln_list:
                                continue
                            else:
                                break
                gradient_lock[ln].acquire()
                if self.row!=None:
                    gradient_lock[rank_index][row_index].acquire()
                    if self.stop_func!=None and self.stop_func(gradient_lock[rank_index][row_index]):
                        return 0,0
                    self.ln_list.append([rank_index,row_index])
                else:
                    gradient_lock[ln].acquire()
                    if self.stop_func!=None and self.stop_func(gradient_lock[ln]):
                        return 0,0
                    self.ln_list.append(ln)
                gradient=gradient(tape,loss,param)
                if self.attenuate!=None:
                    gradient=self.attenuate(gradient,opt_counter[t])
                if self.row!=None:
                    self.ln_list.remove([rank_index,row_index])
                    self.gradient_lock[rank_index][row_index].release()
                else:
                    self.ln_list.remove(ln)
                    self.gradient_lock[ln].release()
                self.thread_lock.acquire()
                if self.stop_func!=None and self.stop_func(thread_lock):
                    return 0,0
                try:
                    opt(gradient,param)
                except:
                    opt(gradient,param,t)
                if self.attenuate!=None:
                    opt_counter+=1
                thread_lock.release()
            return output,loss
        elif platform=='pytorch':
            try:
                output,loss=fp(data,labels)
            except:
                output,loss=fp(data,labels,t)
            if self.attenuate!=None:
                opt_counter[t]=0
            if self.PO==1:
                thread_lock.acquire()
                if self.stop_func!=None and self.stop_func(thread_lock):
                    return 0,0
                try:
                    opt(loss)
                except:
                    opt(loss,t)
                thread_lock.release()
            elif self.PO==2:
                thread_lock[0].acquire()
                if self.stop_func!=None and self.stop_func(thread_lock[0]):
                    return 0,0
                try:
                    gradient(loss)
                except:
                    gradient(loss,t)
                if self.attenuate!=None:
                    self.attenuate(opt_counter[t])
                thread_lock[0].release()
                thread_lock[1].acquire()
                if self.stop_func!=None and self.stop_func(thread_lock[1]):
                    return 0,0
                try:
                    opt()
                except:
                    opt(t)
                thread_lock[1].release()
            elif self.PO==3:
                if self.row==None and len(gradient_lock)==self.thread:
                    ln=t
                else:
                    if self.row!=None:
                        while True:
                            rank_index=np.random.choice(len(gradient_lock))
                            row_index=np.random.choice(len(gradient_lock[rank_index]))
                            if [rank_index,row_index] in self.ln_list:
                                continue
                            else:
                                break
                    else:
                        while True:
                            ln=np.random.choice(len(gradient_lock))
                            if ln in self.ln_list:
                                continue
                            else:
                                break
                if self.row!=None:
                    gradient_lock[rank_index][row_index].acquire()
                    if self.stop_func!=None and self.stop_func(gradient_lock[rank_index][row_index]):
                        return 0,0
                    self.ln_list.append([rank_index,row_index])
                else:
                    gradient_lock[ln].acquire()
                    if self.stop_func!=None and self.stop_func(gradient_lock[ln]):
                        return 0,0
                    self.ln_list.append(ln)
                try:
                    gradient(loss)
                except:
                    gradient(loss,t)
                gradient_list[t]=self.grad()
                if self.attenuate!=None:
                    self.attenuate(opt_counter[t],gradient_list[t])
                if self.row!=None:
                    self.ln_list.remove([rank_index,row_index])
                    gradient_lock[rank_index][row_index].release()
                else:
                    self.ln_list.remove(ln)
                    gradient_lock[ln].release()
                thread_lock.acquire()
                if self.stop_func!=None and self.stop_func(thread_lock):
                    return 0,0
                try:
                    opt(gradient_list[t])
                except:
                    opt(gradient_list[t],t)
                gradient_list[t]=None
                if self.attenuate!=None:
                    opt_counter+=1
                thread_lock.release()
            return output,loss