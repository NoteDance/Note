import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
import time


def write_data(data,path):
    output_file=open(path+'.dat','wb')
    pickle.dump(data,output_file)
    output_file.close()
    return


def read_data(path,dtype=np.float32):
    input_file=open(path+'.dat','rb')
    data=pickle.load(input_file)
    return np.array(data,dtype=dtype)


class K_means:
    def __init__(self,data):
        self.data=data
        self.K=None
        self.cluster_center=None
        self.closest_cluster=None
        self.cost=None
        self.clusters=None
        self.flag=0
    
    
    def cluster(self,K,init_times,json_path=None,csv_path=None,save=False,restore=False):
        t1=time.time()
        if restore==True:
            self.restore(json_path,csv_path)
        self.K=K
        self.closest_cluster=np.zeros([self.K,self.data.shape[0]],dtype=np.int32)
        closest_cluster=np.zeros([self.K,self.data.shape[0]],dtype=np.int32)
        self.clusters=list(range(self.K+1))
        self.cost=np.zeros([self.K])+np.inf
        print()
        for k in range(self.K):
            k=k+1
            for t in range(init_times):
                flag=0
                if restore==True and self.flag==1:
                    cluster_center=self.cluster_center
                    closest_cluster=self.closest_cluster
                    self.flag=0
                else:
                    cluster_center=np.zeros([k,self.data.shape[1]])
                    random=np.random.randint(0,self.data.shape[0],size=[k])
                    cluster_center+=self.data[random]
                if init_times%10!=0:
                    times=init_times-init_times%10
                    times=int(times/10)
                else:
                    times=init_times/10
                if times==0:
                    times=1
                if save==True and t%times*2==0:
                    self.save(json_path,csv_path)
                while 1:
                    for i in range(self.data.shape[0]):
                        closest_cluster[k-1][i]=np.argmin(np.sum(np.square(cluster_center-self.data[i]),axis=1))+1
                    cost=np.sum(np.square(self.data-cluster_center[closest_cluster[k-1]-1]))/self.data.shape[0]
                    for i in range(k):
                        if np.sum(closest_cluster[k-1]==i+1)==0:
                                continue
                        if np.sum((cluster_center[i]-(np.sum(self.data[closest_cluster[k-1]==i+1],axis=0)/np.sum(closest_cluster[k-1]==i+1))))==0:
                            flag+=1
                        else:
                            cluster_center[i]=np.sum(self.data[closest_cluster[k-1]==i+1],axis=0)/np.sum(closest_cluster[k-1]==i+1)
                        if flag==k:
                            if self.cost[k-1]>cost:
                                self.clusters[i+1]=list(range(i+2))
                                j=i+1
                                for i in range(len(self.clusters[i+1])-1):
                                    self.clusters[j][i+1]=self.data[closest_cluster[k-1]==i+1]
                    if flag==k:
                        if self.cost[k-1]>cost:
                            self.cost[k-1]=cost
                            self.cluster_center=cluster_center
                            self.closest_cluster[k-1]=closest_cluster[k-1]
                        break
                    flag=0
            print('{0}-means loss:{1:.3f}'.format(k,self.cost[k-1]))
        t2=time.time()
        self.time=t2-t1
        print()
        print('time:{0:.3f}s'.format(self.time))
        return self.clusters
    
    
    def data_visual(self,data,clusters=None,two=False):
        color=['b', 'g', 'r', 'c', 'm', 'y', 'k','C1','C4','C7']
        if two!=False:
            plt.figure(1)
            pca=PCA(n_components=2)
            data2D=pca.fit_transform(data)
            plt.plot(data2D[:,0],data2D[:,1],'b.')
        if clusters!=None:
            plt.figure(2)
            pca=PCA(n_components=2)
            data2D=pca.fit_transform(data)
            for i in range(len(clusters)-1):
                plt.plot(data2D[self.closest_cluster[len(clusters)-2]==i+1,0],data2D[self.closest_cluster[len(clusters)-2]==i+1,1],color[i]+'.')
        else:
            data2D=pca.fit_transform(data)
            plt.plot(data2D[:,0],data2D[:,1],'b.')
        return
        
        
    def loss_visual(self):
        print()
        plt.plot(np.arange(self.K)+1,self.cost)
        plt.title('K-means loss')
        plt.xlabel('K')
        plt.ylabel('loss')
        for k in range(self.K):
            print('{0}-means loss:{1:.3f}'.format(k+1,self.cost[k]))
        return
    
    
    def info(self):
        print()
        print('K:{0}'.format(self.K))
        print()
        print('cost list:{0:.3f}'.format(self.cost))
        return
    
    
    def save(self,save_path):
        output_file=open(save_path+'.dat','wb')
        pickle.dump(self.K,output_file)
        pickle.dump(self.cluster_center,output_file)
        pickle.dump(self.closest_cluster,output_file)
        output_file.close()
        return
    
    
    def restore(self,save_path):
        input_file=open(save_path+'.dat','rb')
        self.K=pickle.load(input_file)
        self.cluster_center=pickle.load(input_file)
        self.closest_cluster=pickle.load(input_file)
        self.flag=1
        input_file.close()
        return
    
    
    def save_result(self,path):
        for i in range(self.K):
            for j in range(i+1):
                cluster=self.cluster[i+1][j+1]
                output_file=open(path+'/{0}/cluster-{1}.dat'.format(i+1,j+1),'wb')
                pickle.dump(cluster,output_file)
                output_file.close()
        return
