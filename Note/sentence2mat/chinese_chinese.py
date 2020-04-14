import numpy as np
import jieba


def one_hot(sentence_list_Q,sentence_list_A,sentence_c_sum_Q,sentence_c_sum_A,word_list_Q,word_list_A,word_sum,word_table):
    sentence_Q=np.zeros([len(sentence_list_Q),sentence_c_sum_Q,word_sum],dtype=np.float32)
    sentence_A=np.zeros([len(sentence_list_A),sentence_c_sum_A,word_sum],dtype=np.float32)
    for i in range(len(sentence_Q)):
        for j in range(len(word_list_Q[i])):
            word=word_list_Q[i][j]
            sentence_Q[i][j][word_table[word]-1]=1
    for i in range(len(sentence_A)):
        for j in range(len(word_list_A[i])):
            word=word_list_A[i][j]
            sentence_A[i][j][word_table[word]-1]=1
    return sentence_Q,sentence_A
        
    
def s2m(path,endsign=None):
    flag1=0
    flag2=2
    sentence_list=[]
    sentence_list_Q=[]
    sentence_list_A=[]
    divide_list_Q=[]
    divide_list_A=[]
    word_list_Q=[]
    word_list_A=[]
    word_lib_Q=set()
    word_lib_A=set()
    word_lib=set()
    word_table=dict()
    one_hot_word_table=dict()
    file=open(path,'r')
    for line in file:
        sentence_list.append(line.strip())
        if line.strip()=='':
            sentence_list.pop()
        if flag2==2:
            sentence_list_Q.append(line.strip())
            flag2=0
            if line.strip()=='':
                sentence_list_Q.pop()
            continue
        if flag2==0:
            flag2+=1
            continue
        if flag2==1:
            flag2+=1
    sentence_c_sum_Q=0
    sentence_c_sum_A=0
    for i in range(len(sentence_list)):
        if flag1==0:
            if sentence_c_sum_Q<len(sentence_list[i]):
                sentence_c_sum_Q=len(sentence_list[i])
                sentence_c_sum_Q=len(sentence_list[i])
            else:
                sentence_c_sum_Q=sentence_c_sum_Q
            dw=''
            dw=jieba.cut(sentence_list[i],cut_all=False)
            dw='/'.join(dw)
            dw=dw+'/'
            divide_list_Q.append(dw)
            dw=''
            flag1=1
            continue
        if flag1==1:
            if sentence_c_sum_A<len(sentence_list[i]):
                sentence_c_sum_A=len(sentence_list[i])+1
                sentence_c_sum_A=len(sentence_list[i])
            else:
                sentence_c_sum_A=sentence_c_sum_A+1
            sentence_list_A.append(sentence_list[i])
            dw=jieba.cut(sentence_list[i],cut_all=False)
            dw='/'.join(dw)
            dw=dw+'/'
            divide_list_A.append(dw)
            dw=''
            flag1=0
    for i in range(len(divide_list_Q)):
        word=''
        word_list_Q.append([])
        for j in range(len(divide_list_Q[i])):
            if divide_list_Q[i][j]=='/':
                word_list_Q[i].append(word)
                word_lib_Q.add(word)
                word_lib.add(word)
                word=''
                continue
            word=word+divide_list_Q[i][j]
    for i in range(len(divide_list_A)):
        word=''
        word_list_A.append([])
        for j in range(len(divide_list_A[i])):
            if divide_list_A[i][j]=='/':
                word_list_A[i].append(word)
                word_lib_A.add(word)
                word_lib.add(word)
                word=''
                continue
            word=word+divide_list_A[i][j]
        word_list_A[i].append(endsign)
    word_lib.add(endsign)
    word_sum=len(word_lib)
    for word in word_lib:
        if word==endsign:
            continue
        word_table[word]=len(word_table)+1
    word_table[endsign]=len(word_table)+1
    for word in word_lib:
        if word==endsign:
            continue
        one_hot_word_table[word]=np.zeros([word_sum],dtype=np.float32)
        one_hot_word_table[word][word_table[word]-1]=1
    one_hot_word_table[endsign]=np.zeros([word_sum])
    one_hot_word_table[endsign][word_table[endsign]-1]=1
    sentence_Q,sentence_A=one_hot(sentence_list_Q,sentence_list_A,sentence_c_sum_Q,sentence_c_sum_A,word_list_Q,word_list_A,word_sum,word_table)
    return sentence_Q,sentence_A,word_table                