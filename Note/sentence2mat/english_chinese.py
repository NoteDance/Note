import numpy as np
import jieba


def one_hot(original_sentence_list,translation_sentence_list,original_sentence_c_sum,translation_sentence_c_sum,original_word_list,translation_word_list,word_table_original,word_table_translation,word_sum):
    original_sentence=np.zeros([len(original_sentence_list),original_sentence_c_sum,word_sum],dtype=np.float32)
    translation_sentence=np.zeros([len(translation_sentence_list),translation_sentence_c_sum,word_sum],dtype=np.float32)
    for i in range(len(original_sentence)):
        for j in range(len(original_word_list[i])):
            word=original_word_list[i][j]
            original_sentence[i][j][word_table_original[word]-1]=1
    for i in range(len(translation_sentence)):
        for j in range(len(translation_word_list[i])):
            word=translation_word_list[i][j]
            translation_sentence[i][j][word_table_translation[word]-1]=1
    original_sentence=original_sentence
    translation_sentence=translation_sentence
    return original_sentence,translation_sentence
        
    
def s2m(path,endsign=None):
    flag1=0
    flag2=2
    sentence_list=[]
    original_sentence_list=[]
    translation_sentence_list=[]
    translation_divide_list=[]
    original_word_list=[]
    translation_word_list=[]
    word_lib_original=set()
    word_lib_translation=set()
    word_table_original=dict()
    word_table_translation=dict()
    one_hot_word_table_original=dict()
    one_hot_word_table_translation=dict()
    file=open(path,'r')
    for line in file:
        sentence_list.append(line.strip())
        if line.strip()=='':
            sentence_list.pop()
        if flag2==2:
            translation_sentence_list.append(line.strip())
            flag2=0
            if line.strip()=='':
                translation_sentence_list.pop()
            continue
        if flag2==0:
            flag2+=1
            continue
        if flag2==1:
            flag2+=1
    original_sentence_c_sum=0
    translation_sentence_c_sum=0
    for i in range(len(sentence_list)):
        if flag1==0:
            if translation_sentence_c_sum<len(sentence_list[i]):
                translation_sentence_c_sum=len(sentence_list[i])
                translation_sentence_c_sum=len(sentence_list[i])
            else:
                translation_sentence_c_sum=translation_sentence_c_sum
            dw=''
            dw=jieba.cut(sentence_list[i],cut_all=False)
            dw='/'.join(dw)
            dw=dw+'/'
            translation_divide_list.append(dw)
            dw=''
            flag1=1
            continue
        if flag1==1:
            if original_sentence_c_sum<len(sentence_list[i]):
                original_sentence_c_sum=len(sentence_list[i])
                original_sentence_c_sum=len(sentence_list[i])
            else:
                original_sentence_c_sum=original_sentence_c_sum
            original_sentence_list.append(sentence_list[i])
            flag1=0
    for i in range(len(translation_divide_list)):
        word=''
        translation_word_list.append([])
        for j in range(len(translation_divide_list[i])):
            if translation_divide_list[i][j]=='/':
                translation_word_list[i].append(word)
                word_lib_translation.add(word)
                word=''
                continue
            word=word+translation_divide_list[i][j]
        translation_word_list[i].append(endsign)
    for i in range(len(original_sentence_list)):
        word=''
        original_word_list.append([])
        for j in range(len(original_sentence_list[i])):
            if original_sentence_list[i][j]==' ' or original_sentence_list[i][j] in [',','.','?','!'] or j==len(original_sentence_list[i])-1:
                if j==len(original_sentence_list[i])-1 and original_sentence_list[i][j] not in [',','.','?','!']:
                    word=word+original_sentence_list[i][j]
                    original_word_list[i].append(word)
                    word_lib_original.add(word)
                    break
                if word=='':
                    continue
                original_word_list[i].append(word)
                word_lib_original.add(word)
                word=''
                continue
            word=word+original_sentence_list[i][j]
    original_word_list.append([',','.','?','!'])
    word_lib_original.add(',')
    word_lib_original.add('.')
    word_lib_original.add('?')
    word_lib_original.add('!')
    if len(word_lib_original)>len(word_lib_translation):
        word_sum=len(word_lib_original)
    else:
        word_sum=len(word_lib_translation)
    for word in word_lib_original:
        word_table_original[word]=len(word_table_original)+1
        one_hot_word_table_original[word]=np.zeros([word_sum],dtype=np.float32)
        one_hot_word_table_original[word][word_table_original[word]-1]=1
    for word in word_lib_translation:
        if word==endsign:
            continue
        word_table_translation[word]=len(word_table_translation)+1
        one_hot_word_table_translation[word]=np.zeros([word_sum],dtype=np.float32)
        one_hot_word_table_translation[word][word_table_translation[word]-1]=1
    word_table_translation[endsign]=len(word_table_translation)+1
    one_hot_word_table_translation[endsign]=np.zeros([word_sum])
    one_hot_word_table_translation[endsign][word_table_translation[endsign]-1]=1
    original_sentence,translation_sentence=one_hot(original_sentence_list,translation_sentence_list,original_sentence_c_sum,translation_sentence_c_sum,original_word_list,translation_word_list,word_table_original,word_table_translation,word_sum)
    return original_sentence,translation_sentence,word_table_original,word_table_translation                