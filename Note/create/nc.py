class compiler:
    def __init__(self,filename):
        self.filename=filename
        self.init={'z':'tf.zeros(','n':'tf.random.normal(','u':'tf.random.uniform(','o':'tf.ones('}
        self.operator=['.*','.^','.|','.||','./','.=']
        self._operator=['*','^','|','||','/','=']
        self.define=dict()
        self.define_list=None
        self.index_list=None
        self.oj1=['','','','']
        self.oj2=['','']
        self.oj3=''
        self.oj4=['','','']
        self.oj5=''
        self.oj6=['','']
        self.line=''
        self.test=[]
    
    
    def tf_function(self,oj1=None,oj2=None,oj3=None,oj4=None,oj5=None,oj6=None,init=None):
        if oj1!=None:
            return self.init[init]+oj1[0]+oj1[2]+oj1[3]+'dtype='+oj1[1]+')'
        if oj2!=None:
            return 'tf.matmul('+oj2[0]+','+oj2[1]+')'
        elif oj3!=None:
            return 'tf.reverse('+oj3+')'
        elif oj4!=None:
            if oj4[2]=='.|':
                return 'tf.concat('+'['+oj4[0]+','+oj4[1]+']'+','+'0'+')'
            else:
                return 'tf.concat('+'['+oj4[0]+','+oj4[1]+']'+','+'1'+')'
        elif oj5!=None:
            return 'tf.math.sqrt('+oj5+')'
        elif oj6!=None:
            return 'state_ops.assign('+oj6[0]+','+oj6[1]+')'
    
    
    def getchar(self,line):
        self.line=line
        flag=None
        if len(self.define)!=0:
            index=[define in line for define in self.define]
            for i in range(len(index)):
                if index[i]==True:
                    while 1:
                        if self.define_list[i] in line:
                            line=line.replace(self.define_list[i],self.define[self.define_list[i]])
                        else:
                            break
        for i in range(len(line)):
            try:
                if line[i]=='.' and line[i+1]==' ':
                    if 'tf.Variable' not in line and '=' in line:
                        indexf=line.find('=')+1
                        init=line[indexf]
                    elif 'tf.Variable' in line:
                        indexf=line.find('(')+1
                        init=line[indexf]
                    else:
                        indexf=line.find('r')+7
                        init=line[indexf]
                    self.oj1[1]='tf.'+line[indexf+1:i]
                    flag=True
                    continue
            except IndexError:
                pass
            if flag==True:
                if line[i]=='[':
                    index1=i
                elif line[i]==']':
                    self.oj1[0]=line[index1:i+1]+','
                    indexl=i
                elif line[i]=='(':
                    if line[i+1]!=init:
                        self.oj1[2]=line[i+1]+','
                elif line[i]==')':
                    if line[i+1]!='\n':
                        self.oj1[3]=line[i-1]
                        indexl=i
                    string=self.tf_function(oj1=self.oj1,init=init)
                    self.line=self.line.replace(line[indexf:indexl+1],string)
                    self.oj1=['','','','']
                    flag=False
                continue
            if line[i]=='(':
                index1=i
                continue
            elif line[i]=='.' and (line[i+1] in self._operator or line[i+1:i+3] in self._operator):
                index2=i
                continue
            elif line[i]==')':
                try:
                    if line[index2+2:i]==line[index2+2:i] or line[index2+3:i]==line[index2+2:i]:
                        index3=i
                except IndexError:
                    pass
                if '.*' in line[index1:index3+1]:
                    self.oj2[0]=line[index1+1:index2]
                    self.oj2[1]=line[index2+2:index3]
                    string=self.tf_function(oj2=self.oj2)
                    self.line=self.line.replace(line[index1:index3+1],string)
                if '.^' in line[index1:index3+1]:
                    self.oj2=line[index1+1:index2]
                    string=self.tf_function(oj3=self.oj3)
                    self.line=self.line.replace(line[index1:index3+1],string)
                if '.|' in line[index1:index3+1]:
                    self.oj4[0]=line[index1+1:index2]
                    self.oj4[1]=line[index2+2:index3]
                    self.oj4[2]='.|'
                    string=self.tf_function(oj4=self.oj4)
                    self.line=self.line.replace(line[index1:index3+1],string)
                if '.||' in line[index1:index3+1]:
                    self.oj4[0]=line[index1+1:index2]
                    self.oj4[1]=line[index2+3:index3]
                    self.oj4[2]='.||'
                    string=self.tf_function(oj4=self.oj4)
                    self.line=self.line.replace(line[index1:index3+1],string)
                if './' in line[index1:index3+1]:
                    self.oj5=line[index1+1:index2]
                    string=self.tf_function(oj5=self.oj5)
                    self.line=self.line.replace(line[index1:index3+1],string)
                if '.=' in line[index1:index3+1]:
                    self.oj6[0]=line[index1+1:index2]
                    self.oj6[1]=line[index2+2:index3]
                    string=self.tf_function(oj6=self.oj6)
                    self.line=self.line.replace(line[index1:index3+1],string)
        return
    
    
    def readlines(self,line):
        self.getchar(line)
        return
    
    
    def writelines(self):
        flag=None
        outfile=self.filename
        outfile=outfile[:outfile.rfind('.')]+'.py'
        outfile=open(outfile,'w')
        with open(self.filename) as infile:
            while 1:
                line=infile.readline()
                if line=='':
                    break
                if line=='"""' or line=="'''":
                    flag=False
                    outfile.write(line)
                elif flag==False:
                    outfile.write(line)
                    if line=='"""' or line=="'''":
                        flag=True
                    continue
                if 'define. ' in line:
                    self.define[line[line.find(' ')+1,line.rfine(' ')]]=line[line.rfind(' ')+1:]
                    continue
                if len(self.define)!=0 and self.define_list==None:
                    self.define_list=[define for define in self.define]
                if ('. ' in line and ("'" not in line or '"' not in line)) or ("'" not in line and True in [operator in line for operator in self.operator]):
                    self.readlines(line)
                    outfile.write(self.line)
                    self.line=''
                elif '#' in line:
                    outfile.write(line)
                else:
                    outfile.write(line)
            outfile.close()
        return
    
    
    def Compile(self):
        self.writelines()
        return
