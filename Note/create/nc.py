class compiler:
    def __init__(self,filename):
        self.filename=filename
        self.init={'z':'tf.zeros(','n':'tf.random.normal(','u':'tf.random.uniform(','o':'tf.ones('}
        self.operator=['.*','.^','.|','.||','./','.=']
        self.oj1=['','','','']
        self.oj2=['','']
        self.oj3=''
        self.oj4=['','','']
        self.oj5=''
        self.oj6=['','']
        self.line=''
    
    
    def tf_function(self,oj1=None,oj2=None,oj3=None,oj4=None,oj5=None,oj6=None,init=None):
        if oj1!=None:
            return self.init[init]+oj1[0]+oj1[1]+oj1[2]+oj1[3]+')'
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
    
    
    def concat(self,string_list):
        line=''
        for i in range(len(string_list)):
            if i!=len(string_list)-1:
                if i==0:
                    line+=self.line[string_list[i][0]-1:]+self.line.replace(self.line[string_list[i][0]:],string_list[i][2])+self.line[string_list[i][1]+1:string_list[i+1][0]]
                else:
                    line+=string_list[i][2]+self.line[string_list[i][1]+1:string_list[i+1][0]]
            else:
                line+=string_list[i][2]+self.line[string_list[i][1]+1:]
        self.line=line
        return
    
    
    def getchar(self,line):
        self.line=line
        string_list=[]
        for i in range(len(line)):
            if self.line[i]=='.' and self.line[i+1]==' ':
                if '=' in self.line:
                    indexf=self.line.find('=')+1
                    init=self.line[indexf]
                else:
                    indexf=self.line.find('r')+7
                    init=self.line[indexf]
                self.oj1[1]=self.line[indexf+1:i]
            elif self.line[i]=='[':
                index1=i
            elif self.line[i]==']':
                self.oj1[0]=self.line[index1:i+1]+','
                indexl=i
            elif self.line[i]=='(':
                self.oj1[2]=self.line[i+1]+','
            elif self.line[i]==')':
                self.oj1[3]=self.line[i-1]
                indexl=i
            else:
                if self.oj1[2]!='':
                    self.oj1[1]+=','
                line=self.tf_function(oj1=self.oj1,init=init)+self.index[indexl+1:]
                self.line=self.line.replace(self.line[indexf:],line)
                self.oj1=['','','','']
            if self.line[i]=='(':
                index1=i
                continue
            elif self.line[i]=='.':
                index2=i
                continue
            elif self.line[i]==')' and index1!=None:
                index3=i
                if '.*' in self.line[index1:index3+1]:
                    self.oj1[0]=self.line[index1+1:index2]
                    self.oj2[1]=self.line[index2+2:index3]
                    string=self.tf_function(oj2=self.oj2)
                    string_list.append([index1,index3,string])
                if '.^' in self.line[index1:index3+1]:
                    self.oj2=self.line[index1+1:index2]
                    string=self.tf_function(oj3=self.oj3)
                    string_list.append([index1,index3,string])
                if '.|' in self.line[index1:index3+1]:
                    self.oj4[0]=self.line[index1+1:index2]
                    self.oj4[1]=self.line[index2+2:index3]
                    self.oj4[2]='.|'
                    string=self.tf_function(oj4=self.oj4)
                    string_list.append([index1,index3,string])
                if '.||' in self.line[index1:index3+1]:
                    self.oj4[0]=self.line[index1+1:index2]
                    self.oj4[1]=self.line[index2+3:index3]
                    self.oj4[2]='.||'
                    string=self.tf_function(oj4=self.oj4)
                    string_list.append([index1,index3,string])
                if './' in self.line[index1:index3+1]:
                    self.oj5=self.line[index1+1:index2]
                    string=self.tf_function(oj5=self.oj5)
                    string_list.append([index1,index3,string])
                if '.=' in self.line[index1:index3+1]:
                    self.oj6[0]=self.line[index1+1:index2]
                    self.oj6[1]=self.line[index2+2:index3]
                    string=self.tf_function(oj6=self.oj6)
                    string_list.append([index1,index3,string])
                index1=None
        self.concat(string_list)
        return
    
    
    def readlines(self,line):
        return self.getchar(line)
    
    
    def writelines(self):
        outfile=self.filename
        outfile.replace(outfile[-2:],'py')
        outfile=open(outfile,'w')
        with open(self.filename) as infile:
            while 1:
                line=infile.readline()
                if ('. ' in line and "'" not in line or '"' not in line) or ("'" not in line and True in [operator in line for operator in self.operator]):
                    self.readlines(line)
                    outfile.write(self.line)
                    self.line=''
                elif '#' in line:
                    outfile.write(line)
                elif line=='"""' or line=="'''":
                    flag=False
                    outfile.write(line)
                elif flag==False:
                    outfile.write(line)
                elif flag==False and line=='"""' or  line=="'''":
                    flag=True
                    outfile.write(line)
                elif line=='':
                    break
                else:
                    outfile.write(line)
            outfile.close()
        return
    
    
    def Compile(self):
        self.writelines()
        return
