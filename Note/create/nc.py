class compiler:
    def __init__(self,filename):
        self.filename=filename
        self.init={'z':'tf.zeros(','n':'tf.random.normal(','u':'tf.random.uniform('}
        self.oj1=['','','','','']
        self.oj2=['','']
        self.oj3=''
        self.line=''
    
    
    def tf_function(self,oj1=None,oj2=None,oj3=None,init=None):
        if oj1!=None:
            return oj1[0]+'='+self.init[init]+oj1[1]+oj1[2]+oj1[3]+oj1[4]+')'
        if oj2!=None:
            return 'tf.matmul('+oj2[0]+','+oj2[1]+')'
        elif oj3!=None:
            return 'tf.reverse('+oj3+')'
    
    
    def getchar(self,line):
        self.line=line
        for i in range(len(line)):
            if self.line[i]=='.' and self.line[i+1]==' ':
                if self.line[i-4]=='z' or self.line[i-4]=='n' or self.line[i-4]=='u':
                    init=self.line[i-4]
                    self.oj1[2]=self.line[i-3:i]
                    index=i-4
                else:
                    init=self.line[i-6]
                    self.oj1[2]=self.line[i-5:i]
                    index=i-6
                index1=i+1
            elif self.line[i]=='[':
                self.oj1[0]=self.line[index1+1:i]
                index2=i
            elif self.line[i]==']':
                self.oj1[1]=self.line[index2:]+','
            elif self.line[i]=='(':
                self.oj1[3]=self.line[i+1]+','
            elif self.line[i]==')':
                self.oj1[4]=self.line[i-1]
            else:
                if self.oj1[3]!='':
                    self.oj1[2]+=','
                line=self.tf_function(oj1=self.oj1,init=init)
                self.line=self.line.replace(self.line[index-1:],line)
                self.oj1=['','','','','']
            if self.line[i]=='(':
                index1=i
                continue
            elif self.line[i]=='.':
                index2=i
                continue
            elif self.line[i]==')':
                index3=i
                if '.*' in self.line[index1:index2]:
                    self.oj1[0]=self.line[index1+1:index2]
                    self.oj2[1]=self.line[index2+2:index3]
                    string1=self.line[:index1]
                    string2=self.line[index3+1:]
                    string=self.tf_function(self,oj2=self.oj2)
                    self.line=string1+string+string2
                if '.^' in self.line[index1:index2]:
                    self.oj2=self.line[index1+1:index2]
                    string1=self.line[:index1]
                    string2=self.line[index3+1:]
                    string=self.tf_function(self,oj3=self.oj3)
                    self.line=string1+string+string2
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
                if '. ' in line or '.*'  in line or '.^' in line:
                    self.readlines(line)
                    outfile.write(self.line)
                    self.line=''
                elif '#' in line:
                    outfile.write(line)
                elif line=='"""':
                    flag=False
                    outfile.write(line)
                elif flag==False:
                    outfile.write(line)
                elif line=='"""' and flag==False:
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