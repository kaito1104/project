import pandas as pd
import csv
 
df=pd.read_csv('data.csv')
df2=df['seqence']
df3=df['label']
#print(df2)
seq=[]
seq1=[]
seq2=[]

with open('encode_data.csv','w',newline="") as f:
 csv_writer = csv.writer(f)
 csv_writer.writerow(['encode1','encode2','encode3','encode4','label'])
 #df3.to_csv('encode3_data.csv', columns=['value'], index=None)
 #print(df3)
 for i in range(len(df2)):
   seq.append(df2[i])
   for z in range(4):
       if seq[0][z]=='A':
          seq1.append(1)
          
       elif seq[0][z]=='T':
          seq1.append(2)
         
       elif seq[0][z]=='G':
          seq1.append(3)
          
       elif seq[0][z]=='C':
          seq1.append(4)
   seq1.append(df3[i])
   print(seq1)
   csv_writer.writerow(seq1)
   #print(type(seq[0][0]))
   #print(''.join(seq1))
   seq1.clear()
   seq.clear()
#df3.to_csv('encode3_data.csv', columns=['value'], index=None)
f.close()

#with open('encode3_data.csv','a') as f:
 #csv_writer = csv.writer(f)
 #csv_writer.writerow([df3])
#f.close()
    