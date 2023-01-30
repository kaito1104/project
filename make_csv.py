import random
import csv

y=int(input()) #出力する文字列の長さ
t=int(input()) #出力するデータの数

#A,T,G,Cの中からランダムに出力
chars=('A','T','G','C')
def randchars(chars,length):
    return ''.join(random.choices(chars,k=length))

seq=[]

with open('data.csv','w',newline="") as f:
 csv_writer = csv.writer(f)
 csv_writer.writerow(['seqence','label'])
 for i in range(t):
   seq.append(randchars(chars,y))   
   #csv_writer.writerow(seq)
   if seq[0][1]=='A' and seq[0][2]=='G':
       seq.append('1')
   elif seq[0][1]=='G' and seq[0][2]=='A':
       seq.append('1')
   elif seq[0][1]=='A' and seq[0][2]=='T':
       seq.append('1')
   elif seq[0][1]=='T' and seq[0][2]=='A':
       seq.append('1')
   elif seq[0][1]=='C' and seq[0][2]=='A':
       seq.append('1')
   elif seq[0][1]=='A' and seq[0][2]=='C':
       seq.append('1')
   elif seq[0][1]=='C' and seq[0][2]=='T':
       seq.append('1')
   elif seq[0][1]=='T' and seq[0][2]=='C':
       seq.append('1')
   else: seq.append('0')
   #print(seq)
   csv_writer.writerow(seq)    
       
   seq.clear()
    
f.close()