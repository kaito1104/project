import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,confusion_matrix,matthews_corrcoef,roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score


df=pd.read_csv("encode3_data.csv")

training_data=df.drop("label",axis=1)
y=df["label"].values
x=training_data.values
#print(y)
#print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#print(x_train)
#print(y_train)
print(x_test)
print(y_test)
clf=RandomForestClassifier(random_state=42)
clf.fit(x_train,y_train)
pre=clf.predict(x_test)
proba=clf.predict_proba(x_test)
print(pre)

print("元データ:",y_test)
print("予測結果:",pre)

z=recall_score(y_test, pre) #Sensitivityの算出
print('Sensitivity:',z)

def specificity_score(y_test, pre): #specificityの算出
    tn, fp, fn, tp = confusion_matrix(y_test, pre).flatten()
    return tn / (tn + fp)
    
z1=specificity_score(y_test, pre)
print('Specificity',z1)

z2=roc_auc_score(y_test,pre) #AUCの算出
print('AUC',z2)

z3=matthews_corrcoef(y_test,pre) #MCCの算出
print('MCC:',z3)

#正解率の計算
acc_score=[]
acc=accuracy_score(pre,y_test)
print('Accuracy:',acc)

fpr, tpr, thresholds = roc_curve(y_test,proba[:,1],drop_intermediate=False) #ROC曲線の描画
print(fpr)
print(tpr)
print(thresholds)

plt.plot(fpr,tpr,marker='o')
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.grid()
plt.savefig('sklearn_roc_curve3.png') 