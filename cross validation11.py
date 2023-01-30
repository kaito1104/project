from sklearn.model_selection import cross_val_score,KFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,confusion_matrix,matthews_corrcoef,roc_auc_score,roc_curve,precision_score
from sklearn import metrics

df=pd.read_csv("encode_data.csv")
training_data=df.drop("label",axis=1)
y=df["label"].values
x=training_data.values

model=RandomForestClassifier(random_state=1234)

#pandasからnumpyへ変換
#x2=np.array(x)
#y2=np.array(y)
x2=x
y2=y
#print(x2)
#print(y2)
fpr_a = []
tpr_a = []
auc_a = []
n=1
Sen_su=0
Spe_su=0
Pre_su=0
Mcc_su=0
Acc_su=0
Auc_su=0


#交差検証(5分割)
K=5
kf=KFold(n_splits=K)

for train_index,test_index in kf.split(x):
    train_x=x2[train_index]
    train_y=y2[train_index]
    test_x=x2[test_index]
    test_y=y2[test_index]

    model.fit(train_x, train_y) #学習
    pre=model.predict(test_x) #予測
    
    #正解率の計算
    #acc_score=[]
    #acc=accuracy_score(pre,test_y)
    #acc_score.append(acc)
    #avg_acc_score = sum(acc_score)/K

    #print('accuracy of each fold - {}'.format(acc_score))
    #print('Avg accuracy : {}'.format(avg_acc_score))
    TP=0  # 真陽性	true positive
    FN=0  # 偽陰性	false negative
    FP=0  # 偽陽性 false positive
    TN=0  # 真陰性	true positive

    for i in range(len(test_index)):
        if test_y[i] == 1 and pre[i] == 1:
            TP += 1
        elif test_y[i] == 1 and pre[i] == 0:
            FN += 1
        elif test_y[i] == 0 and pre[i] == 1:
            FP += 1
        else:
            TN += 1

    Sen=recall_score(test_y, pre) #Sensitivityの算出
    
    def specificity_score(y_test, pre): #specificityの算出
        tn, fp, fn, tp = confusion_matrix(test_y, pre).flatten()
        return tn / (tn + fp)
    
    Spe=specificity_score(test_y, pre)
    Pre=precision_score(test_y,pre)
    Mcc=matthews_corrcoef(test_y,pre)
    Acc=accuracy_score(test_y,pre)
    
    print('%d回目の結果'%(n))
    print('元データ：',test_y)
    print('予測結果：',pre)
    print('TP:%d,FN:%d,FP:%d,TN:%d'%(TP,FN,FP,TN))
    print("Sensitivity = %f" % (Sen))
    print("Spesificity = %f" % (Spe))
    print("Precision = %f" % (Pre))
    print("MCC = %f" % (Mcc))
    print("Accuracy = %f" % (Acc))

    #AUC算出
    test_proba = model.predict_proba(test_x)
    fpr, tpr, threshold = metrics.roc_curve(test_y, test_proba[:, 1],drop_intermediate=False)
    auc = metrics.auc(fpr, tpr)
    fpr_a.append(fpr)
    tpr_a.append(tpr)
    auc_a.append(auc)
    Auc_su+=auc
    #print(fpr)

    #平均
    Sen_su+=Sen
    Spe_su+=Spe
    Pre_su+=Pre
    Mcc_su+=Mcc
    Acc_su+=Acc
    #Auc_su+=metrics.roc_auc_score(test_y,pre)
    n+=1

print('平均評価')
print("Sensitivity = %f" % (Sen_su/5))
print("Spesificity = %f" % (Spe_su/5))
print("Precision = %f" % (Pre_su/5))
print("MCC = %f" % (Mcc_su/5))
print("Accuracy = %f" % (Acc_su/5))
print("AUC = %f" % (Auc_su/5))
#print(auc_a[1])

print(fpr_a)
#ROC曲線
print(auc_a)
fig = plt.figure()
col = ["blue", "green", "red", "black", "orange"]
l = ["1", "2", "3", "4", "5"]
#for iz in range(5):
    #plt.plot(fpr_a[iz], tpr_a[iz],marker='o', color=col[iz], ls="-",label=str(l[iz]) + 'ROC curve (area = %.2f)' % auc_a[iz])
plt.plot(fpr_a[0], tpr_a[0],marker='o', color="blue", ls="-",label="1" + 'ROC curve (area = %.2f)' % auc_a[0])
plt.plot(fpr_a[1], tpr_a[1],marker='o', color="green", ls="-",label="2" + 'ROC curve (area = %.2f)' % auc_a[1])
plt.plot(fpr_a[2], tpr_a[2],marker='o', color="red", ls="-",label="3" + 'ROC curve (area = %.2f)' % auc_a[2])
plt.plot(fpr_a[3], tpr_a[3],marker='o', color="black", ls="-",label="4" + 'ROC curve (area = %.2f)' % auc_a[3])
plt.plot(fpr_a[4], tpr_a[4],marker='o', color="orange", ls="-",label="5" + 'ROC curve (area = %.2f)' % auc_a[4])
plt.legend()
plt.xlabel('FPR: False positive rate')
plt.ylabel('TPR: True positive rate')
plt.title('ROC curve')
plt.grid()
fig.savefig("cross_roc_curve11")