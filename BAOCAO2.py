import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import warnings
from sklearn.svm import SVC

import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

heart_df = pd.read_csv("heart_failure_clinical_records_dataset.csv")

#heart_df.drop(['education'], axis = 1, inplace=True)

heart_df.rename(columns={'age':'TUỔI'},inplace=True)
heart_df.rename(columns={'anaemia':'THIẾU MÁU'},inplace=True)
heart_df.rename(columns={'creatinine_phosphokinase':'CHỈ SỐ CPK'},inplace=True)
heart_df.rename(columns={'diabetes':'TIỂU ĐƯỜNG'},inplace=True)
heart_df.rename(columns={'ejection_fraction':'CHỈ SỐ EF'},inplace=True)
heart_df.rename(columns={'high_blood_pressure':'H.A CAO'},inplace=True)
heart_df.rename(columns={'platelets':'TIỂU CẦU'},inplace=True)
heart_df.rename(columns={'serum_creatinine':'CHỈ SỐ SC'},inplace=True)
heart_df.rename(columns={'serum_sodium':'CHỈ SỐ SS'},inplace=True)
heart_df.rename(columns={'sex':'GIỚI TÍNH'},inplace=True)
heart_df.rename(columns={'smoking':'HÚT THUỐC'},inplace=True)
heart_df.rename(columns={'time':'TG ĐIỀU TRỊ'},inplace=True)
heart_df.rename(columns={'DEATH_EVENT':'NGUY CƠ TỬ VONG'},inplace=True)

thecolumns = ['TUỔI','THIẾU MÁU','CHỈ SỐ CPK','TIỂU ĐƯỜNG','CHỈ SỐ EF','H.A CAO','TIỂU CẦU','CHỈ SỐ SC','CHỈ SỐ SS','GIỚI TÍNH','HÚT THUỐC','TG ĐIỀU TRỊ']

print('\n', heart_df)

# print(heart_df.isnull().sum(axis=0))

                                            # 'TRỰC QUAN HÓA DỮ LIỆU' (VISUALIZE DATA)
count=0
for i in heart_df.isnull().sum(axis=1):
    if i>0:
        count=count+1

print('\n> Tổng số hàng có giá trị bị thiếu là:', count)
print('chiếm',round((count/len(heart_df.index))*100, 4), '% của toàn bộ tập dữ liệu nên những hàng có giá trị còn thiếu sẽ bị loại trừ...')

heart_df.dropna(axis=0, inplace=True)
# print('\n', heart_df)

                                            # PHÂN LOẠI BỘ DỮ LIỆU (SPLIT THE DATAFRAME)

data = heart_df[thecolumns]
label = heart_df['NGUY CƠ TỬ VONG']

data_train, data_test, label_train, label_test = train_test_split(data, label, train_size = 0.8, random_state=13)

                                    # BIỂU DIỄN LOGISTIC REGRESSION (PERFORM LOGISTIC REGRESSION)

logistic_regression = LogisticRegression()
logistic_regression.fit(data_train, label_train)
label_predict = logistic_regression.predict(data_test)

                                    # HIỂN THỊ CONFUSION MATRIX (DISPLAY THE CONFUSION MATRIX)

Confusion_matrix = pd.crosstab(label_test, label_predict, rownames=['THỰC TẾ'], colnames=['DỰ ĐOÁN'])
sn.heatmap(Confusion_matrix, annot=True)
# plt.show()
print('\n\nCONFUSION MATRIX:\n\n', Confusion_matrix, '\n')
print('Accuracy: ', round(metrics.accuracy_score(label_test, label_predict)*100,2), '%')
print('Precision:', round(metrics.precision_score(label_test, label_predict)*100,2), '%')
print('Recall:', round(metrics.recall_score(label_test, label_predict)*100,2), '%')
print('F1 score:', round(metrics.f1_score(label_test, label_predict)*100,2), '%')

                                                # DỰ ĐOÁN CHO BỘ DỮ LIỆU TRUYỀN VÀO

print('\n\n\nDỰ ĐOÁN:\n\tDỮ LIỆU CỦA NHỮNG NGƯỜI THAM GIA DỰ ĐOÁN NGUY CƠ TỬ VONG DO BỆNH TIM:\n')
new_data = {'TUỔI':[70,45,42,45,55],'THIẾU MÁU':[0,0,1,1,0],'CHỈ SỐ CPK':[69,242,102,66,217],'TIỂU ĐƯỜNG':[0,1,1,0,0],'CHỈ SỐ EF':[40,30,42,25,25],
            'H.A CAO':[0,0,1,0,1],'TIỂU CẦU':[293000,334000,237000,233000,314000],'CHỈ SỐ SC':[1.8,1.1,1.2,0.8,1.4],'CHỈ SỐ SS':[133,137,140,135,128],
            'GIỚI TÍNH':[0,1,1,0,1],'HÚT THUỐC':[0,1,0,0,1],'TG ĐIỀU TRỊ':[75,20,80,230,24]}

dataframe2 = pd.DataFrame(new_data, columns = thecolumns)
label_predict2 = logistic_regression.predict(dataframe2)

print (dataframe2)

count1=0
for i in dataframe2.count(axis=1):
        count1=count1+1

for i in range(count1):
    if (list(label_predict2)[i] == [1]):
        print('\n\t\t\t\t(*) KẾT QUẢ DỰ ĐOÁN stt(', i,'): [', list(label_predict2)[i], '] => | CÓ NGUY CƠ TỬ VONG |', sep = '')
    else:
        print('\n\t\t\t\t(*) KẾT QUẢ DỰ ĐOÁN stt(', i,'): [', list(label_predict2)[i], '] => | KHÔNG CÓ NGUY CƠ TỬ VONG |', sep = '')