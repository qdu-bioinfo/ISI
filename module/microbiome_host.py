import os
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, normalize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
from matplotlib import pyplot

import warnings
warnings.filterwarnings("ignore")


# 将宿主变量表中的非数值型数据使用LabelEncoder转换为数值型
def encode(host_features_df):

    integer_encodeds = []
    for col in host_features_df.columns:
        if host_features_df[col].dtype == 'object':
            le = LabelEncoder()
            integer_encoded = le.fit_transform(host_features_df[col])
            integer_encodeds.append(integer_encoded)
        else:
            integer_encodeds.append(host_features_df[col].values)
    integer_encodeds = np.array(integer_encodeds).T

    # One-Hot编码宿主变量
    encoder = OneHotEncoder()
    host_features_encoded = encoder.fit_transform(integer_encodeds).toarray()
    host_features_encoded_df = pd.DataFrame(host_features_encoded,
                                            columns=encoder.get_feature_names_out(host_features_df.columns))

    return host_features_encoded_df



def train_genus_host(host_features_df, genus_features_df, meta_df,out_model):

    model_filename = os.path.join(out_model, "xgb_microbiome_host.pkl")

    # 将宿主变量表中的非数值型数据使用LabelEncoder转换为数值型
    host_features_encoded_df = encode(host_features_df)

    # 合并宿主和属特征
    X = pd.concat([genus_features_df, host_features_encoded_df], axis=1)
    y = meta_df['Group']

    # 筛选特征列表
    screen = ['Skincolor_C_0', 'Age_2', 'g__Streptococcus;', 'g__Propionibacterium;', 'Age_1', 'g__TM7;', 'g__Veillonella;', 'g__Granulicatella;', 'Nose_skin_1',
              'g__Brevundimonas;', 'Heme_C_0', 'LC_0', 'R2_elasticity_C_0', 'g__Peptoniphilus;', 'cardiovascular_cerebrovascular_0', 'g__Rhodococcus;',
              'g__Staphylococcus;', 'g__Enhydrobacter;']

    # 建模
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_k = XGBClassifier(n_estimators=50,
                            max_depth=5,
                            min_child_weight=4,
                            subsample=0.7,
                            learning_rate=0.05,
                            gamma=4)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model_k.fit(X_train[screen], y_train)  # 训练模型
        preds = model_k.predict_proba(X_test[screen])[:, 1]
        y_pred = model_k.predict(X_test[screen])

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # 计算AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 绘制每折的ROC曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')


    # 保存训练好的模型
    joblib.dump(model_k, model_filename)
    print(f"Microbiome-Host-Model saved to {model_filename}")

    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

    # 绘制随机猜测的线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Genus+Host Model ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图片为 PDF 文件
    plt.savefig('result/genus_host_roc.pdf', format='pdf', bbox_inches='tight')  # bbox_inches='tight' 确保图片边缘没有多余空白

#属+宿主 模型测试
def test_genus_host(host_features_df, genus_features_df, meta_df):

    screen = ['Skincolor_C_0', 'Age_2', 'g__Streptococcus;', 'g__Propionibacterium;', 'Age_1', 'g__TM7;',
              'g__Veillonella;', 'g__Granulicatella;', 'Nose_skin_1',
              'g__Brevundimonas;', 'Heme_C_0', 'LC_0', 'R2_elasticity_C_0', 'g__Peptoniphilus;',
              'cardiovascular_cerebrovascular_0', 'g__Rhodococcus;',
              'g__Staphylococcus;', 'g__Enhydrobacter;']

    host_features_encoded_df = encode(host_features_df)

    # 合并宿主和属特征
    X = pd.concat([genus_features_df, host_features_encoded_df], axis=1)
    y = meta_df['Group']

    #模型路径
    model_filename = 'model/xgb_microbiome_host.pkl'

    # 加载保存的模型
    model_k = joblib.load(model_filename)

    # 使用加载的模型进行预测
    preds = model_k.predict_proba(X[screen])[:, 1]  # 获取预测概率
    auc_tmp = roc_auc_score(y, preds)

    # print('Microbiome-Host Model AUC (Test):', auc_tmp)
