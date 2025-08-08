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

# 图像模型训练
def train_images(images_features_df, meta_df,out_model):

    X = images_features_df
    y = meta_df['Group']

    model_filename = os.path.join(out_model, "xgb_images.pkl")

    # 建模--添加K折验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_k = XGBClassifier(n_estimators=50,
                            max_depth=5,
                            min_child_weight=4,
                            subsample=0.7,
                            learning_rate=0.05,
                            gamma=4
                            )  # 手动调参

    tprs = []  # 存储每一折的插值tpr
    aucs = []  # 存储每一折的AUC值
    mean_fpr = np.linspace(0, 1, 100)  # 定义一个统一的fpr范围，便于插值
    auc_list = []

    plt.figure(figsize=(10, 8))  # 设置绘图区域

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model_k.fit(X_train, y_train)  # 训练模型
        preds = model_k.predict_proba(X_test)[:, 1]  # 返回预测的正例概率
        y_pred = model_k.predict(X_test)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_test, preds)

        # 插值处理
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # 确保起始点为(0, 0)
        tprs.append(interp_tpr)

        # 计算AUC
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        auc_list.append(roc_auc)

        # 绘制每个折叠的ROC曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {roc_auc:.2f})')

    # 保存训练好的模型
    joblib.dump(model_k, model_filename)
    print(f"Images-Model saved to {model_filename}")

    # 计算平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0  # 确保终点为(1, 1)
    mean_auc = auc(mean_fpr, mean_tpr)

    # 绘制平均ROC曲线
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label=f'Mean ROC (AUC = {mean_auc:.2f})', lw=2)

    # 绘制随机猜测的线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Image Model ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    # 保存图片为 PDF 文件
    plt.savefig('result/images_roc.pdf', format='pdf', bbox_inches='tight')  # bbox_inches='tight' 确保图片边缘没有多余空白

#图像模型测试
def test_images(images_features_df, meta_df):

    X = images_features_df
    y = meta_df['Group']

    #模型路径
    model_filename = 'model/xgb_images.pkl'

    # 加载保存的模型
    model_k = joblib.load(model_filename)

    # 使用加载的模型进行预测
    preds = model_k.predict_proba(X)[:, 1]  # 获取预测概率
    auc_tmp = roc_auc_score(y, preds)

    # print('Images-Model AUC (Test):', auc_tmp)