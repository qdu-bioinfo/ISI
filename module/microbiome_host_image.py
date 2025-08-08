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

import openpyxl
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


# 属+宿主+图像 模型 训练
def train_genus_host_images(genus_features_df, host_features_df, images_features_df,meta_df,out_model):

    model_filename = os.path.join(out_model, "xgb_microbiome_host_images.pkl")

    # 将宿主变量表中的非数值型数据使用LabelEncoder转换为数值型
    host_features_encoded_df = encode(host_features_df)

    X = pd.concat([genus_features_df, host_features_encoded_df, images_features_df], axis=1)
    y = meta_df['Group']

    # 筛选特征列表
    screen = ['Skincolor_C_0', 't37', 't188', 't137', 't34', 't141', 't82', 't17', 'Age_2', 't170', 'Heme_C_0',
              't7', 't182', 't30', 'R2_elasticity_C_0', 't84', 't158', 'g__Ralstonia;', 't22', 'g__Brevundimonas;', 't16']

    # 建模
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_k = XGBClassifier(n_estimators=50,
                            max_depth=5,
                            min_child_weight=4,
                            subsample=0.7,
                            learning_rate=0.05,
                            gamma=4
                            )
    model_k2 = XGBClassifier(n_estimators=50,
                            max_depth=5,
                            min_child_weight=4,
                            subsample=0.7,
                            learning_rate=0.05,
                            gamma=4
                            )

    model_k2.fit(X[screen], y)

    data_list = []  # 用于存储每折数据

    plt.figure(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []
    auc_list = []
    accuracy_list = []

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):

        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        model_k.fit(X_train[screen], y_train)  # 训练模型
        preds = model_k.predict_proba(X_test[screen])[:, 1]  # 正类概率
        y_pred = model_k.predict(X_test[screen])

        auc_tmp = roc_auc_score(y_test, preds)
        accuracy_tmp = accuracy_score(y_test, y_pred)

        auc_list.append(auc_tmp)
        accuracy_list.append(accuracy_tmp)

        # 保存每折的真实标签和预测概率
        fold_data = pd.DataFrame({
            'True_Label': y_test.values,
            'Pred_Prob': preds,
            'Fold': i
        }, index=test_index)
        data_list.append(fold_data)

        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_test, preds)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 绘制每个折叠的ROC曲线
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    # 保存训练好的模型
    joblib.dump(model_k2, model_filename)
    print(f"Microbiome-Host-Images-Model saved to {model_filename}")

    # 合并所有折的结果
    final_df = pd.concat(data_list).sort_index()

    # 绘制平均ROC曲线
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', linestyle='--', label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)

    # 绘制随机猜测的线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guessing')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Genus+Host+Image Model ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    # 保存图片为 PDF 文件
    plt.savefig('result/genus_host_images_roc.pdf', format='pdf',
                bbox_inches='tight')  # bbox_inches='tight' 确保图片边缘没有多余空白

    # 绘制箱线图
    # 调整 Pred_Prob 的值
    final_df.loc[final_df['True_Label'] == 1, 'Pred_Prob'] -= 0.2
    final_df.loc[final_df['True_Label'] == 0, 'Pred_Prob'] -= 0.2

    # 去除异常点的函数
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    # 去除异常点

    final_df = remove_outliers(final_df, 'Pred_Prob').reset_index(drop=True)

    # 绘制箱线图
    sns.set(style="whitegrid")
    plt.figure(figsize=(4, 6))
    sns.boxplot(x='True_Label', y='Pred_Prob', data=final_df, palette="Set3", showfliers=False)

    plt.axhline(y=0, color='r', linestyle='--')  # 绘制y=0的水平线，颜色为红色，线型为虚线

    plt.title('Score Distribution')
    plt.ylabel('Score')
    plt.xticks([0, 1], ['Non-Ideal', 'Ideal'])

    # 保存图片为 PDF 文件
    plt.savefig('result/score.pdf', format='pdf',
                bbox_inches='tight')  # bbox_inches='tight' 确保图片边缘没有多余空白

#属+宿主 模型 测试
def test_genus_host_images(genus_features_df, host_features_df, images_features_df,sample):
    print("sample+++++")  
    print(sample)
    screen = ['Skincolor_C_0', 't37', 't188', 't137', 't34', 't141', 't82', 't17', 'Age_2', 't170', 'Heme_C_0',
              't7', 't182', 't30', 'R2_elasticity_C_0', 't84', 't158', 'g__Ralstonia;', 't22', 'g__Brevundimonas;', 't16']

    model_filename = 'model/xgb_microbiome_host_images.pkl'

    # 将宿主变量表中的非数值型数据使用LabelEncoder转换为数值型
    host_features_encoded_df = encode(host_features_df)

    X = pd.concat([genus_features_df, host_features_encoded_df, images_features_df], axis=1)

    # 加载保存的模型
    model_k = joblib.load(model_filename)

    # 使用加载的模型进行预测
    preds = model_k.predict_proba(X[screen])[:, 1]  # 获取预测概率


    fold_data = pd.DataFrame({
        # 'True_Label': y.values,
        'Pred_Prob': preds,
    }, index=sample)

    print(fold_data)

    # 创建 DataFrame
    # fold_data = pd.DataFrame(data, index=index)

    # 处理索引：去掉括号和逗号
    fold_data.index = fold_data.index.map(lambda x: x[0])  # 提取元组中的第一个元素

    fold_data['Pred_Prob'] = fold_data['Pred_Prob'] - 0.2

    # 保存为 Excel 文件
    # fold_data.to_excel('fold_data.xlsx', index=True)
    
    fold_data.to_excel('result/Scores.xlsx', index=True)

    # 绘制柱状图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    # 动态设置颜色：小于 0 的柱子为红色，其余为蓝色
    colors = ['red' if x < 0 else 'skyblue' for x in fold_data['Pred_Prob']]
    bars = plt.bar(fold_data.index, fold_data['Pred_Prob'], color=colors)  # 绘制柱状图
    # bars = plt.bar(fold_data.index, fold_data['Pred_Prob'], color='skyblue')  # 绘制柱状图

    # 在柱子上方标注数值
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            # 正值的柱子：标注在上方
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=10)
        else:
            # 负值的柱子：标注在下方
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                     ha='center', va='top', fontsize=10)

    # plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Threshold (0)')
    # 设置标题和坐标轴标签
    plt.title('Ideal Skin Index', fontsize=16)
    plt.xlabel('SampleID', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    # 保存柱状图为 PDF
    plt.savefig('result/BarChart.pdf', format='pdf', bbox_inches='tight')

    # 显示图像（可选）
    plt.show()

    # print('Microbiome-Host-Image Model AUC (Test):', auc_tmp)
