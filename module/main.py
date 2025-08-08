import pandas as pd
import argparse

from module import images
from module import microbiome
from module import microbiome_host
from module import microbiome_host_image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and Test the SCA model")

    # 训练标志
    parser.add_argument('--train', action='store_true', help='Train the SCA model')

    # 测试标志
    parser.add_argument('--test', action='store_true', help='Test the SCA model')

    # 数据文件路径
    parser.add_argument('--images', type=str, required=True, help='Path to the images feature table (CSV)')
    parser.add_argument('--group', type=str, required=False, help='Path to the meta information table (CSV)')
    parser.add_argument('--microbiome', type=str, required=True, help='Path to the microbiome information table (CSV)')
    parser.add_argument('--host', type=str, required=True, help='Path to the host information table (CSV)')

    # 模型文件存储路径
    parser.add_argument('--output', type=str, required=False, help='Path to save the model output')

    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args)

    # 加载数据
    genus_features_df = pd.read_csv(args.microbiome)  # 属特征丰度表
    host_features_df = pd.read_csv(args.host)  # 宿主变量特征表
    images_features_df = pd.read_csv(args.images)  # 图像特征表

    out_model = args.output  # 模型输出路径

    if args.group is not None:
        meta_df = pd.read_csv(args.group)  # group信息表

    print('-' * 50)
    print('Data imported successfully')

    # 保留第一列（样本信息）
    sample_ids = images_features_df.iloc[:, 0]  # 直接从属特征丰度表的第一列获取样本信息
    #print(sample_ids)
    # 删除第一列，保留后面的特征数据
    genus_features_df = genus_features_df.iloc[:, 1:]
    host_features_df = host_features_df.iloc[:, 1:]
    images_features_df = images_features_df.iloc[:, 1:]

    # 训练部分
    if args.train:
        print('-' * 50)
        print("Training the SCA model...")
        images.train_images(images_features_df, meta_df, out_model)
        microbiome.train_genus(genus_features_df, meta_df, out_model)
        microbiome_host.train_genus_host(host_features_df, genus_features_df, meta_df, out_model)
        microbiome_host_image.train_genus_host_images(genus_features_df, host_features_df, images_features_df, meta_df, out_model)
        print('-' * 50)

    # 测试部分
    if args.test:
        print('-' * 50)
        print("Testing the SCA model...")
        # 使用从第一列获取的样本信息进行测试
        microbiome_host_image.test_genus_host_images(genus_features_df, host_features_df, images_features_df, sample_ids)
        print(sample_ids)
        print('-' * 50)

if __name__ == "__main__":
    print("Script is running...")
    main()
