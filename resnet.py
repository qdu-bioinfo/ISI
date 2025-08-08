import argparse
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 加载本地模型
def load_model():
    model_path = "/var/www/html/Ideal_Skin/cache/hub/checkpoints/resnet18-f37072fd.pth"  # 模型路径写死
    print("Loading model...")

    try:
        if os.path.exists(model_path):  # 检查本地模型路径是否存在
            print(f"Loading model from local path: {model_path}")
            model = models.resnet18()
            model.load_state_dict(torch.load(model_path))  # 从本地路径加载模型
        else:
            print("Model not found at the specified path. Loading default ResNet-18.")
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # 使用预训练的 ResNet-18
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # 删除最后一层
        feature_extractor.eval()  # 设置为评估模式
        print("Model loaded successfully.")
        return feature_extractor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# 图像预处理
def get_preprocess():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 提取图像特征
def extract_features(img_path, feature_extractor, preprocess, desired_feature_dim=224):
    img = Image.open(img_path)
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # 添加一个维度，成为批次中的单个样本

    # 使用模型提取图像特征
    with torch.no_grad():
        features = feature_extractor(img_tensor)

    # 将特征张量调整为指定的维度
    features = torch.flatten(features, start_dim=1)
    if features.shape[1] > desired_feature_dim:
        features = features[:, :desired_feature_dim]
    elif features.shape[1] < desired_feature_dim:
        padding = torch.zeros((features.shape[0], desired_feature_dim - features.shape[1]))
        features = torch.cat((features, padding), dim=1)
    return features

# 保存特征到 CSV 文件
def save_features_to_csv(sample_ids, features, output_file_path):
    result_data = np.column_stack((sample_ids, features))  # 直接使用 features，不调用 .numpy()
    with open(output_file_path, "a") as file:
        np.savetxt(file, result_data, delimiter=",", fmt='%s')  # 使用 '%s' 以确保样本ID作为字符串写入文件


# 主功能：处理图像并保存特征
def process_images(directory_path, feature_extractor, preprocess, output_file_path, desired_feature_dim=224):
    sample_ids = []
    features_list = []

    # 获取目录下所有图像文件
    directory_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for file in directory_files:
        # 使用文件名（不包括扩展名）作为样本 ID
        sample_id = os.path.splitext(os.path.basename(file))[0]
        sample_ids.append(sample_id)

# 输出正在处理的图片名称
        print(f"Processing image: {sample_id} ({file})")

        if os.path.exists(file):
            features = extract_features(file, feature_extractor, preprocess, desired_feature_dim)
            if features is not None:
                features_list.append(features.numpy())
            else:
                print(f"Skipping {file}, features could not be extracted.")
    
    if features_list:
        save_features_to_csv(sample_ids, np.vstack(features_list), output_file_path)
    else:
        print("No features extracted.")

# 命令行参数解析
def main(args):
    feature_extractor = load_model()  # 不再使用 --model_path 参数
    if feature_extractor:
        preprocess = get_preprocess()
        process_images(args.directory_path, feature_extractor, preprocess, args.output_file_path, args.feature_dim)
    else:
        print("Model could not be loaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from images using ResNet-18")
    parser.add_argument('--directory_path', type=str, required=True, help="Directory containing image files")
    parser.add_argument('--output_file_path', type=str, required=True, help="Path to the output CSV file")
    parser.add_argument('--feature_dim', type=int, default=224, help="Desired feature dimension for output (default is 224)")
    args = parser.parse_args()
    main(args)
