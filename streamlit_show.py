import streamlit as st
import pydicom
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import base64
from os.path import join, dirname, realpath

NUM_CLASS = 2
from os.path import join, dirname, realpath

def get_base64_of_bin_file(bin_file):
    """
    将二进制文件转换成base64字符串
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

# 假设您的图片名称是 'background.png'
# 并且该图片位于与您的Streamlit脚本相同的目录中
png_image = join(dirname(realpath(__file__)), "background(2).png")
set_png_as_page_bg(png_image)



# 定义图像预处理函数
def img_resize(img, size=224):
    img = cv2.resize(img, (size, size))
    return img

def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())  # 归一化[0,1]
    img = img * 255  # 0-255
    img = img.astype(np.uint8)
    return img

def extend_channels(img):
    img_channels = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
    img_channels[:, :, 0] = img
    img_channels[:, :, 1] = img
    img_channels[:, :, 2] = img
    return img_channels

def data_preprocess_base(img, size=224):
    # 缩放尺寸 224*224
    img = img_resize(img, size)
    # 归一化[0,255]
    img = normalize(img)
    # 扩展为3通道 224*224*3
    img = extend_channels(img)
    return img

# 设置页面标题
st.title('头部CT伪影识别')

# 上传.dcm文件
uploaded_file = st.file_uploader("请上传.dcm格式的CT图像", type=["dcm"])

# 模型加载与预测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 模型路径
model_paths = {
    "alexnet": r"D:\LostAndFound\pythonProject\HeadCT-Tai-Artifact-Removal\git-streamlit\alexnet_model.pkl",
    "densenet": r"D:\LostAndFound\pythonProject\HeadCT-Tai-Artifact-Removal\git-streamlit\densenet_model.pkl",
    "resnet18": r"D:\LostAndFound\pythonProject\HeadCT-Tai-Artifact-Removal\git-streamlit\resnet18_model.pkl",
    "vgg16": r"D:\LostAndFound\pythonProject\HeadCT-Tai-Artifact-Removal\git-streamlit\vgg16_model.pkl"
}

# 显示原始图像和预处理后的图像
if uploaded_file is not None:
    # 读取DICOM文件
    ds = pydicom.dcmread(uploaded_file)
    img_array = ds.pixel_array
    normalized_img_array = normalize(img_array)

    # 显示原始图像和预处理后的图像在同一行
    col1, col2 = st.columns(2)

    with col1:
        st.image(normalized_img_array, caption='原始DICOM图像', use_column_width=True, channels="GRAY")

    # 图像预处理
    img_processed = data_preprocess_base(img_array)

    with col2:
        st.image(img_processed, caption='预处理后的图像', use_column_width=True)

    # 模型选择
    model_name = st.selectbox("请选择预测模型", list(model_paths.keys()))
    # 根据选择创建模型实例
    if model_name == "alexnet":
        model = models.alexnet(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)  # 假设 NUM_CLASS 已经定义
    elif model_name == "densenet":
        model = models.densenet121(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, NUM_CLASS)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASS)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, NUM_CLASS)

    # 加载选定的模型
    model_path = model_paths[model_name]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 将图像转换为模型所需的格式
    img_tensor = transform(Image.fromarray(img_processed)).unsqueeze(0).to(device)

    # 开始识别
    if st.button("开始识别"):
        # 进行预测
        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        # 映射预测结果到标签
        class_labels = {0: "该CT头部影像无伪影", 1: "该CT头部影像有伪影"}
        prediction_label = class_labels[predicted.item()]

        # 显示预测结果
        st.write(f"预测结果: {prediction_label}")

        # 如果需要，还可以显示预测的概率
        probabilities = torch.softmax(output, dim=1)
        prob_no_artifact = probabilities[0][0].item()
        prob_with_artifact = probabilities[0][1].item()

        st.write(f"无伪影的概率: {prob_no_artifact:.2%}")
        st.write(f"有伪影的概率: {prob_with_artifact:.2%}")


st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f8f9fa;
            text-align: center;
            padding: 5px 0;
            font-size: 12px;
            color: #6c757d;
        }
    </style>
    <div class="footer">
        &copy; 2024 头部CT伪影识别系统 | 由 Streamlit 构建
    </div>
""", unsafe_allow_html=True)

# 运行Streamlit应用
# 在命令行中输入: streamlit run D:\LostAndFound\pythonProject\HeadCT-Tai-Artifact-Removal\CaseStudy\streamlit_show.py
