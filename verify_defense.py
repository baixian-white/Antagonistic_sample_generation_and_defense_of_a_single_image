from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np


#加载你原来训练的分类模型（比如 InceptionV3）

# 1.加载 InceptionV3 模型，使用 imagenet 权重
model = InceptionV3(weights='imagenet')

# 2.载入“DCT 去噪后”的图片并预处理
# 读取图片路径
denoised_img_path = 'denoised_images/3.png'  # 例如：'output/denoised_adversarial.png'
original_img_path = 'original_images/3.png'
adversarial_img_path = 'adversarial_images/3.png'

image_paths = [
    'original_images/3.png',
    'adversarial_images/3.png',
    'denoised_images/3.png'
]
image_labels = ['Original', 'Adversarial', 'Denoised']
for img_path, label in zip(image_paths, image_labels):
    #zip() 函数的作用是将多个可迭代对象（比如列表、元组等）“打包”成一个个元组，按对应位置一一配对，返回一个迭代器。
    # 载入图像并调整大小为模型要求的尺寸（InceptionV3 要求 299x299）
#       list1 = ['a', 'b', 'c']
#       list2 = [1, 2, 3]
#       for x, y in zip(list1, list2):
#       print(x, y)

    # 输出
    # a 1
    # b 2
    # c 3
    img = image.load_img(denoised_img_path, target_size=(299, 299))

    # 转换为数组并预处理
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = preprocess_input(x)

    # 3.预测图像的类别并输出结果
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]  # 这是一个 Keras 提供的工具函数，用于将预测结果转为可读的分类标签,取模型输出的前3个最可能的分类结果
    # 然后它会把每一行预测转换成 top-k 的预测结果列表，返回格式是：
    # [[
    #   ('n02504458', 'African_elephant', 0.832),
    #   ('n01871265', 'tusker', 0.116),
    #   ('n02504013', 'Indian_elephant', 0.051)
    # ]]
    # 对应的含义：
    # 'n02504458'	ImageNet 中该类的 WordNet ID
    # 'African_elephant'	可读类别名
    #  0.832	模型对该类别的置信度（概率）

    # 输出结果
    print(f"\n Prediction for [{label}] image:")
    for class_id, name, prob in decoded:
        print(f" - {name:20s}: {prob:.7f}")
