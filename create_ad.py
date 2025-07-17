import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import inception_v3
from PIL import Image

print("当前工作目录是：", os.getcwd())

# 检查图片路径是否存在
image_path = os.path.join(os.getcwd(), "original_images", "3.png")
print("当前检查路径:", image_path)
if not os.path.exists(image_path):
    print("❌ 图片不存在，请检查路径或文件名")
    exit(1)
else:
    print("✅ 图片存在")

# 加载预训练模型
model = inception_v3.InceptionV3(weights='imagenet')

# 目标攻击类别，这里选 ImageNet 类别编号 1（goldfish）,目标就是让模型把输入判定为“goldfish”，即制造对抗样本
object_type_to_fake = 1

# 攻击学习率,# 设置对抗攻击中梯度更新的步长为 0.5（每次扰动幅度）
learning_rate = 0.5

# 迭代次数最大限制（防止死循环）最大迭代次数 100
max_iterations = 200

# 遍历你想攻击的图片编号
for i in range(3, 4):
    # 加载并预处理图片
    img = image.load_img(f"original_images/{i}.png", target_size=(299, 299))
    original_image = image.img_to_array(img)#转换成 numpy 数组，形状为 (299, 299, 3)，像素是0-255范围
#实例：
    # 归一化到 [-1,1]，InceptionV3的输入预处理
    original_image = (original_image / 255.0 - 0.5) * 2.0

    # 增加 batch 维度,方便模型批量输入
    original_image = np.expand_dims(original_image, axis=0).astype(np.float32)

    # 允许的最大扰动范围
    max_change_above = original_image + 0.01
    max_change_below = original_image - 0.01

    # 初始化对抗图像为原始图像（tf.Variable 可被梯度修改），将 numpy 数组转换成 TensorFlow 的可变变量 tf.Variable
    hacked_image = tf.Variable(original_image)
    ##详细解释：numpy 数组是 Python 中一种非常常用的多维数组类型，用来存储数值数据，比如图片像素、矩阵等。它是静态的数据结构，数据本身不会自动响应任何操作或追踪变化。
    #tf.Variable 是 TensorFlow 中用于存储和更新状态的张量对象。它和普通张量（tf.Tensor）不同的是：tf.Variable 是可变的，意味着你可以在计算过程中修改它的值。TensorFlow 的自动微分（梯度计算）系统会追踪对 tf.Variable 的操作，方便进行反向传播和梯度更新。
    ##所以之所以要转换，是因为从图片文件中读到的 numpy 数组是静态数据，不能直接参与“梯度计算”和“优化”。在对抗攻击中，需要反复调整输入图像的像素值，使得模型的输出朝着攻击目标方向变化。这就要求输入图像是一个可以被 TensorFlow 跟踪和修改的变量，这时就要将 numpy 数组转换成 tf.Variable

    # 构造目标 one-hot 标签，
    #
    # InceptionV3 模型输出1000类概率，目标是第 object_type_to_fake 个类别
    #
    # reshape成 [1, 1000] 的形状匹配模型输出
    target_label = tf.one_hot(object_type_to_fake, 1000)
    target_label = tf.reshape(target_label, (1, 1000))


    #迭代开始：

# 使用 GradientTape 追踪对 hacked_image 的梯度
#
# 运行模型得到预测 predictions
#
# 计算目标类别概率 cost（这里直接取预测的对应类概率）
#
# 计算 cost 对输入图像的梯度 gradients
#
# 根据梯度调整输入图片，使模型对目标类别的信心增加
#
# 限制像素扰动不超过之前设置的范围，并保持像素值合法
#
# 打印当前模型认为输入图像属于目标类别的置信度
#
# 直到置信度达到 0.20 (20%) 或迭代次数超过限制，停止攻击
    iteration = 0
    cost = 0.0

    while cost < 0.20 and iteration < max_iterations:
        iteration += 1

        with tf.GradientTape() as tape:#with ... as ...: 是 Python 里的一个语法结构，叫上下文管理器（context manager）。它用于自动管理资源的申请和释放，比如打开文件时用 with open(...) as f:，不用自己关闭文件，系统自动帮你管理。在 TensorFlow 里，tf.GradientTape() 就是一个上下文管理器。
            #tf.GradientTape 是 TensorFlow 2.x 提供的自动求导（autograd）工具，用来自动记录在“上下文块”内的所有张量操作，方便后续计算梯度。也就是说，你在 with tf.GradientTape() as tape: 代码块内做的张量运算，TensorFlow 会帮你“记笔记”，等到你调用 tape.gradient() 时，就可以基于这些操作计算出某个输出关于输入的导数（梯度）。
            tape.watch(hacked_image)#告诉 TensorFlow 开始监控变量 hacked_image，后面要对它求梯度。
            predictions = model(hacked_image, training=False)
            # [:, object_type_to_fake] 取的是批次中所有样本，目标类别对应的预测值
            cost = predictions[:, object_type_to_fake]


        gradients = tape.gradient(cost, hacked_image) #根据 cost 关于输入图像 hacked_image 的变化率，计算梯度。这里得到的 gradients 是一个张量，表示输入图像的每个像素对目标类别概率的影响方向和大小。
        #使用 TensorFlow 的 GradientTape，计算出当前图像 hacked_image 对于目标输出（cost）的梯度。说白了就是：“我想让模型更相信这张图片是某个目标类别，我该怎么改图片的像素？”

        # 更新图像
        hacked_image.assign_add(learning_rate * gradients)#使用梯度上升法：沿梯度方向调整图像像素值。乘以 learning_rate 控制每次修改的步长，防止扰动过大。assign_add 是 tf.Variable 的原地加法更新操作。

        # 限制扰动范围，防止图像失真太严重
        hacked_image.assign(tf.clip_by_value(hacked_image, max_change_below, max_change_above))
        hacked_image.assign(tf.clip_by_value(hacked_image, -1.0, 1.0))
        #第一句限制像素值变化在原图的 ±0.01 范围内，避免过度失真。第二句确保所有像素值都在模型接受的输入范围 [-1, 1] 内，避免无效输入。tf.clip_by_value 是裁剪张量元素的函数。这两步操作保证扰动既有效又不破坏图像基本结构。

        print(f"Iteration {iteration}: 模型认为目标类别的置信度为 {cost.numpy()[0] * 100:.6f}%")

    # 反归一化，将像素从 [-1, 1] 转回 [0, 255]
    img_result = hacked_image.numpy()[0]
    img_result = (img_result / 2.0 + 0.5) * 255.0
    img_result = np.clip(img_result, 0, 255).astype(np.uint8)

    # 保存对抗样本
    result_image = Image.fromarray(img_result)
    save_dir = "adversarial_images"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{i}.png")
    result_image.save(save_path)
    print(f"保存成功: {save_path}")
    print("------------------------------------------------------")