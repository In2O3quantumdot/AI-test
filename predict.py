import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import network


class DigitPredictor:
    def __init__(self, model_file='mnist_model.pkl'):
        """加载训练好的模型"""
        print(f"加载模型: {model_file}")
        with open(model_file, 'rb') as f:
            weights, biases = pickle.load(f)

        # 创建网络（结构必须匹配训练时的结构）
        self.net = network.Network([784, 30, 10])
        self.net.weights = weights
        self.net.biases = biases
        print("✅ 模型加载成功")

    def preprocess_image(self, image_path):
        """预处理图片：转为28x28灰度图，反色，归一化"""
        # 打开图片并转为灰度图
        img = Image.open(image_path).convert('L')

        # 调整大小为28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # 转换为numpy数组并归一化到[0,1]
        img_array = np.array(img) / 255.0

        # 反色（因为MNIST是黑底白字，而通常手写是白底黑字）
        # 如果你的图片是白底黑字，需要反色；如果是黑底白字，则不需要
        img_array = 1.0 - img_array

        # 重塑为 (784, 1) 向量
        img_vector = img_array.reshape(784, 1)

        return img_vector, img_array.reshape(28, 28)

    def predict(self, image_path, show_image=True):
        """预测图片中的数字"""
        # 预处理
        img_vector, img_display = self.preprocess_image(image_path)

        # 预测
        digit, confidence = self.net.predict(img_vector)

        # 显示图片（可选）
        if show_image:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4, 4))
            plt.imshow(img_display, cmap='gray')
            plt.title(f'预测结果: {digit} (置信度: {confidence:.2%})', fontsize=14)
            plt.axis('off')
            plt.show()

        return digit, confidence


# 使用示例
if __name__ == "__main__":
    # 初始化预测器
    predictor = DigitPredictor('mnist_model.pkl')

    # 预测你的手写图片
    image_path = 'my_predict.png'  # 替换为你的图片路径
    digit, confidence = predictor.predict(image_path)

    print(f"\n🔢 预测结果: {digit}")
    print(f"📊 置信度: {confidence:.2%}")