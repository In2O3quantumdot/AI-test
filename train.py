import mnist_loader
import network
import pickle

# 加载数据
print("加载数据...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# 创建网络
print("创建网络...")
net = network.Network([784, 30, 10])

# 训练
print("开始训练...")
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

# 保存模型
print("保存模型...")
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump((net.weights, net.biases), f)
print("✅ 模型已保存到 mnist_model.pkl")

# 可选：同时保存为人类可读的文本格式
with open('model_weights.txt', 'w') as f:
    for i, w in enumerate(net.weights):
        f.write(f"Layer {i} weights shape: {w.shape}\n")
        f.write(str(w) + "\n\n")