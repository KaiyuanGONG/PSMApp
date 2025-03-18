import os
import pickle

import joblib
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 模型文件路径
MODEL_DIR = "models"
DUMMY_MODEL_DIR = "dummy_models"

# 连续特征、离散特征和二元特征的列表
CONTINUOUS_FEATURES = [
    'Std. dev: g/mL_Choline_Bone+',
    'Min: g/mL_Choline_Liver',
    'Std. dev: g/mL_Choline_Bone-',
    'Peak: g/mL_Choline_Kidney',
    'Peak: g/mL_Choline_Bone-',
    'Neutrophils (G/L)',
    'Leukocytes (G/L)',
    'Alkaline Phosphatase (ALP) levels'
]

DISCRETE_FEATURES = [
    'Number of lymph node involvements (supradiaphragmatic)',
    'Number of lymph node involvements (subdiaphragmatic)',
    'Invasion score of Pelvis'
]

BINARY_FEATURES = [
    'Liver involvement',
    'PSMA-/FDG+',
    'PSMA-/Choline+'
]

# 所有特征列表
ALL_FEATURES = CONTINUOUS_FEATURES + DISCRETE_FEATURES + BINARY_FEATURES

# 将DummyModel类移到模块级别
class DummyModel:
    """一个简单的模拟模型，总是预测FBTP类别，概率为0.8"""
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def predict(self, X):
        """预测类别标签
        
        参数:
            X: 输入特征，维度为 [n_samples, n_features]
            
        返回:
            预测结果, shape=[n_samples]
        """
        return np.array(['FBTP'] * len(X))
    
    def predict_proba(self, X):
        """预测类别概率
        
        参数:
            X: 输入特征，维度为 [n_samples, n_features]
            
        返回:
            预测概率, shape=[n_samples, n_classes]
        """
        # 为了使结果更自然，添加一点随机性
        probs = np.zeros((len(X), 2))
        probs[:, 0] = np.random.uniform(0.1, 0.3, len(X))  # NFBTP的概率
        probs[:, 1] = 1 - probs[:, 0]  # FBTP的概率
        return probs

def create_real_models():
    """
    创建实际的机器学习模型作为替代模型，保持与最终模型相同的输入输出维度。
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # 为连续特征创建GaussianNB模型
    gaussian_nb = GaussianNB()
    X_continuous = np.random.rand(100, len(CONTINUOUS_FEATURES))
    y_continuous = np.random.choice(['NFBTP', 'FBTP'], size=100)
    gaussian_nb.fit(X_continuous, y_continuous)
    
    # 为离散特征创建MultinomialNB模型
    multinomial_nb = MultinomialNB()
    X_discrete = np.random.randint(0, 5, size=(100, len(DISCRETE_FEATURES)))
    y_discrete = np.random.choice(['NFBTP', 'FBTP'], size=100)
    multinomial_nb.fit(X_discrete, y_discrete)
    
    # 为二元特征创建BernoulliNB模型
    bernoulli_nb = BernoulliNB()
    X_binary = np.random.randint(0, 2, size=(100, len(BINARY_FEATURES)))
    y_binary = np.random.choice(['NFBTP', 'FBTP'], size=100)
    bernoulli_nb.fit(X_binary, y_binary)
    
    # 为元分类器创建SVC模型
    svc_meta = SVC(probability=True)
    X_meta = np.random.rand(100, 3)  # 3个基础分类器的输出
    y_meta = np.random.choice(['NFBTP', 'FBTP'], size=100)
    svc_meta.fit(X_meta, y_meta)
    
    # 创建并保存标准化器
    scaler = StandardScaler()
    scaler.fit(X_continuous)
    
    # 保存模型
    joblib.dump(gaussian_nb, os.path.join(MODEL_DIR, 'gaussian_nb.joblib'))
    joblib.dump(multinomial_nb, os.path.join(MODEL_DIR, 'multinomial_nb.joblib'))
    joblib.dump(bernoulli_nb, os.path.join(MODEL_DIR, 'bernoulli_nb.joblib'))
    joblib.dump(svc_meta, os.path.join(MODEL_DIR, 'svc_meta.joblib'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))

def create_dummy_models():
    """
    创建并保存虚拟模型，用于开发阶段。
    """
    if not os.path.exists(DUMMY_MODEL_DIR):
        os.makedirs(DUMMY_MODEL_DIR)
    
    # 创建并保存虚拟的GaussianNB模型
    dummy_gnb = DummyModel(random_state=42)
    with open(os.path.join(DUMMY_MODEL_DIR, 'gaussian_nb.pkl'), 'wb') as f:
        pickle.dump(dummy_gnb, f)
    
    # 创建并保存虚拟的MultinomialNB模型
    dummy_mnb = DummyModel(random_state=43)
    with open(os.path.join(DUMMY_MODEL_DIR, 'multinomial_nb.pkl'), 'wb') as f:
        pickle.dump(dummy_mnb, f)
    
    # 创建并保存虚拟的BernoulliNB模型
    dummy_bnb = DummyModel(random_state=44)
    with open(os.path.join(DUMMY_MODEL_DIR, 'bernoulli_nb.pkl'), 'wb') as f:
        pickle.dump(dummy_bnb, f)
    
    # 创建并保存虚拟的SVC模型
    dummy_svc = DummyModel(random_state=45)
    with open(os.path.join(DUMMY_MODEL_DIR, 'svc_meta.pkl'), 'wb') as f:
        pickle.dump(dummy_svc, f)
    
    # 创建并保存虚拟的标准化器
    dummy_scaler = StandardScaler()
    # 假设每个连续特征的均值为0，标准差为1
    dummy_scaler.mean_ = np.zeros(len(CONTINUOUS_FEATURES))
    dummy_scaler.scale_ = np.ones(len(CONTINUOUS_FEATURES))
    joblib.dump(dummy_scaler, os.path.join(DUMMY_MODEL_DIR, 'scaler.joblib'))

def load_models(use_dummy=True):
    """
    加载模型和标准化器。
    
    参数:
        use_dummy (bool): 是否使用虚拟模型。默认为True。
    
    返回:
        tuple: (gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta, scaler)
    """
    if use_dummy:
        # 检查是否存在虚拟模型，如果不存在则创建
        if not os.path.exists(os.path.join(DUMMY_MODEL_DIR, 'gaussian_nb.pkl')):
            create_dummy_models()
        
        # 加载虚拟模型
        model_dir = DUMMY_MODEL_DIR
        with open(os.path.join(model_dir, 'gaussian_nb.pkl'), 'rb') as f:
            gaussian_nb = pickle.load(f)
        with open(os.path.join(model_dir, 'multinomial_nb.pkl'), 'rb') as f:
            multinomial_nb = pickle.load(f)
        with open(os.path.join(model_dir, 'bernoulli_nb.pkl'), 'rb') as f:
            bernoulli_nb = pickle.load(f)
        with open(os.path.join(model_dir, 'svc_meta.pkl'), 'rb') as f:
            svc_meta = pickle.load(f)
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    else:
        # 检查是否存在实际模型，如果不存在则创建
        if not os.path.exists(os.path.join(MODEL_DIR, 'gaussian_nb.joblib')):
            create_real_models()
        
        # 加载实际模型
        model_dir = MODEL_DIR
        gaussian_nb = joblib.load(os.path.join(model_dir, 'gaussian_nb.joblib'))
        multinomial_nb = joblib.load(os.path.join(model_dir, 'multinomial_nb.joblib'))
        bernoulli_nb = joblib.load(os.path.join(model_dir, 'bernoulli_nb.joblib'))
        svc_meta = joblib.load(os.path.join(model_dir, 'svc_meta.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
    
    return gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta, scaler

def preprocess_data(data, scaler):
    """
    预处理输入数据。
    
    参数:
        data (dict): 包含所有特征的字典。
        scaler (StandardScaler): 用于标准化连续特征的标准化器。
    
    返回:
        tuple: (normalized_continuous, discrete_values, binary_values)
    """
    # 提取连续特征
    continuous_values = np.array([[data[feat] for feat in CONTINUOUS_FEATURES]])
    # 对连续特征进行标准化
    normalized_continuous = scaler.transform(continuous_values)[0]
    
    # 提取离散特征
    discrete_values = np.array([data[feat] for feat in DISCRETE_FEATURES])
    
    # 提取二元特征
    binary_values = np.array([data[feat] for feat in BINARY_FEATURES])
    
    return normalized_continuous, discrete_values, binary_values

def make_prediction(data, models, scaler):
    """
    使用模型进行预测。
    
    参数:
        data (dict): 包含所有特征的字典。
        models (tuple): (gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta)
        scaler (StandardScaler): 用于标准化连续特征的标准化器。
    
    返回:
        tuple: (prediction, probability)
    """
    gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta = models
    
    # 预处理数据
    normalized_continuous, discrete_values, binary_values = preprocess_data(data, scaler)
    
    # 使用各个基础分类器进行预测
    prob_gaussian = gaussian_nb.predict_proba([normalized_continuous])[0][1]  # FBTP的概率
    prob_multinomial = multinomial_nb.predict_proba([discrete_values])[0][1]  # FBTP的概率
    prob_bernoulli = bernoulli_nb.predict_proba([binary_values])[0][1]  # FBTP的概率
    
    # 将基础分类器的预测概率作为元分类器的输入
    meta_input = np.array([[prob_gaussian, prob_multinomial, prob_bernoulli]])
    
    # 使用元分类器进行最终预测
    final_pred = svc_meta.predict(meta_input)[0]
    final_prob = svc_meta.predict_proba(meta_input)[0][1]  # FBTP的概率
    
    return final_pred, final_prob 
