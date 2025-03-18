# mCRPC [177Lu]Lu-PSMA 疗法响应预测应用

这个应用程序基于机器学习模型，用于预测转移性去势抵抗性前列腺癌(mCRPC)患者对[177Lu]Lu-PSMA疗法的响应。

## 功能

- 输入患者的14个临床特征
- 预测患者是否为完全有益治疗患者(FBTP)或非完全有益治疗患者(NFBTP)
- 提供预测的概率估计
- 可视化预测结果

## 安装与运行

1. 安装Python 3.8或更高版本
2. 安装所需依赖：
   ```
   pip install -r requirements.txt
   ```
3. 运行应用程序：
   ```
   streamlit run app.py
   ```

## 特征说明

应用程序需要输入以下特征：

### 连续特征
- Std. dev: g/mL_Choline_Bone+：胆碱PET扫描中骨骼阳性病变的标准差
- Min: g/mL_Choline_Liver：肝脏中胆碱PET扫描的最小值
- Std. dev: g/mL_Choline_Bone-：胆碱PET扫描中骨骼阴性区域的标准差
- Peak: g/mL_Choline_Kidney：肾脏胆碱PET扫描的峰值
- Peak: g/mL_Choline_Bone-：胆碱PET扫描中骨骼阴性区域的峰值
- Neutrophils (G/L)：中性粒细胞计数
- Leukocytes (G/L)：白细胞计数
- Alkaline Phosphatase (ALP) levels：碱性磷酸酶水平

### 离散特征
- Number of lymph node involvements (supradiaphragmatic)：膈上淋巴结受累数量
- Number of lymph node involvements (subdiaphragmatic)：膈下淋巴结受累数量
- Invasion score of Pelvis：盆腔侵袭评分

### 二元特征
- Liver involvement：肝脏受累(1=是，0=否)
- PSMA-/FDG+：PSMA阴性但FDG阳性(1=是，0=否)
- PSMA-/Choline+：PSMA阴性但胆碱阳性(1=是，0=否)

## 预测结果解释

- 完全有益治疗患者(FBTP)：完成所有治疗周期，PSA水平降低至少50%
- 非完全有益治疗患者(NFBTP)：未达到上述标准

## 开发者信息

此应用程序基于论文《使用机器学习预测mCRPC中对[177Lu]Lu-PSMA疗法的反应》开发。 
