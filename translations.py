"""
存储应用程序的所有翻译文本
"""

# 语言配置
LANGUAGES = {
    'en': 'English',
    'fr': 'Français',
    'zh': '中文'
}

# 翻译文本，按语言分类
TRANSLATIONS = {
    'en': {
        # 通用
        'language_select': 'Select Language',
        'app_title': 'mCRPC [177Lu]Lu-PSMA Therapy Response Prediction',
        'app_description': 'This application uses machine learning models to predict the response of metastatic castration-resistant prostate cancer (mCRPC) patients to [177Lu]Lu-PSMA therapy. Please enter the patient\'s clinical feature data, and the system will predict whether the patient is a Fully Beneficial Therapy Patient (FBTP) or a Non-Fully Beneficial Therapy Patient (NFBTP).',
        'welcome_doctor': 'Welcome, Doctor',
        'logout_button': 'Logout',
        'patient_id': 'Patient ID',
        'patient_id_label': 'Patient ID:',
        'patient_data_input': 'Patient Data Input',
        'predict_button': 'Make Prediction',
        'predicting': 'Making prediction...',
        'prediction_results': 'Prediction Results',
        'prediction_probability': 'Prediction Probability:',
        'feature_contribution': 'Feature Contribution Analysis',
        'feature_importance_note': 'Note: The following is an example of feature contributions based on the current input data. The actual model will provide more accurate analysis.',
        'result_interpretation': 'Result Interpretation:',
        'prediction_disclaimer': 'This prediction is based on machine learning model analysis, only as a clinical decision support tool, and should not replace the physician\'s professional judgment. Specific treatment plans should be formulated by clinicians based on the patient\'s overall condition.',
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA Therapy Response Prediction | Developed based on the paper <a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">\'Predicting Response to [177Lu]Lu-PSMA Therapy in mCRPC Using Machine Learning\'</a>',
        
        # 特征说明
        'feature_explanation': 'Feature Explanation',
        'definition_tab': 'Definitions',
        'continuous_features_tab': 'Continuous Features',
        'discrete_features_tab': 'Discrete Features',
        'binary_features_tab': 'Binary Features',
        'fbtp_definition_title': 'Fully Beneficial Therapy Patient (FBTP) Definition',
        'fbtp_definition': 'Completed all treatment cycles, PSA level reduced by at least 50%',
        'nfbtp_definition_title': 'Non-Fully Beneficial Therapy Patient (NFBTP) Definition',
        'nfbtp_definition': 'Did not meet the FBTP criteria',
        'feature_unit_choline': 'Choline PET scan value, unit in g/mL',
        'feature_unit_neutrophils': 'Neutrophil count, unit in G/L',
        'feature_unit_leukocytes': 'Leukocyte count, unit in G/L',
        'feature_unit_alp': 'Alkaline Phosphatase level, unit in U/L',
        'feature_lymph_supra': 'Number of supradiaphragmatic lymph node involvements',
        'feature_lymph_sub': 'Number of subdiaphragmatic lymph node involvements',
        'feature_invasion_pelvis': 'Pelvis invasion score (0-3)',
        'feature_liver': 'Liver involvement (0=No, 1=Yes)',
        'feature_psma_fdg': 'PSMA negative but FDG positive (0=No, 1=Yes)',
        'feature_psma_choline': 'PSMA negative but Choline positive (0=No, 1=Yes)',
        
        # 预测结果
        'fbtp_result': 'Fully Beneficial Therapy Patient (FBTP)',
        'nfbtp_result': 'Non-Fully Beneficial Therapy Patient (NFBTP)',
        'fbtp_explanation': 'The patient is likely to complete all treatment cycles, and the PSA level is likely to decrease by at least 50%.',
        'nfbtp_explanation': 'The patient may not complete all treatment cycles, or the PSA level may not decrease by at least 50%.',
        'fbtp_full': 'FBTP (Fully Beneficial Therapy Patient): Completed all treatment cycles, PSA level reduced by at least 50%.',
        'nfbtp_full': 'NFBTP (Non-Fully Beneficial Therapy Patient): Did not meet the above criteria.',
        'fbtp_probability': 'FBTP Probability',
        'nfbtp_probability': 'NFBTP Probability',
        
        # 特征分析
        'feature': 'Feature',
        'importance_score': 'Importance Score',
        'relative_importance': 'Relative Importance',
        'top_features': 'Top 10 Most Important Features',
        
        # 登录相关
        'login_title': 'Login System',
        'username': 'Username',
        'password': 'Password',
        'login_button': 'Login',
        'login_error': '😕 Username or password incorrect',
        
        # 历史记录
        'prediction_tab': 'Make Prediction',
        'history_tab': 'Prediction History',
        'prediction_history': 'Prediction History',
        'select_prediction': 'Select a prediction to view details',
        'prediction_details': 'Prediction Details',
        'basic_info': 'Basic Information',
        'prediction_date': 'Prediction Date',
        'prediction_result': 'Prediction Result',
        'model_type': 'Model Type',
        'feature_values': 'Feature Values',
        'value': 'Value',
        'no_prediction_history': 'No prediction history available.',
        
        # 其他
        'yes': 'Yes',
        'no': 'No',
        'continuous_features': 'Continuous Features',
        'continuous_features_cont': 'Continuous Features (cont.)',
        'discrete_features': 'Discrete Features',
        'binary_features': 'Binary Features',
        
        # 添加新的欢迎文本
        'welcome_tool': 'Welcome to our prediction tool',
        'enter_patient_id': 'Enter Patient ID',
        
        # 更新页脚文本
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA Therapy Response Prediction | Developed based on the paper <a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">\'Predicting Response to [177Lu]Lu-PSMA Therapy in mCRPC Using Machine Learning\'</a>',
        'empty_tab': ' ',  # 空标签占位符
    },
    'fr': {
        # 通用
        'language_select': 'Sélectionner la Langue',
        'app_title': 'Prédiction de réponse à la thérapie [177Lu]Lu-PSMA pour mCRPC',
        'app_description': 'Cette application utilise des modèles d\'apprentissage automatique pour prédire la réponse des patients atteints de cancer de la prostate métastatique résistant à la castration (mCRPC) à la thérapie [177Lu]Lu-PSMA. Veuillez entrer les données cliniques du patient, et le système prédira si le patient est un Patient à Thérapie Entièrement Bénéfique (FBTP) ou un Patient à Thérapie Non Entièrement Bénéfique (NFBTP).',
        'welcome_doctor': 'Bienvenue, Docteur',
        'logout_button': 'Déconnexion',
        'patient_id': 'ID du Patient',
        'patient_id_label': 'ID du Patient:',
        'patient_data_input': 'Saisie des Données du Patient',
        'predict_button': 'Faire une Prédiction',
        'predicting': 'Prédiction en cours...',
        'prediction_results': 'Résultats de la Prédiction',
        'prediction_probability': 'Probabilité de Prédiction:',
        'feature_contribution': 'Analyse de la Contribution des Caractéristiques',
        'feature_importance_note': 'Remarque: Ce qui suit est un exemple de contributions de caractéristiques basé sur les données d\'entrée actuelles. Le modèle réel fournira une analyse plus précise.',
        'result_interpretation': 'Interprétation des Résultats:',
        'prediction_disclaimer': 'Cette prédiction est basée sur l\'analyse de modèles d\'apprentissage automatique, uniquement comme outil d\'aide à la décision clinique, et ne doit pas remplacer le jugement professionnel du médecin. Les plans de traitement spécifiques doivent être formulés par les cliniciens en fonction de l\'état général du patient.',
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA Therapy Response Prediction | Développé sur la base de l\'article <a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">\'Predicting Response to [177Lu]Lu-PSMA Therapy in mCRPC Using Machine Learning\'</a>',
        
        # 特征说明
        'feature_explanation': 'Explication des Caractéristiques',
        'definition_tab': 'Définitions',
        'continuous_features_tab': 'Caractéristiques Continues',
        'discrete_features_tab': 'Caractéristiques Discrètes',
        'binary_features_tab': 'Caractéristiques Binaires',
        'fbtp_definition_title': 'Définition du Patient à Thérapie Entièrement Bénéfique (FBTP)',
        'fbtp_definition': 'A terminé tous les cycles de traitement, niveau de PSA réduit d\'au moins 50%',
        'nfbtp_definition_title': 'Définition du Patient à Thérapie Non Entièrement Bénéfique (NFBTP)',
        'nfbtp_definition': 'N\'a pas satisfait aux critères FBTP',
        'feature_unit_choline': 'Valeur de scan PET à la choline, unité en g/mL',
        'feature_unit_neutrophils': 'Nombre de neutrophiles, unité en G/L',
        'feature_unit_leukocytes': 'Nombre de leucocytes, unité en G/L',
        'feature_unit_alp': 'Niveau de phosphatase alcaline, unité en U/L',
        'feature_lymph_supra': 'Nombre d\'atteintes ganglionnaires supradiaphragmatiques',
        'feature_lymph_sub': 'Nombre d\'atteintes ganglionnaires sous-diaphragmatiques',
        'feature_invasion_pelvis': 'Score d\'invasion pelvienne (0-3)',
        'feature_liver': 'Atteinte hépatique (0=Non, 1=Oui)',
        'feature_psma_fdg': 'PSMA négatif mais FDG positif (0=Non, 1=Oui)',
        'feature_psma_choline': 'PSMA négatif mais Choline positif (0=Non, 1=Oui)',
        
        # 预测结果
        'fbtp_result': 'Patient à Thérapie Entièrement Bénéfique (FBTP)',
        'nfbtp_result': 'Patient à Thérapie Non Entièrement Bénéfique (NFBTP)',
        'fbtp_explanation': 'Le patient est susceptible de terminer tous les cycles de traitement, et le niveau de PSA est susceptible de diminuer d\'au moins 50%.',
        'nfbtp_explanation': 'Le patient peut ne pas terminer tous les cycles de traitement, ou le niveau de PSA peut ne pas diminuer d\'au moins 50%.',
        'fbtp_full': 'FBTP (Patient à Thérapie Entièrement Bénéfique): A terminé tous les cycles de traitement, niveau de PSA réduit d\'au moins 50%.',
        'nfbtp_full': 'NFBTP (Patient à Thérapie Non Entièrement Bénéfique): N\'a pas satisfait aux critères ci-dessus.',
        'fbtp_probability': 'Probabilité FBTP',
        'nfbtp_probability': 'Probabilité NFBTP',
        
        # 特征分析
        'feature': 'Caractéristique',
        'importance_score': 'Score d\'Importance',
        'relative_importance': 'Importance Relative',
        'top_features': 'Top 10 des Caractéristiques les Plus Importantes',
        
        # 登录相关
        'login_title': 'Système de Connexion',
        'username': 'Nom d\'utilisateur',
        'password': 'Mot de passe',
        'login_button': 'Connexion',
        'login_error': '😕 Nom d\'utilisateur ou mot de passe incorrect',
        
        # 历史记录
        'prediction_tab': 'Faire une Prédiction',
        'history_tab': 'Historique des Prédictions',
        'prediction_history': 'Historique des Prédictions',
        'select_prediction': 'Sélectionnez une prédiction pour voir les détails',
        'prediction_details': 'Détails de la Prédiction',
        'basic_info': 'Informations de Base',
        'prediction_date': 'Date de Prédiction',
        'prediction_result': 'Résultat de la Prédiction',
        'model_type': 'Type de Modèle',
        'feature_values': 'Valeurs des Caractéristiques',
        'value': 'Valeur',
        'no_prediction_history': 'Aucun historique de prédiction disponible.',
        
        # 其他
        'yes': 'Oui',
        'no': 'Non',
        'continuous_features': 'Caractéristiques Continues',
        'continuous_features_cont': 'Caractéristiques Continues (suite)',
        'discrete_features': 'Caractéristiques Discrètes',
        'binary_features': 'Caractéristiques Binaires',
        
        # 添加新的欢迎文本
        'welcome_tool': 'Bienvenue à notre outil de prédiction',
        'enter_patient_id': 'Entrez ID du Patient',
        
        # 更新页脚文本
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA Therapy Response Prediction | Développé sur la base de l\'article <a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">\'Predicting Response to [177Lu]Lu-PSMA Therapy in mCRPC Using Machine Learning\'</a>',
        'empty_tab': ' ',  # 空标签占位符
    },
    'zh': {
        # 通用
        'language_select': '选择语言',
        'app_title': 'mCRPC [177Lu]Lu-PSMA 疗法响应预测',
        'app_description': '该应用程序使用机器学习模型预测转移性去势抵抗性前列腺癌(mCRPC)患者对[177Lu]Lu-PSMA疗法的响应。请输入患者的临床特征数据，系统将预测患者是否为完全有益治疗患者(FBTP)或非完全有益治疗患者(NFBTP)。',
        'welcome_doctor': '欢迎，医生',
        'logout_button': '退出登录',
        'patient_id': '患者ID',
        'patient_id_label': '患者ID:',
        'patient_data_input': '患者数据输入',
        'predict_button': '进行预测',
        'predicting': '正在进行预测...',
        'prediction_results': '预测结果',
        'prediction_probability': '预测概率:',
        'feature_contribution': '特征贡献分析',
        'feature_importance_note': '注意：以下是基于当前输入数据的特征贡献示例。实际模型将提供更准确的分析。',
        'result_interpretation': '结果解释:',
        'prediction_disclaimer': '该预测基于机器学习模型分析，仅作为临床决策辅助工具，不应替代医生的专业判断。具体治疗方案应由临床医生根据患者的整体情况制定。',
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA 治疗响应预测 | 基于论文<a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">《使用机器学习预测mCRPC对[177Lu]Lu-PSMA治疗的响应》</a>开发',
        
        # 特征说明
        'feature_explanation': '特征说明',
        'definition_tab': '定义',
        'continuous_features_tab': '连续特征',
        'discrete_features_tab': '离散特征',
        'binary_features_tab': '二元特征',
        'fbtp_definition_title': '完全有益治疗患者(FBTP)定义',
        'fbtp_definition': '完成所有治疗周期，PSA水平降低至少50%',
        'nfbtp_definition_title': '非完全有益治疗患者(NFBTP)定义',
        'nfbtp_definition': '未达到FBTP的标准',
        'feature_unit_choline': '胆碱PET扫描值，单位为g/mL',
        'feature_unit_neutrophils': '中性粒细胞计数，单位为G/L',
        'feature_unit_leukocytes': '白细胞计数，单位为G/L',
        'feature_unit_alp': '碱性磷酸酶水平，单位为U/L',
        'feature_lymph_supra': '膈上淋巴结受累数量',
        'feature_lymph_sub': '膈下淋巴结受累数量',
        'feature_invasion_pelvis': '盆腔侵袭评分（0-3）',
        'feature_liver': '肝脏受累（0=否，1=是）',
        'feature_psma_fdg': 'PSMA阴性但FDG阳性（0=否，1=是）',
        'feature_psma_choline': 'PSMA阴性但胆碱阳性（0=否，1=是）',
        
        # 预测结果
        'fbtp_result': '完全有益治疗患者 (FBTP)',
        'nfbtp_result': '非完全有益治疗患者 (NFBTP)',
        'fbtp_explanation': '患者可能会完成所有治疗周期，并且PSA水平可能降低至少50%。',
        'nfbtp_explanation': '患者可能无法完成所有治疗周期，或PSA水平可能不会降低至少50%。',
        'fbtp_full': 'FBTP (完全有益治疗患者): 完成所有治疗周期，PSA水平降低至少50%。',
        'nfbtp_full': 'NFBTP (非完全有益治疗患者): 未达到上述标准。',
        'fbtp_probability': 'FBTP概率',
        'nfbtp_probability': 'NFBTP概率',
        
        # 特征分析
        'feature': '特征',
        'importance_score': '重要性得分',
        'relative_importance': '相对重要性',
        'top_features': '前10个最重要特征',
        
        # 登录相关
        'login_title': '登录系统',
        'username': '用户名',
        'password': '密码',
        'login_button': '登录',
        'login_error': '😕 用户名或密码不正确',
        
        # 历史记录
        'prediction_tab': '进行预测',
        'history_tab': '预测历史',
        'prediction_history': '预测历史',
        'select_prediction': '选择预测记录查看详情',
        'prediction_details': '预测详情',
        'basic_info': '基本信息',
        'prediction_date': '预测日期',
        'prediction_result': '预测结果',
        'model_type': '模型类型',
        'feature_values': '特征值',
        'value': '值',
        'no_prediction_history': '暂无预测历史记录。',
        
        # 其他
        'yes': '是',
        'no': '否',
        'continuous_features': '连续特征',
        'continuous_features_cont': '连续特征 (续)',
        'discrete_features': '离散特征',
        'binary_features': '二元特征',
        
        # 添加新的欢迎文本
        'welcome_tool': '欢迎使用我们的预测工具',
        'enter_patient_id': '请输入患者ID',
        
        # 更新页脚文本
        'footer': '© 2025 mCRPC [177Lu]Lu-PSMA 治疗响应预测 | 基于论文<a href="https://www.mdpi.com/2075-4426/14/11/1068" target="_blank">《使用机器学习预测mCRPC对[177Lu]Lu-PSMA治疗的响应》</a>开发',
        'empty_tab': ' ',  # 空标签占位符
    }
}

def get_text(key, lang="en"):
    """获取指定语言的文本"""
    if lang not in TRANSLATIONS:
        lang = "en"  # 默认使用英语
    
    if key not in TRANSLATIONS[lang]:
        # 如果在指定语言中找不到该键，尝试从英语中获取
        if key in TRANSLATIONS["en"]:
            return TRANSLATIONS["en"][key]
        return f"Missing: {key}"  # 如果完全找不到，返回错误信息
    
    return TRANSLATIONS[lang][key] 
