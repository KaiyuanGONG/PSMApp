import base64
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db_utils import (get_all_predictions, get_prediction_details,
                      get_predictions_by_user, save_prediction)
from model_utils import (ALL_FEATURES, BINARY_FEATURES, CONTINUOUS_FEATURES,
                         DISCRETE_FEATURES, load_models, make_prediction)
from translations import LANGUAGES, get_text

# 设置页面配置
st.set_page_config(
    page_title="mCRPC [177Lu]Lu-PSMA Therapy Response Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化会话状态变量
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'
if 'use_dummy_model' not in st.session_state:
    st.session_state['use_dummy_model'] = True
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

# 自定义CSS样式，美化界面
def local_css():
    st.markdown("""
    <style>
        /* 应用基础样式 */
        .main {
            background-color: #f5f7fa;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* 语言选择器样式修复 */
        div[data-testid="stSelectbox"] {
            overflow: visible !important;
        }
        
        div[data-testid="stSelectbox"] > div {
            overflow: visible !important;
        }
        
        div[data-testid="stSelectbox"] > div > div {
            overflow: visible !important;
            min-height: 40px !important;
            display: flex !important;
            align-items: center !important;
        }
        
        div[data-testid="stSelectbox"] label {
            display: none !important;
        }
        
        /* 下拉菜单层级修复 */
        div[data-baseweb="popover"] {
            z-index: 1000 !important;
        }
        
        div[data-baseweb="menu"] {
            z-index: 1001 !important;
        }
        
        /* 修正所有输入框的样式 */
        .stTextInput, .stNumberInput {
            width: 100% !important; 
            overflow: visible !important;
            margin-bottom: 20px !important;
            position: relative !important;
            z-index: 1 !important;
        }
        
        /* 确保输入框内的元素完全可见 */
        .stTextInput > div, .stNumberInput > div {
            width: 100% !important;
            overflow: visible !important;
            height: auto !important;
            min-height: 40px !important;
        }
        
        /* 输入框样式 */
        .stTextInput input, .stNumberInput input {
            width: 100% !important;
            min-width: 0 !important;
            font-size: 16px !important;
            padding: 10px !important;
            border: 1px solid #ccc !important;
            border-radius: 4px !important;
            height: auto !important;
            min-height: 40px !important;
            box-sizing: border-box !important;
        }
        
        /* 特征标签样式 */
        .feature-label {
            display: block !important;
            width: 100% !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            color: #1f5386 !important;
            padding: 0 0 8px 0 !important;
            line-height: 1.4 !important;
            min-height: 30px !important;
            margin-bottom: 5px !important;
            word-break: break-word !important;
            overflow-wrap: break-word !important;
            white-space: normal !important;
        }
        
        /* 特征输入容器样式 */
        .input-container {
            margin-bottom: 15px !important;
            padding: 0 !important;
            position: relative !important;
            z-index: 1 !important;
            overflow: visible !important;
        }
        
        /* 特征容器样式 */
        .feature-container {
            background-color: #f8f9fa !important;
            padding: 20px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
            width: 100% !important;
            box-sizing: border-box !important;
            overflow: visible !important;
        }
        
        /* 特征标题容器 */
        .feature-title-container {
            background-color: #e6f3ff !important;
            padding: 10px 15px !important;
            border-radius: 5px !important;
            margin-bottom: 15px !important;
            border-left: 5px solid #1f5386 !important;
        }
        
        /* 特征类别标题 */
        .feature-title {
            color: #1f5386 !important;
            font-size: 16px !important;
            font-weight: 600 !important;
            margin: 0 !important;
            text-align: left !important;
        }
        
        /* 确保表单内容完全显示 */
        .stForm > div {
            overflow: visible !important;
        }
        
        /* 确保表单内的所有元素正确显示 */
        form {
            overflow: visible !important;
        }
        
        /* 改进列布局 */
        .row-widget.stHorizontalBlock {
            gap: 10px !important;
            margin-bottom: 15px !important;
            overflow: visible !important;
            flex-wrap: wrap !important;
        }
        
        .row-widget.stHorizontalBlock > div {
            flex: 1 1 30% !important;
            min-width: 0 !important;
            position: relative !important;
            z-index: 1 !important;
            overflow: visible !important;
        }
        
        /* 移除表单的默认边框 */
        .stForm > div:first-child {
            border: none !important;
            box-shadow: none !important;
            background-color: transparent !important;
            padding: 0 !important;
        }
        
        /* 改进表单标题样式 */
        .patient-form-title {
            background-color: white !important;
            padding: 20px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
            margin-bottom: 20px !important;
        }
        
        /* 调整二元变量选项显示 */
        .stRadio > div {
            width: 100% !important;
        }
        
        .stRadio > div > div {
            display: flex !important;
            flex-direction: row !important;
            gap: 20px !important;
        }
        
        /* 确保单选按钮文本完全显示 */
        .stRadio label span {
            white-space: normal !important;
            overflow: visible !important;
            display: inline-block !important;
            line-height: 1.2 !important;
        }
        
        /* 移动设备适配 */
        @media (max-width: 768px) {
            .row-widget.stHorizontalBlock > div {
                flex: 1 1 100% !important;
                max-width: 100% !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# 修改背景图片功能
def add_bg_from_local(image_path):
    """使用本地图片作为背景"""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded});
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 使用本地背景图片
add_bg_from_local("assets/background.jpg")

# 简化特征名称的函数
def simplify_feature_name(feature_name):
    """将长特征名简化为更易读的短名称"""
    if 'Choline' in feature_name:
        if 'Bone+' in feature_name:
            return 'Choline_Bone+'
        elif 'Bone-' in feature_name:
            return 'Choline_Bone-'
        elif 'Liver' in feature_name:
            return 'Choline_Liver'
        elif 'Kidney' in feature_name:
            return 'Choline_Kidney'
        else:
            return 'Choline'
    elif 'Neutrophils' in feature_name:
        return 'Neutrophils'
    elif 'Leukocytes' in feature_name:
        return 'Leukocytes'
    elif 'Alkaline Phosphatase' in feature_name:
        return 'ALP'
    elif 'lymph node' in feature_name:
        if 'supradiaphragmatic' in feature_name:
            return 'Lymph_Supra'
        elif 'subdiaphragmatic' in feature_name:
            return 'Lymph_Sub'
    elif 'Invasion score' in feature_name:
        return 'Pelvis_Invasion'
    elif 'Liver involvement' in feature_name:
        return 'Liver_Inv'
    elif 'PSMA-/FDG+' in feature_name:
        return 'PSMA-/FDG+'
    elif 'PSMA-/Choline+' in feature_name:
        return 'PSMA-/Choline+'
    else:
        return feature_name

# 修改创建语言选择器函数
def create_unified_language_selector():
    current_language = st.session_state.get('language', 'en')
    language_options = list(LANGUAGES.keys())
    
    # 创建语言选择器
    with st.sidebar:
        # 添加内联样式，确保当前选择器的正确显示
        st.markdown("""
        <style>
        /* 特定于侧边栏语言选择器的样式 */
        div.language-section div[data-testid="stSelectbox"] {
            margin-bottom: 20px !important;
        }
        div.language-section div[data-testid="stSelectbox"] > div {
            overflow: visible !important;
        }
        div.language-section div[data-testid="stSelectbox"] > div > div {
            min-height: 40px !important;
            padding: 5px 10px !important;
            overflow: visible !important;
            border: 1px solid #ccc !important;
            border-radius: 5px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 使用div包装以添加特定样式
        st.markdown('<div class="language-section">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #1f5386; margin-bottom: 10px;'>Language / 语言</h4>", unsafe_allow_html=True)
        
        selected_language = st.selectbox(
            label="Language",
            options=language_options,
            format_func=lambda x: LANGUAGES[x],
            index=language_options.index(current_language),
            key="unified_language_selector",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if selected_language != current_language:
            st.session_state['language'] = selected_language
            st.rerun()
    
    return selected_language

# 登录功能
def check_password():
    """返回`True` 如果用户输入了正确的用户名和密码"""
    def password_entered():
        """检查用户名和密码"""
        if (
            st.session_state["username"] in st.secrets.get("credentials", {}) and
            st.session_state["password"] == st.secrets.get("credentials", {}).get(st.session_state["username"])
        ):
            st.session_state["password_correct"] = True
            st.session_state["current_username"] = st.session_state["username"]  # 使用新的session state变量保存用户名
            # 检查用户角色
            st.session_state["is_admin"] = (
                st.secrets.get("roles", {}).get(st.session_state["username"]) == "admin"
            )
            del st.session_state["password"]  # 不要在会话状态中保存密码
        else:
            st.session_state["password_correct"] = False
            st.session_state["show_login_error"] = True  # 显示错误信息的标志

    # 初始化会话状态变量
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if "show_login_error" not in st.session_state:
        st.session_state["show_login_error"] = False
        
    # 如果用户未登录，显示登录表单
    if not st.session_state["password_correct"]:
        # 创建统一的语言选择器
        create_unified_language_selector()
        
        with st.container():
            st.markdown(f"""
            <div style="display: flex; justify-content: center; margin-top: 50px;">
                <div style="background-color: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); width: 400px;">
                    <h1 style="text-align: center; color: #2c3e50; margin-bottom: 30px;">{get_text("login_title", st.session_state['language'])}</h1>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_input(get_text("username", st.session_state['language']), key="username")
                st.text_input(get_text("password", st.session_state['language']), type="password", key="password")
                st.button(get_text("login_button", st.session_state['language']), on_click=password_entered)
            
            # 只有在尝试登录后才显示错误信息
            if st.session_state["show_login_error"] and not st.session_state["password_correct"]:
                st.error(get_text("login_error", st.session_state['language']))
            
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)
        return False
    else:
        return True

# 获取翻译文本的辅助函数
def get_translated_text(key):
    return get_text(key, st.session_state['language'])

# 主程序入口
if check_password():
    # 添加页面标题
    st.title(get_translated_text("app_title"))
    
    # 添加统一的语言选择器
    create_unified_language_selector()
    
    # 应用描述
    st.markdown(f"""
    <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px;">
        <p style="font-size: 18px; line-height: 1.6; color: #2c3e50;">
            {get_translated_text("app_description")}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 在侧边栏中添加特征解释和用户信息
    with st.sidebar:
        # 添加两个并排的图标
        col1, col2 = st.columns(2)
        with col1:
            st.image("./assets/doctor_icon.png", use_container_width=True)
        with col2:
            st.image("./assets/hospital_icon.png", use_container_width=True)
        
        # 修改欢迎文本
        st.markdown(f"<h3 style='text-align: center;'>{get_translated_text('welcome_tool')}</h3>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # 开发者选项（模型类型切换）
        with st.expander("Developer Options", expanded=False):
            st.write("These options are for development and testing purposes only.")
            model_type = st.radio(
                "Model Type",
                ["Dummy Model", "Real Model"],
                index=0 if st.session_state['use_dummy_model'] else 1
            )
            st.session_state['use_dummy_model'] = (model_type == "Dummy Model")
            
            st.info(f"""
            Currently using: {'**Dummy Model**' if st.session_state['use_dummy_model'] else '**Real Model**'}  
            
            - Dummy Model: Simple predictive model for testing
            - Real Model: Pre-trained machine learning model
            """)
        
        st.header(get_translated_text("feature_explanation"))
        
        tabs = st.tabs([
            get_translated_text("definition_tab"),
            get_translated_text("continuous_features_tab"),
            get_translated_text("discrete_features_tab"),
            get_translated_text("binary_features_tab")
        ])
        
        with tabs[0]:
            st.subheader(get_translated_text("fbtp_definition_title"))
            st.info(get_translated_text("fbtp_definition"))
            
            st.subheader(get_translated_text("nfbtp_definition_title"))
            st.info(get_translated_text("nfbtp_definition"))
        
        with tabs[1]:
            # 连续特征说明
            for feature in CONTINUOUS_FEATURES:
                if 'Choline' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_unit_choline')}")
                elif 'Neutrophils' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_unit_neutrophils')}")
                elif 'Leukocytes' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_unit_leukocytes')}")
                elif 'Alkaline Phosphatase' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_unit_alp')}")
        
        with tabs[2]:
            # 离散特征说明
            for feature in DISCRETE_FEATURES:
                if 'lymph node' in feature and 'supradiaphragmatic' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_lymph_supra')}")
                elif 'lymph node' in feature and 'subdiaphragmatic' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_lymph_sub')}")
                elif 'Invasion score' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_invasion_pelvis')}")
        
        with tabs[3]:
            # 二元特征说明
            for feature in BINARY_FEATURES:
                if 'Liver involvement' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_liver')}")
                elif 'PSMA-/FDG+' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_psma_fdg')}")
                elif 'PSMA-/Choline+' in feature:
                    st.markdown(f"- **{feature}**: {get_translated_text('feature_psma_choline')}")
        
        # 添加登出按钮
        if st.button(get_translated_text("logout_button")):
            st.session_state["password_correct"] = False
            st.rerun()
    
    # 创建主页面标签 - 修复标签显示问题
    tab1, tab2 = st.tabs([
        get_translated_text("prediction_tab"),
        get_translated_text("history_tab") if st.session_state["is_admin"] else get_translated_text("empty_tab")
    ])
    
    # 修改标签和患者ID之间的间距
    st.markdown("""
    <style>
    /* 移除标签和患者ID之间的白色条 */
    .stTabs {
        margin-bottom: 0 !important;
    }
    .stTabs + div {
        margin-top: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with tab1:
        # 患者ID输入
        st.markdown('<div style="margin-bottom: 30px; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">', unsafe_allow_html=True)
        st.markdown(f'<h5 style="color: #1f5386; margin-bottom: 15px;">{get_translated_text("patient_id")}</h5>', unsafe_allow_html=True)
        patient_id = st.text_input(
            label=" ",  # 使用空格而不是空字符串
            placeholder=get_translated_text("enter_patient_id"),
            value="",
            key="patient_id_input",
            label_visibility="collapsed"  # 隐藏标签
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 使用卡片样式创建一个表单，用于收集患者的数据
        with st.form("patient_data_form"):
            # 表单标题
            st.markdown(f"""
            <div class="patient-form-title">
                <h3 style="color: #2c3e50; border-bottom: 2px solid #4c9be8; padding-bottom: 10px;">{get_translated_text("patient_data_input")}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # 初始化数据字典
            data = {}
            
            # 连续特征标题
            st.markdown(f"""
            <div class="feature-title-container">
                <h4 class="feature-title">{get_translated_text("continuous_features")}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 连续特征输入 - 调整样式确保完全显示
            cols = st.columns(3)
            for i, feature in enumerate(CONTINUOUS_FEATURES):
                col_idx = i % 3
                with cols[col_idx]:
                    default_value = 0.0
                    step = 0.1
                    if 'Neutrophils' in feature or 'Leukocytes' in feature:
                        default_value = 5.0
                    elif 'Alkaline Phosphatase' in feature:
                        default_value = 100.0
                        step = 1.0
                    
                    # 唯一的容器ID确保样式隔离
                    container_id = f"container-cont-{i}"
                    
                    # 使用自定义样式确保标签和输入框完全显示
                    st.markdown(f"""
                    <style>
                    #{container_id} {{
                        margin-bottom: 20px !important;
                        position: relative !important;
                        z-index: 1 !important;
                        overflow: visible !important;
                    }}
                    #{container_id} .feature-label {{
                        font-size: 14px !important;
                        font-weight: 500 !important;
                        color: #1f5386 !important;
                        margin-bottom: 8px !important;
                        line-height: 1.4 !important;
                        display: block !important;
                        white-space: normal !important;
                        word-break: break-word !important;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 使用带唯一ID的容器
                    st.markdown(f'<div id="{container_id}" class="input-container">', unsafe_allow_html=True)
                    st.markdown(f'<label class="feature-label">{feature}</label>', unsafe_allow_html=True)
                    
                    # 创建输入字段并确保宽度正确
                    data[feature] = st.number_input(
                        label=" ",  # 使用空格而不是空字符串
                        min_value=0.0,
                        value=default_value,
                        step=step,
                        format="%.2f",
                        key=f"cont_{i}",
                        label_visibility="collapsed"  # 隐藏标签
                    )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # 离散特征标题
            st.markdown(f"""
            <div class="feature-title-container">
                <h4 class="feature-title">{get_translated_text("discrete_features")}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 离散特征输入 - 调整样式确保完全显示
            cols = st.columns(3)
            for i, feature in enumerate(DISCRETE_FEATURES):
                col_idx = i % 3
                with cols[col_idx]:
                    # 唯一的容器ID确保样式隔离
                    container_id = f"container-disc-{i}"
                    
                    # 使用自定义样式确保标签和输入框完全显示
                    st.markdown(f"""
                    <style>
                    #{container_id} {{
                        margin-bottom: 20px !important;
                        position: relative !important;
                        z-index: 1 !important;
                        overflow: visible !important;
                    }}
                    #{container_id} .feature-label {{
                        font-size: 14px !important;
                        font-weight: 500 !important;
                        color: #1f5386 !important;
                        margin-bottom: 8px !important;
                        line-height: 1.4 !important;
                        display: block !important;
                        white-space: normal !important;
                        word-break: break-word !important;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 使用带唯一ID的容器
                    st.markdown(f'<div id="{container_id}" class="input-container">', unsafe_allow_html=True)
                    st.markdown(f'<label class="feature-label">{feature}</label>', unsafe_allow_html=True)
                    
                    if 'Invasion score' in feature:
                        data[feature] = st.number_input(
                            label=" ",  # 使用空格而不是空字符串
                            min_value=0,
                            max_value=3,
                            value=0,
                            step=1,
                            key=f"disc_{i}",
                            label_visibility="collapsed"  # 隐藏标签
                        )
                    else:
                        data[feature] = st.number_input(
                            label=" ",  # 使用空格而不是空字符串
                            min_value=0,
                            value=0,
                            step=1,
                            key=f"disc_{i}",
                            label_visibility="collapsed"  # 隐藏标签
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # 二元特征标题
            st.markdown(f"""
            <div class="feature-title-container">
                <h4 class="feature-title">{get_translated_text("binary_features")}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # 二元特征输入 - 调整样式确保完全显示
            cols = st.columns(2)
            for i, feature in enumerate(BINARY_FEATURES):
                col_idx = i % 2
                with cols[col_idx]:
                    # 使用带样式的特征名称标签
                    st.markdown(f"""
                    <div class="input-container">
                        <label class="feature-label">{feature}</label>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 创建唯一的CSS类名，确保样式只应用于当前单选按钮
                    unique_class = f"binary-options-{i}"
                    
                    # 添加自定义CSS优化单选按钮显示
                    st.markdown(f"""
                    <style>
                    /* 优化当前单选按钮组的样式 */
                    .{unique_class} div[data-testid="stRadio"] > div > div {{
                        display: flex !important;
                        justify-content: space-between !important;
                        gap: 15px !important;
                        margin-top: 5px !important;
                    }}
                    .{unique_class} div[data-testid="stRadio"] > div > div > label {{
                        flex: 1 !important;
                        padding: 8px 5px !important;
                        border: 1px solid #ddd !important;
                        border-radius: 5px !important;
                        text-align: center !important;
                        font-size: 14px !important;
                        cursor: pointer !important;
                        transition: all 0.2s !important;
                    }}
                    .{unique_class} div[data-testid="stRadio"] > div > div > label:hover {{
                        background-color: #f0f8ff !important;
                        border-color: #4c9be8 !important;
                    }}
                    .{unique_class} div[data-testid="stRadio"] > div > div > label > div:first-child {{
                        margin-right: 5px !important;
                    }}
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # 使用唯一类名包装单选按钮
                    st.markdown(f'<div class="{unique_class}">', unsafe_allow_html=True)
                    data[feature] = st.radio(
                        label=" ",  # 使用空格而不是空字符串
                        options=[0, 1],
                        format_func=lambda x: get_translated_text("no") if x == 0 else get_translated_text("yes"),
                        horizontal=True,
                        key=f"binary_{i}",
                        label_visibility="collapsed"  # 隐藏标签
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # 提交按钮
            submit_button = st.form_submit_button(get_translated_text("predict_button"))
            st.markdown("""
            <style>
            div.stButton > button:first-child {
                width: 100%;
                height: 3em;
                font-size: 16px;
                font-weight: bold;
                background-color: #4c9be8;
                color: white;
            }
            div.stButton > button:hover {
                background-color: #3d8bd5;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # 如果用户点击了提交按钮，则进行预测
        if submit_button:
            # 显示加载动画
            with st.spinner(get_translated_text("predicting")):
                # 加载模型
                models = load_models(use_dummy=st.session_state['use_dummy_model'])
                gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta, scaler = models
                
                # 进行预测
                prediction, probability = make_prediction(data, (gaussian_nb, multinomial_nb, bernoulli_nb, svc_meta), scaler)
                
                # 使用新的session state变量获取用户名
                current_username = st.session_state.get("current_username", "unknown")
                
                # 保存预测记录到数据库
                save_prediction(
                    patient_id=patient_id,
                    username=current_username,  # 使用新的session state变量
                    features=data,
                    prediction_result=prediction,
                    probability=probability,
                    model_type="Dummy Model" if st.session_state['use_dummy_model'] else "Real Model"
                )
                
                # 显示预测结果
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 30px;">
                    <h3 style="color: #2c3e50; border-bottom: 2px solid #4c9be8; padding-bottom: 10px;">{get_translated_text("prediction_results")}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 创建两列布局来展示结果
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # 根据预测结果显示不同的消息和颜色
                    if prediction == "FBTP":
                        st.markdown(f"""
                        <div class="prediction-box success">
                            <h3 style="color: #155724;">{get_translated_text("fbtp_result")}</h3>
                            <p><b>{get_translated_text("patient_id_label")}</b> {patient_id}</p>
                            <p><b>{get_translated_text("prediction_probability")}</b> {probability:.2f} (or {probability*100:.1f}%)</p>
                            <p style="color: #155724;">{get_translated_text("fbtp_explanation")}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box error">
                            <h3 style="color: #721c24;">{get_translated_text("nfbtp_result")}</h3>
                            <p><b>{get_translated_text("patient_id_label")}</b> {patient_id}</p>
                            <p><b>{get_translated_text("prediction_probability")}</b> {1-probability:.2f} (or {(1-probability)*100:.1f}%)</p>
                            <p style="color: #721c24;">{get_translated_text("nfbtp_explanation")}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # 创建可视化图表
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = probability if prediction == "FBTP" else 1-probability,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': get_translated_text("fbtp_probability") if prediction == "FBTP" else get_translated_text("nfbtp_probability"), 'font': {'size': 24, 'color': '#2c3e50'}},
                        gauge = {
                            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#2c3e50"},
                            'bar': {'color': "#27ae60" if prediction == "FBTP" else "#e74c3c"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#2c3e50",
                            'steps': [
                                {'range': [0, 0.33], 'color': "#f2f2f2"},
                                {'range': [0.33, 0.67], 'color': "#d9d9d9"},
                                {'range': [0.67, 1], 'color': "#bfbfbf"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': probability if prediction == "FBTP" else 1-probability
                            }
                        }
                    ))
                    
                    # 设置图表布局
                    fig.update_layout(
                        height=300,
                        margin=dict(l=10, r=10, t=50, b=10),
                        paper_bgcolor="white",
                        font=dict(size=14, color="#2c3e50")
                    )
                    
                    # 显示图表
                    st.plotly_chart(fig, use_container_width=True)
                
                # 特征重要性分析标题
                st.markdown(f"""
                <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-top: 30px;">
                    <h3 style="color: #2c3e50; border-bottom: 2px solid #4c9be8; padding-bottom: 10px; text-align: left;">{get_translated_text("feature_contribution")}</h3>
                    <p style="color: #7f8c8d; font-style: italic; margin-top: 10px;">{get_translated_text("feature_importance_note")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 创建虚拟的特征重要性
                np.random.seed(42)  # 设置随机种子以获得一致的结果
                feature_importance = {}
                for i, feature in enumerate(ALL_FEATURES):
                    # 为每个特征分配一个随机的重要性值
                    importance_value = np.random.uniform(0, 1)
                    # 使用简化的特征名
                    simplified_name = simplify_feature_name(feature)
                    feature_importance[simplified_name] = importance_value
                
                # 将特征重要性转换为DataFrame并排序
                importance_df = pd.DataFrame({
                    get_translated_text("feature"): list(feature_importance.keys()),
                    get_translated_text("importance_score"): list(feature_importance.values())
                })
                importance_df = importance_df.sort_values(get_translated_text("importance_score"), ascending=False).reset_index(drop=True)
                
                # 只显示前10个最重要的特征
                top_features = importance_df.head(10)
                
                # 使用Plotly创建条形图
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_features[get_translated_text("importance_score")],
                    y=top_features[get_translated_text("feature")],
                    orientation='h',
                    marker=dict(
                        color=top_features[get_translated_text("importance_score")],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title=get_translated_text("importance_score"))
                    )
                ))
                
                # 设置图表布局 - 标题左对齐
                fig.update_layout(
                    title=dict(
                        text=get_translated_text("top_features"),
                        font=dict(size=22, color="#2c3e50"),
                        x=0.0,  # 设置为0表示左对齐
                        xanchor="left"  # 确保锚点在左侧
                    ),
                    xaxis_title=dict(text=get_translated_text("relative_importance"), font=dict(size=14, color="#2c3e50")),
                    yaxis_title=dict(text=get_translated_text("feature"), font=dict(size=14, color="#2c3e50")),
                    height=500,
                    margin=dict(l=50, r=30, t=80, b=30),  # 增加左右边距
                    paper_bgcolor="white",
                    plot_bgcolor="white",
                    font=dict(size=12, color="#2c3e50"),
                    xaxis=dict(
                        gridcolor='#EEEEEE',
                        showgrid=True,
                        zeroline=False,
                        showline=True,
                        linecolor='#CCCCCC'
                    ),
                    yaxis=dict(
                        gridcolor='#EEEEEE',
                        showgrid=False,
                        zeroline=False,
                        showline=True,
                        linecolor='#CCCCCC'
                    )
                )
                
                # 显示图表
                st.plotly_chart(fig, use_container_width=True)
                
                # 添加说明
                st.markdown(f"""
                <div style="background-color: #e6f3ff; padding: 20px; border-radius: 8px; margin-top: 20px;">
                    <h4 style="color: #1f5386; text-align: left;">{get_translated_text("result_interpretation")}</h4>
                    <ul style="margin-left: 20px;">
                        <li>{get_translated_text("fbtp_full")}</li>
                        <li>{get_translated_text("nfbtp_full")}</li>
                    </ul>
                    <p style="margin-top: 10px;">{get_translated_text("prediction_disclaimer")}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 如果是管理员，显示历史记录标签页
    if st.session_state["is_admin"]:
        with tab2:
            st.header(get_translated_text("prediction_history"))
            
            # 获取所有预测记录
            df = get_all_predictions()
            
            if not df.empty:
                # 显示预测记录表格
                st.dataframe(df)
                
                # 添加查看详情功能
                selected_id = st.selectbox(
                    get_translated_text("select_prediction"),
                    options=df['id'].tolist(),
                    format_func=lambda x: f"ID: {x}"
                )
                
                if selected_id:
                    # 获取预测详情
                    details = get_prediction_details(selected_id)
                    
                    if details:
                        # 显示预测详情
                        st.subheader(get_translated_text("prediction_details"))
                        
                        # 创建两列布局
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                <h4 style="color: #2c3e50;">{get_translated_text("basic_info")}</h4>
                                <p><b>{get_translated_text("patient_id_label")}</b> {details['patient_id']}</p>
                                <p><b>{get_translated_text("prediction_date")}</b> {details['prediction_date']}</p>
                                <p><b>{get_translated_text("prediction_result")}</b> {details['prediction_result']}</p>
                                <p><b>{get_translated_text("prediction_probability")}</b> {details['probability']:.2f}</p>
                                <p><b>{get_translated_text("model_type")}</b> {details['model_type']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # 显示特征值
                            st.markdown(f"""
                            <div style="background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                                <h4 style="color: #2c3e50;">{get_translated_text("feature_values")}</h4>
                            """, unsafe_allow_html=True)
                            
                            # 将特征值转换为DataFrame以便更好地显示
                            features_df = pd.DataFrame({
                                get_translated_text("feature"): list(details['features'].keys()),
                                get_translated_text("value"): list(details['features'].values())
                            })
                            
                            st.dataframe(features_df)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info(get_translated_text("no_prediction_history"))
    
    # 修改页脚，使用翻译的文本
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; padding: 10px;">
        <p>{get_translated_text("footer")}</p>
    </div>
    """, unsafe_allow_html=True) 
