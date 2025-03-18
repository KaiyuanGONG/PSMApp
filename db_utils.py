import datetime
import json
import os
import sqlite3

import pandas as pd

# 数据库文件路径
DB_PATH = "prediction_history.db"

def init_db():
    """初始化数据库，创建表（如果不存在）"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 创建预测历史表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS prediction_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT NOT NULL,
        username TEXT NOT NULL,
        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        features TEXT NOT NULL,
        prediction_result TEXT NOT NULL,
        probability REAL NOT NULL,
        model_type TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_prediction(patient_id, username, features, prediction_result, probability, model_type):
    """保存预测记录到数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 确保username不为空，如果为空则使用默认值
    if username is None or username == "":
        username = "unknown_user"
    
    # 将特征字典转换为JSON字符串
    features_json = json.dumps(features)
    
    # 插入预测记录
    cursor.execute(
        '''
        INSERT INTO prediction_history 
        (patient_id, username, prediction_date, features, prediction_result, probability, model_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (patient_id, username, datetime.datetime.now(), features_json, prediction_result, probability, model_type)
    )
    
    conn.commit()
    conn.close()

def get_all_predictions():
    """获取所有预测记录"""
    conn = sqlite3.connect(DB_PATH)
    
    # 读取所有记录
    query = '''
    SELECT id, patient_id, username, prediction_date, prediction_result, probability, model_type
    FROM prediction_history
    ORDER BY prediction_date DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def get_predictions_by_user(username):
    """获取指定用户的预测记录"""
    conn = sqlite3.connect(DB_PATH)
    
    # 读取指定用户的记录
    query = '''
    SELECT id, patient_id, username, prediction_date, prediction_result, probability, model_type
    FROM prediction_history
    WHERE username = ?
    ORDER BY prediction_date DESC
    '''
    
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    
    return df

def get_prediction_details(prediction_id):
    """获取单条预测记录的完整详情，包括特征数据"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 读取指定ID的记录
    cursor.execute(
        '''
        SELECT * FROM prediction_history WHERE id = ?
        ''',
        (prediction_id,)
    )
    
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return None
    
    # 将记录转换为字典
    columns = ['id', 'patient_id', 'username', 'prediction_date', 'features', 'prediction_result', 'probability', 'model_type']
    record_dict = dict(zip(columns, record))
    
    # 将features从JSON字符串解析回字典
    record_dict['features'] = json.loads(record_dict['features'])
    
    return record_dict

# 确保应用启动时初始化数据库
init_db() 
