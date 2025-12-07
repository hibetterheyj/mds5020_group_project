'''
FilePath: \Docker\app.py
Author: GoldfishZ 2366385033@qq.com
Date: 2025-12-07 15:48:50
LastEditors: GoldfishZ 2366385033@qq.com
LastEditTime: 2025-12-07 19:50:18
Copyright: 2025 CUHK(SZ).DS institude. All Rights Reserved.
Description: Work done by GoldfishZ!
'''
from flask import Flask, request
import joblib
from subtask_1_model.subtask_1_model.subtask_1_model import predict_sentiment as model_predict_sentiment

app = Flask(__name__)
app.json.ensure_ascii = False
app.config['JSON_AS_ASCII'] = False

model_sentiment = joblib.load('subtask_1_model/subtask_1_model/model.joblib')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    title = request.json.get('title')
    if title:
        model = model_sentiment['model']
        vectorizer = model_sentiment['vectorizer']
        prediction, probabilities = model_predict_sentiment(model, vectorizer, title)
        probability_for_prediction = probabilities[prediction]
        return {
            'prediction': int(prediction),
            'probability': f"{probability_for_prediction:.4f}"
        }   
    
    return None

import numpy as np
from subtask_2_model.topic_model.Proj.sub2 import preprocess_text
model_topic = joblib.load('subtask_2_model/topic_model/Proj/topic_classification_svm_pipeline_split.joblib')
topic_dictionary = {'上市保荐书': 1, '发行保荐书': 10, '保荐/核查意见': 2, '年度报告全文': 11, '公司章程': 3, '年度报告摘要': 12, '公司章程修订': 4, '独立董事候选人声明': 13, '关联交易': 5, '独立董事提名人声明': 14, '分配方案决议公告': 6, '独立董事述职报告': 15, '分配方案实施': 7, '股东大会决议公告': 16, '分配预案': 8, '诉讼仲裁': 17, '半年度报告全文': 9, '高管人员任职变动': 18}
@app.route('/predict_topic', methods=['POST'])
def predict_topic():
    title = request.json.get('title')
    if title:
        processed_title = preprocess_text(title)
        input_data = [processed_title]
        predicted_topic = model_topic.predict(input_data)[0]
        probabilities = model_topic.predict_proba(input_data)[0]
        predicted_label_index = np.where(model_topic.classes_ == predicted_topic)[0][0]
        predicted_probability = probabilities[predicted_label_index]
        
        return {
            'topic': str(topic_dictionary[predicted_topic]),
            'probability': f"{predicted_probability:.4f}"
        }
    
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5724)
