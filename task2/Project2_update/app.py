import numpy as np
from subtask_2_model.topic_model.Proj.sub2 import preprocess_text
from flask import Flask, request
import joblib
from subtask_1_model.subtask_1_model.subtask_1_model import predict_sentiment as model_predict_sentiment

app = Flask(__name__)
app.json.ensure_ascii = False
app.config['JSON_AS_ASCII'] = False

model_sentiment = joblib.load('subtask_1_model/subtask_1_model/model.joblib')


@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    news_text = request.json.get('news_text')
    if news_text:
        model = model_sentiment['model']
        vectorizer = model_sentiment['vectorizer']
        prediction, probabilities = model_predict_sentiment(
            model, vectorizer, news_text)
        probability_for_prediction = probabilities[prediction]

        sentiment_str = str(prediction)
        # 0=负面, 1=正面
        if prediction == 0:
            sentiment_str = "-1"
        elif prediction == 1:
            sentiment_str = "1"

        return {
            'sentiment': sentiment_str,
            'probability': f"{probability_for_prediction:.2f}"
        }

    return {"error": "No news_text provided"}, 400


model_topic = joblib.load(
    'subtask_2_model/topic_model/Proj/topic_classification_svm_pipeline_split.joblib')

topic_dictionary = {
    '上市保荐书': '1', '保荐/核查意见': '2', '公司章程': '3', '公司章程修订': '4',
    '关联交易': '5', '分配方案决议公告': '6', '分配方案实施': '7', '分配预案': '8',
    '半年度报告全文': '9', '发行保荐书': '10', '年度报告全文': '11', '年度报告摘要': '12',
    '独立董事候选人声明': '13', '独立董事提名人声明': '14', '独立董事述职报告': '15',
    '股东大会决议公告': '16', '诉讼仲裁': '17', '高管人员任职变动': '18'
}


@app.route('/predict_topic', methods=['POST'])
def predict_topic():
    news_text = request.json.get('news_text')
    if news_text:
        processed_text = preprocess_text(news_text)
        input_data = [processed_text]
        predicted_topic = model_topic.predict(input_data)[0]
        probabilities = model_topic.predict_proba(input_data)[0]
        predicted_label_index = np.where(
            model_topic.classes_ == predicted_topic)[0][0]
        predicted_probability = probabilities[predicted_label_index]

        topic_number = topic_dictionary.get(predicted_topic, '0')

        return {
            'topic': topic_number,
            'probability': f"{predicted_probability:.2f}"
        }

    return {"error": "No news_text provided"}, 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5724)
