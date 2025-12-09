# Task2-Subtask1（金融新闻情感分析）优化方案与轻量部署推荐

## doubao

结合你的实验现状（传统ML模型效果≈81%，手工特征提升有限，需满足轻量部署约束），以下从**效果优化**、**轻量模型推荐**、**部署落地**三个维度提供系统性方案，优先保证「资源约束」（容器内存≤900MB、镜像≤4GB）和「预测速度」（占5%评分），同时最大化「加权F1得分」（占20%评分）。


### 一、现状诊断与核心问题
首先明确当前实验的关键结论，为后续优化提供依据：
1. **模型表现**：线性模型（Logistic Regression、LinearSVC）效果最优（≈81%），集成模型（XGBoost/LightGBM）反而更低（≈77%），原因是文本数据（4300条）量较小，集成模型易过拟合，线性模型更稳健。
2. **手工特征价值**：提升幅度仅0.15%-0.36%，问题在于特征设计偏通用（如“profit”“loss”），未结合金融文本的**领域特异性**，且特征维度单一。
3. **过拟合风险**：5折交叉验证标准差（0.006-0.014）较小，模型稳定性良好，无明显过拟合；若需进一步规避，可通过正则化、特征选择优化。
4. **部署优势**：现有线性模型（LogReg/LinearSVC）本身轻量（模型文件KB级，推理内存几十MB），完全满足部署约束，无需更换“重模型”。


### 二、优先推荐：优化现有线性模型（成本最低，效果立竿见影）
你的核心目标是「提升F1+保持轻量」，**优化现有线性模型+特征工程**是投入产出比最高的方案，无需引入新依赖，部署成本为零。


#### 2.1 特征工程优化（关键突破点）
手工特征提升有限的核心是「通用情感词+单一维度」，需结合金融文本特性设计**精准特征**，以下是可落地的改进方案：

##### （1）扩充金融领域情感词库
现有正负词库偏通用，需补充金融专属术语（基于你的EDA高频词和金融常识）：
```python
# 优化后的情感词库（替换handcrafted_features.py中的positive/negative_words）
positive_words = [
    # 原有基础词
    'profit', 'rise', 'increase', 'growth', 'higher', 'gain',
    'win', 'success', 'improve', 'boost', 'surge', 'soar',
    'strong', 'positive', 'beat', 'exceed',
    # 新增金融专属正面词
    'eps', 'roe', 'dividend', 'revenue', 'hike', 'outperform',
    'upgrade', 'beat_forecast', 'above_expected', 'profit_margin_up',
    'sales_growth', 'cash_flow_positive', 'dividend_increase'
]

negative_words = [
    # 原有基础词
    'loss', 'fall', 'decrease', 'drop', 'cut', 'lower',
    'decline', 'weak', 'negative', 'miss', 'fail', 'warn',
    'lose', 'fell', 'downgrade',
    # 新增金融专属负面词
    'loss_widen', 'revenue_miss', 'dividend_cut', 'eps_drop',
    'downgrade', 'underperform', 'below_expected', 'profit_margin_down',
    'sales_slump', 'cash_flow_negative', 'default_risk'
]
```

##### （2）新增3类高价值特征
基于金融文本的「数字情感」「语境位置」「领域密度」设计特征，提升区分度：
```python
def create_sentiment_features(df):
    """优化后的手工特征函数，新增3类特征"""
    positive_words = [...]  # 上述扩充词库
    negative_words = [...]
    financial_terms = ['eur', 'usd', 'gbp', 'eps', 'roe', 'dividend', 'revenue', 'sales', 'profit', 'loss']  # 金融专属术语

    features = []
    for text in df['news_title']:
        text_lower = str(text).lower()
        words = text_lower.split()

        # 1. 原有基础特征（保留并优化）
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        total_sentiment_words = pos_count + neg_count
        pos_ratio = pos_count / (total_sentiment_words + 1e-10)
        neg_ratio = neg_count / (total_sentiment_words + 1e-10)
        net_sentiment = pos_count - neg_count
        has_strong_positive = int(any(w in text_lower for w in ['soar', 'surge', 'beat_forecast']))
        has_strong_negative = int(any(w in text_lower for w in ['plunge', 'crash', 'loss_widen']))

        # 2. 新增：金融术语密度（细分领域词，比原financial_density更精准）
        finance_count = sum(1 for term in financial_terms if term in text_lower)
        finance_density = finance_count / (len(words) + 1e-10)

        # 3. 新增：数字情感特征（金融文本中数字+趋势词是强信号）
        has_pos_num = int(any(re.search(r'(up|rise|increase|higher) \d+%', text_lower) or
                              re.search(r'\d+% (up|rise)', text_lower)))
        has_neg_num = int(any(re.search(r'(down|fall|decrease|lower) \d+%', text_lower) or
                              re.search(r'\d+% (down|fall)', text_lower)))

        # 4. 新增：情感词位置特征（句首/句尾的情感词权重更高）
        pos_word_at_start = int(words[0] in positive_words) if len(words) > 0 else 0
        neg_word_at_end = int(words[-1] in negative_words) if len(words) > 0 else 0

        features.append([
            pos_count, neg_count, pos_ratio, neg_ratio, net_sentiment,
            has_strong_positive, has_strong_negative,
            finance_count, finance_density,  # 优化+新增领域特征
            has_pos_num, has_neg_num,        # 新增数字情感特征
            pos_word_at_start, neg_word_at_end  # 新增位置特征
        ])

    # 新增特征名称
    feature_names = [
        'pos_word_count', 'neg_word_count', 'pos_ratio', 'neg_ratio', 'net_sentiment',
        'has_strong_positive', 'has_strong_negative',
        'finance_term_count', 'finance_term_density',
        'has_positive_number', 'has_negative_number',
        'pos_word_at_start', 'neg_word_at_end'
    ]
    return pd.DataFrame(features, columns=feature_names)
```

##### （3）文本预处理加强（针对金融文本特性）
修改`text_preprocessor.py`，优化金融领域特殊表达的处理（如“eur100m”“5%growth”）：
```python
def preprocess(self, text, max_words=100):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()

    # 新增：处理金融数字格式（如“eur100m”→“eur 100m”，“5%growth”→“5% growth”）
    text = re.sub(r'([a-z]+)(\d+[a-z%])', r'\1 \2', text)  # 字母+数字（如eur100m→eur 100m）
    text = re.sub(r'(\d+%)([a-z]+)', r'\1 \2', text)       # 数字%+字母（如5%growth→5% growth）

    # 原有处理保留
    text = re.sub(r'\b(\d+)\s*percent\b', r'\1percent', text)
    text = re.sub(r'\b(\d+)\s*pct\b', r'\1percent', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*million\b', r'dollar\1m', text, flags=re.IGNORECASE)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*billion\b', r'dollar\1b', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\w\s\.\,\!\?\-\+\%\$\&]', ' ', text)

    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]

    # 新增：保留金融术语（避免被停用词过滤，如“eps”“roe”）
    financial_terms = {'eur', 'usd', 'gbp', 'eps', 'roe', 'dividend', 'revenue', 'sales'}
    processed_words = []
    for word in words:
        if len(word) < 2 and word not in {'no', 'up', 'in', 'on', 'at', 'to', 'by'}:
            continue
        if word in self.extended_stopwords and word not in financial_terms:  # 金融术语不被停用
            continue
        # 原有词形还原/词干提取保留
        if self.lemmatizer:
            word = self.lemmatizer.lemmatize(word, pos='n')
            word = self.lemmatizer.lemmatize(word, pos='v')
            word = self.lemmatizer.lemmatize(word, pos='a')
        if self.stemmer:
            word = self.stemmer.stem(word)
        processed_words.append(word)
    return ' '.join(processed_words)
```

##### （4）特征选择（减少冗余，避免过拟合）
在模型 pipeline 中加入**特征选择步骤**（如`SelectKBest`），筛选高信息量特征，提升模型效率：
```python
# 以Logistic Regression为例，修改create_logregression_pipeline函数
from sklearn.feature_selection import SelectKBest, chi2

def create_logregression_pipeline(use_handcrafted: bool = True) -> Pipeline:
    if use_handcrafted:
        feature_union = FeatureUnion([
            ('text', Pipeline([
                ('selector', FunctionTransformer(get_text_column, validate=False)),
                ('preprocessor', TextPreprocessorTransformer()),
                ('tfidf', TfidfVectorizer(
                    max_features=5000,  # 先保留更多特征，再筛选
                    ngram_range=(1, 3),
                    min_df=2,  # 过滤低频词（出现<2次的词）
                    use_idf=True
                )),
                ('select_k', SelectKBest(chi2, k=3000))  # 筛选Top3000 TF-IDF特征
            ])),
            ('handcrafted', Pipeline([
                ('selector', FunctionTransformer(get_handcrafted_features, validate=False)),
                ('scaler', StandardScaler()),
                ('select_k_hand', SelectKBest(chi2, k=10))  # 筛选Top10手工特征
            ]))
        ])
        pipeline = Pipeline([
            ('features', feature_union),
            ('classifier', LogisticRegression(
                C=2.0,  # 适当减小C，增强正则化
                penalty='l1',
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ))
        ])
    else:
        # 文本-only pipeline 类似，加入SelectKBest
        ...
    return pipeline
```


#### 2.2 线性模型参数精调
基于优化后的特征，进一步扩大参数搜索范围（以Logistic Regression为例）：
```python
# 修改tune_logregression_hyperparameters中的param_grid
param_grid = {
    'classifier__C': np.logspace(-3, 3, 30),  # 更细的C范围（0.001~1000）
    'classifier__penalty': ['l1', 'l2'],
    'classifier__max_iter': [500, 1000],
    'classifier__class_weight': [None, 'balanced'],
    'features__text__tfidf__max_features': [4000, 5000, 6000],
    'features__text__select_k__k': [2500, 3000, 3500],  # 特征选择的k值也作为参数
    'features__handcrafted__select_k_hand__k': [8, 10, 12]
}
```

**预期效果**：优化后Logistic Regression/LinearSVC的加权F1可提升至**82%-83%** ，且模型更简洁，推理速度不变（仍为毫秒级）。


### 三、次选方案：轻量级集成模型（Voting Classifier）
若优化线性模型后效果仍未达预期，可通过**集成多个轻量线性模型**进一步提升稳定性和效果，且不增加太多资源占用。


#### 3.1 集成思路
选择3个表现最优的轻量基础模型（均为线性模型，避免过拟合），采用**软投票**（基于概率平均）提升效果：
- 基础模型1：优化后的Logistic Regression（F1≈82%）
- 基础模型2：LinearSVC（需开启`probability=True`，支持概率输出）
- 基础模型3：SGDClassifier（loss='log_loss'，即逻辑回归的SGD版本，更轻量）


#### 3.2 代码实现
```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

# 1. 定义3个基础模型（均为优化后的版本）
model1 = LogisticRegression(
    C=2.0, penalty='l1', solver='liblinear', class_weight='balanced', random_state=42
)
model2 = LinearSVC(
    C=0.5, penalty='l2', dual=False, class_weight='balanced', probability=True, random_state=42
)
model3 = SGDClassifier(
    loss='log_loss', penalty='l1', alpha=0.001, class_weight='balanced', max_iter=1000, random_state=42
)

# 2. 构建Voting Classifier（软投票）
voting_clf = VotingClassifier(
    estimators=[
        ('logreg', model1),
        ('linearsvc', model2),
        ('sgd', model3)
    ],
    voting='soft',  # 软投票（基于概率平均）
    weights=[1, 1, 1]  # 权重相等，可根据模型效果调整
)

# 3. 集成到pipeline（与原有特征处理结合）
def create_voting_pipeline() -> Pipeline:
    feature_union = FeatureUnion([  # 复用优化后的特征处理
        ('text', Pipeline([...])),  # 同2.1中的text pipeline
        ('handcrafted', Pipeline([...]))  # 同2.1中的手工特征pipeline
    ])
    pipeline = Pipeline([
        ('features', feature_union),
        ('classifier', voting_clf)
    ])
    return pipeline
```


#### 3.3 资源与效果分析
- **资源占用**：3个线性模型的总内存占用仍≤100MB（模型文件KB级），推理时仅需依次计算3个模型的概率并平均，速度比单个模型慢2-3倍（但仍为毫秒级，1万条数据预测时间≤30秒）。
- **效果预期**：加权F1可提升至**83%-84%** ，5折交叉标准差≤0.01，稳定性优于单个模型。


### 四、进阶方案：轻量级预训练模型（效果跃升）
若需进一步突破85%+的F1，可引入**DistilBERT Tiny**（轻量级预训练模型），专为情感分析优化，且满足资源约束。


#### 4.1 模型选择理由
- **体积小**：DistilBERT Tiny（如`distilbert-base-uncased-finetuned-sst-2-english`）体积仅≈100MB，远小于BERT（400MB+）。
- **速度快**：推理速度比BERT快60%，CPU单次推理≤20ms/条。
- **效果好**：预训练模型能理解上下文（如“profit fell less than expected”这类歧义句），金融情感分析F1可达85%-88%。


#### 4.2 部署资源验证
| 资源项                | 约束要求       | 实际占用       | 满足情况 |
|-----------------------|----------------|----------------|----------|
| Docker镜像大小        | ≤4GB           | ≈2GB（优化后） | ✅        |
| 运行时内存            | ≤900MB         | ≈200-300MB     | ✅        |
| 预测时间（1万条数据） | 无明确限制     | ≈200秒         | ✅（5%评分影响小） |


#### 4.3 代码实现（FastAPI部署）
##### （1）模型加载与预测
```python
# sentiment_model.py
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class LightweightSentimentModel:
    def __init__(self):
        # 加载轻量级预训练模型（本地下载，避免容器联网）
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=True
        )

    def predict(self, news_text: str) -> dict:
        # 模型预测（映射为-1/1标签）
        result = self.pipe(news_text)[0]
        positive_prob = result[1]['score']  # 正面概率
        negative_prob = result[0]['score']  # 负面概率
        sentiment = 1 if positive_prob > negative_prob else -1
        return {
            "sentiment": str(sentiment),
            "probability": str(max(positive_prob, negative_prob))
        }
```

##### （2）FastAPI服务（轻量、高性能）
```python
# main.py（API部署文件）
from fastapi import FastAPI
from pydantic import BaseModel
from sentiment_model import LightweightSentimentModel

app = FastAPI()
model = LightweightSentimentModel()  # 初始化模型（仅加载1次）

# 定义输入格式
class NewsText(BaseModel):
    news_text: str

# 情感分析接口（符合Task2要求的端点）
@app.post("/predict_sentiment")
def predict_sentiment(data: NewsText):
    result = model.predict(data.news_text)
    return result

# 启动命令：uvicorn main:app --host 0.0.0.0 --port 5724
```


#### 4.4 Docker镜像优化（关键！控制大小）
采用**多阶段构建**，仅保留运行必需的文件，避免冗余依赖：
```dockerfile
# Dockerfile
# 第一阶段：构建环境（下载模型+安装依赖）
FROM python:3.9-slim AS builder
WORKDIR /app

# 安装依赖（--no-cache-dir减少缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载模型到本地（容器运行时不联网）
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); \
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english'); \
    tokenizer.save_pretrained('./model'); \
    model.save_pretrained('./model')"

# 第二阶段：运行环境（仅复制必需文件）
FROM python:3.9-slim
WORKDIR /app

# 复制依赖（从构建阶段）
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app/model /app/model  # 复制本地模型

# 复制代码
COPY main.py /app/
COPY sentiment_model.py /app/

# 暴露端口（符合Task2要求的5724）
EXPOSE 5724

# 启动API（禁用GPU，仅用CPU）
ENV TRANSFORMERS_OFFLINE=1  # 离线模式（不联网下载模型）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5724", "--workers", "2"]
```

`requirements.txt`（仅保留必需依赖）：
```
fastapi==0.104.1
uvicorn==0.24.0
transformers==4.35.2
torch==2.1.0+cpu  # CPU版本PyTorch，比GPU版本小80%
pydantic==2.5.2
```

**镜像构建命令**：
```bash
docker build -t sentiment-api:v1 .
# 构建后镜像大小≈2GB，满足≤4GB约束
```


### 五、过拟合解决方案
若优化后仍担心过拟合，可通过以下3个方向验证与缓解：
1. **验证过拟合与否**：对比训练集与验证集F1，若差距≤2%（如训练84%、验证82%），无明显过拟合；若差距≥5%，需调整。
2. **增强正则化**：减小线性模型的`C`（如LogReg的C从2.0→1.0），或增加SGDClassifier的`alpha`（从0.001→0.01）。
3. **文本数据增强**：对训练集进行同义词替换（如“rise”→“increase”，用`nltk.wordnet`），避免修改金融术语（如“eur”“eps”），代码示例：
```python
from nltk.corpus import wordnet
import random

def augment_text(text):
    words = text.split()
    augmented_words = []
    for word in words:
        # 仅替换情感词，不替换金融术语
        if word in positive_words or word in negative_words:
            synonyms = [syn.lemma() for syn in wordnet.synsets(word) if syn.pos() == 'v']
            if synonyms and random.random() < 0.3:  # 30%概率替换
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# 对训练集应用增强（扩大数据量）
df_train['augmented_title'] = df_train['news_title'].apply(augment_text)
df_augmented = df_train.copy()
df_augmented['news_title'] = df_augmented['augmented_title']
df_train = pd.concat([df_train, df_augmented], ignore_index=True)  # 数据量翻倍
```


### 六、方案优先级与落地建议
| 方案                | 预期F1  | 部署复杂度 | 资源占用 | 推荐优先级 |
|---------------------|---------|------------|----------|------------|
| 优化线性模型+特征   | 82%-83% | 低         | 极低     | 1（首选）  |
| 轻量级Voting集成    | 83%-84% | 中         | 低       | 2（次选）  |
| DistilBERT Tiny     | 85%-88% | 中高       | 中       | 3（进阶）  |


#### 落地步骤建议
1. **第一步**：先优化特征工程（按2.1修改`handcrafted_features.py`和`text_preprocessor.py`），重新训练Logistic Regression/LinearSVC，验证效果。
2. **第二步**：若F1<82%，加入Voting集成（按3.2实现），提升稳定性。
3. **第三步**：若需更高效果，尝试DistilBERT Tiny（按4.3-4.4部署），重点优化Docker镜像。
4. **部署测试**：本地启动API后，用`curl`测试响应时间和内存：
   ```bash
   # 测试API
   curl -X POST "http://localhost:5724/predict_sentiment" -H "Content-Type: application/json" -d '{"news_text": "Eur 100m profit rise beats forecast"}'
   # 监控内存（容器名替换为实际名称）
   docker stats <container_id>
   ```


通过以上方案，可在满足「轻量部署约束」的前提下，最大化情感分析的加权F1得分，同时控制预测时间损失，兼顾Task2的评分权重（F1占20%，时间占5%）。

## deepseek

根据你的项目描述、当前进展和部署限制，我来分析并提出改进建议：

### 📊 当前问题分析

1. **模型性能瓶颈（80.9% F1）**：
   - 特征表达能力有限（TF-IDF + 手工特征）
   - 传统机器学习对文本语义理解不足
   - 手工特征提升有限（仅0.0019-0.0036提升）

2. **部署限制挑战**：
   - 内存 ≤ 900MB，镜像 ≤ 4GB
   - 无GPU/网络访问
   - 预测时间影响评分（5%权重）

### 🚀 轻量级改进方案推荐

#### 方案一：FastText（最佳平衡）
```python
# fasttext_try.py
import fasttext
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# 数据格式转换：__label__1 text...
def prepare_fasttext_format(df, text_col='news_title', label_col='sentiment'):
    texts = []
    for idx, row in df.iterrows():
        label = 1 if row[label_col] == 1 else -1
        text = row[text_col].replace('\n', ' ').replace('\r', ' ')
        texts.append(f"__label__{label} {text}")
    return '\n'.join(texts)

# 交叉验证训练
def fasttext_cv(df, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['news_title'], df['sentiment'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # 准备数据
        train_data = prepare_fasttext_format(train_df)
        with open(f'fasttext_train_fold{fold}.txt', 'w') as f:
            f.write(train_data)

        # 训练模型（调参重点）
        model = fasttext.train_supervised(
            input=f'fasttext_train_fold{fold}.txt',
            lr=0.8,            # 学习率
            epoch=50,          # 迭代次数
            wordNgrams=2,      # n-gram特征
            dim=100,           # 词向量维度
            loss='softmax',    # 损失函数
            minCount=2,        # 最小词频
            verbose=2
        )

        # 预测
        predictions = []
        for text in val_df['news_title']:
            pred = model.predict(text, k=1)[0][0]
            pred_label = 1 if '__label__1' in pred else -1
            predictions.append(pred_label)

        # 计算F1
        f1 = f1_score(val_df['sentiment'], predictions, average='weighted')
        scores.append(f1)

    return np.mean(scores), scores
```

**优势**：
- 模型极小（通常<100MB）
- 训练和推理极快
- 内置n-gram特征
- 对短文本效果好
- 支持增量训练

#### 方案二：DistilBERT + 蒸馏（轻量BERT）
```python
# distilbert_try.py
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

class DistilBertSentimentClassifier:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )

    def train(self, train_texts, train_labels, val_texts, val_labels):
        # 精简训练参数
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,           # 少量epochs
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=False,  # 不使用混合精度节省内存
            gradient_accumulation_steps=1,
        )

        # 训练...
```

**优势**：
- 相比BERT体积小40%，速度快60%
- 保持90%以上的BERT性能
- 模型大小~250MB
- 推理时间快

#### 方案三：Ensemble of Simple Models（集成学习）
```python
# ensemble_try.py
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def create_lightweight_ensemble():
    # 1. TF-IDF特征（适度规模）
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    # 2. 轻量级基模型
    models = [
        ('nb', MultinomialNB(alpha=0.1)),
        ('lr', LogisticRegression(
            C=1.0,
            solver='liblinear',
            max_iter=500,
            penalty='l2'
        )),
        ('svm_linear', LinearSVC(
            C=0.5,
            penalty='l2',
            dual=False,
            max_iter=1000
        ))
    ]

    # 3. 软投票集成
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        weights=[1, 1.5, 1]  # 调整权重
    )

    return Pipeline([
        ('tfidf', vectorizer),
        ('ensemble', ensemble)
    ])
```

### 📈 具体实施建议

#### 阶段一：快速验证（1-2天）
1. **先试FastText**：
   ```bash
   pip install fasttext
   # 简单基线，看是否能超过80.9%
   ```

2. **参数调优重点**：
   - `wordNgrams`: [2, 3, 4] (对标题很重要)
   - `dim`: [50, 100, 200] (维度越小越快)
   - `epoch`: [20, 30, 50]
   - `lr`: [0.5, 0.8, 1.0]

#### 阶段二：特征工程增强（并行）
```python
# enhanced_features.py
def create_enhanced_features(df):
    """
    基于EDA的深度特征工程
    """
    features = []

    for text in df['news_title']:
        text_lower = str(text).lower()

        # 1. 领域特定词典（基于你的高频词）
        finance_pos_words = ['profit', 'dividend', 'growth', 'expansion']
        finance_neg_words = ['loss', 'cut', 'downgrade', 'warn']

        # 2. 情感强度特征
        strong_pos = ['soar', 'surge', 'skyrocket', 'rocket']
        strong_neg = ['plummet', 'collapse', 'crash', 'plunge']

        # 3. 公司/行业特定特征
        company_indicators = ['finnish', 'malaysia', 'sen', 'eur']

        # 4. 标题结构特征
        has_colon = ':' in text
        has_question = '?' in text
        starts_with_company = any(text_lower.startswith(word) for word in company_indicators)

        # 5. 数字特征
        import re
        numbers = re.findall(r'\d+', text)
        has_percent = '%' in text or 'percent' in text_lower
        has_million = 'million' in text_lower or 'm' in text_lower

        features.append([
            # 原有特征...
            int(has_colon), int(has_question), int(starts_with_company),
            int(has_percent), int(has_million), len(numbers)
        ])

    return features
```

#### 阶段三：模型优化策略
1. **数据增强**（有限数据时特别有效）：
   ```python
   def augment_text(text):
       # 同义词替换（使用轻量级库）
       synonyms = {
           'profit': ['earnings', 'gain', 'income'],
           'loss': ['deficit', 'shortfall', 'decline'],
           # ...
       }
       # 简单数据增强
       return text
   ```

2. **类别平衡**：
   ```python
   from imblearn.over_sampling import RandomOverSampler
   from imblearn.under_sampling import RandomUnderSampler
   
   # 或使用class_weight参数
   model = LogisticRegression(class_weight='balanced')
   ```

#### 阶段四：部署优化
1. **模型压缩**：
   ```python
   # 对于逻辑回归
   def compress_model(model, feature_names, top_n=100):
       # 只保留最重要的特征
       coef = model.coef_[0]
       important_idx = np.argsort(np.abs(coef))[-top_n:]
       # 重新训练精简模型
   ```

2. **缓存机制**：
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def predict_cached(text):
       return model.predict([text])[0]
   ```

### 🎯 优先级建议

按以下顺序尝试：

1. **立即尝试**：FastText（最轻量，部署最容易）
2. **备选方案**：轻量级集成（LogisticRegression + NaiveBayes + LinearSVC）
3. **性能追求**：DistilBERT（如果FastText达不到82%+）
4. **极致轻量**：ONNX Runtime + 量化模型

### 📝 预期收益

| 方法 | 预期F1提升 | 内存占用 | 推理时间 | 部署复杂度 |
|------|-----------|----------|----------|------------|
| FastText | +1-3% | <100MB | <10ms | 低 |
| 轻量集成 | +0.5-2% | 200-400MB | <20ms | 中 |
| DistilBERT | +2-5% | 300-500MB | 50-100ms | 中高 |

**推荐路线**：从FastText开始，如果效果达到82%+就采用，否则尝试轻量集成，最后考虑DistilBERT。

### 🛠️ 快速启动代码

```python
# quick_start.py
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 1. 加载数据
df = pd.read_excel('training_news-sentiment.xlsx')

# 2. 准备FastText格式
def to_fasttext_format(row):
    label = '__label__1' if row['sentiment'] == 1 else '__label__-1'
    return f"{label} {row['news_title']}\n"

with open('fasttext_data.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(to_fasttext_format(row))

# 3. 训练
model = fasttext.train_supervised(
    input='fasttext_data.txt',
    lr=0.8,
    epoch=30,
    wordNgrams=2,
    dim=100,
    verbose=2
)

# 4. 测试
test_texts = ["Finnish group sees profit rise"]
predictions = model.predict(test_texts)
print(predictions)
```

建议你先用这个简单脚本测试FastText的基线性能，然后决定后续方向。

---

## 同义词数据增强

```
def augment_text(text):
    # 同义词替换（使用轻量级库）
    synonyms = {
        'profit': ['earnings', 'gain', 'income'],
        'loss': ['deficit', 'shortfall', 'decline'],
        'pct': ['percent', 'percentage', '%'],
    }
    # 简单数据增强
    return text
```

3. **文本数据增强**：对训练集进行同义词替换（如“rise”→“increase”，用`nltk.wordnet`），避免修改金融术语（如“eur”“eps”），代码示例：
```python
from nltk.corpus import wordnet
import random

def augment_text(text):
    words = text.split()
    augmented_words = []
    for word in words:
        # 仅替换情感词，不替换金融术语
        if word in positive_words or word in negative_words:
            synonyms = [syn.lemma() for syn in wordnet.synsets(word) if syn.pos() == 'v']
            if synonyms and random.random() < 0.3:  # 30%概率替换
                augmented_words.append(random.choice(synonyms))
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    return ' '.join(augmented_words)

# 对训练集应用增强（扩大数据量）
df_train['augmented_title'] = df_train['news_title'].apply(augment_text)
df_augmented = df_train.copy()
df_augmented['news_title'] = df_augmented['augmented_title']
df_train = pd.concat([df_train, df_augmented], ignore_index=True)  # 数据量翻倍
```

## distilbert trials

### baseline

- https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/blob/main/config.json | config.json &middot; distilbert/distilbert-base-uncased-finetuned-sst-2-english at main

### method

- https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis | mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis &middot; Hugging Face
- https://huggingface.co/spaces/sway0604/news_sentiment | News Sentiment - a Hugging Face Space by sway0604
- https://www.kaggle.com/code/dhaouadiibtihel98/fine-tuning-distilbert-for-sentiment-analysis | Fine-Tuning DistilBERT for Sentiment Analysis
- https://www.kaggle.com/code/joshplnktt/sentiment-analysis-w-distilbert | Sentiment Analysis w/ DistilBERT
- https://www.kaggle.com/code/ocanaydin/financial-sentiment-bert | financial_sentiment_BERT
- https://github.com/vedavyas0105/Financial-Sentiment-Distillation | vedavyas0105/Financial-Sentiment-Distillation: This project leverages knowledge distillation to create a lightweight yet powerful sentiment analysis model, tailored specifically for financial news data. Using a teacher-student approach, the project distills knowledge from a large FinBERT model into a compact DistilBERT-based student model, balancing performance and efficiency.
- https://medium.com/@choudhary.man/fine-tuning-distilbert-for-financial-sentiment-analysis-a-practical-implementation-d6df80e8340f | Fine-Tuning DistilBERT for Financial Sentiment Analysis: A Practical Implementation | by Manish Bansilal Choudhary | Medium
- https://github.com/Ramy-Abdulazziz/Financial-Sentiment-Analysis | Ramy-Abdulazziz/Financial-Sentiment-Analysis: LLM's trained and fine tuned for financial sentiment analysis
- https://huggingface.co/AdityaAI9/distilbert_finance_sentiment_analysis#:~:text=A%20fine-tuned%20DistilBERT%20model%20for%20financial%20text%20sentiment,statements%20into%20three%20categories%3A%20positive%2C%20negative%2C%20and%20neutral. | AdityaAI9/distilbert_finance_sentiment_analysis &middot; Hugging Face

## dataset

- https://huggingface.co/datasets/takala/financial_phrasebank | takala/financial_phrasebank &middot; Datasets at Hugging Face
- https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment | zeroshot/twitter-financial-news-sentiment &middot; Datasets at Hugging Face
