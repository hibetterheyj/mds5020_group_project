## bert readme

### 模型方法核心对比表
| 模型类型                    | 5折交叉 F1 性能         | 单样本耗时     | 核心特点                                   |
| --------------------------- | ----------------------- | -------------- | ------------------------------------------ |
| 基线（Logistic Regression） | 0.8024                  | 0.001s         | 基础模型，效率高、性能一般                 |
| 单分类器（ExtraTrees）      | 0.8194                  | 0.001s左右     | 性能优于基线，效率持平                     |
| 集成学习（Voting/Stacking） | 0.8207-0.8246（未调参） | 高效（近基线） | 性能有提升空间，未达上限                   |
| DistilBERT                  | 0.8711                  | 0.02s          | 精度显著领先，大小和内存满足要求，耗时增加 |

---

### 核心选择方向
1. 优先精度：选 DistilBERT（F1 0.8711），需接受单样本耗时从 0.001s 增至 0.02s。
2. 优先效率：选集成学习（当前 0.82+），需确认调参后能否逼近 DistilBERT 精度。
3. 直接放弃变动，选Voting或DistilBERT其中一种

### methods

- https://huggingface.co/distilbert/distilbert-base-uncased/tree/main | distilbert/distilbert-base-uncased &middot; Hugging Face
  - done with app_optimized.py and distilbert_baseline.py

- https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis | mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis &middot; Hugging Face
  - done with distilroberta_baseline.py

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