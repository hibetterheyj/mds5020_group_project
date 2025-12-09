# external datasets

## takala/financial_phrasebank

> https://huggingface.co/datasets/takala/financial_phrasebank

### Data Instances

```
{ "sentence": "Pharmaceuticals group Orion Corp reported a fall in its third-quarter earnings that were hit by larger expenditures on R&D and marketing .",
  "label": "negative"
}
```

### Data Fields

- sentence: a tokenized line from the dataset
- label: a label corresponding to the class as a string: 'positive', 'negative' or 'neutral'

### Data Splits

There's no train/validation/test split.

However the dataset is available in four possible configurations depending on the percentage of agreement of annotators:

`sentences_50agree`; Number of instances with >=50% annotator agreement: 4846 `sentences_66agree`: Number of instances with >=66% annotator agreement: 4217 `sentences_75agree`: Number of instances with >=75% annotator agreement: 3453 `sentences_allagree`: Number of instances with 100% annotator agreement: 2264

## zeroshot/twitter-financial-news-sentiment

> https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment

### Dataset Description

The Twitter Financial News dataset is an English-language dataset containing an annotated corpus of finance-related tweets. This dataset is used to classify finance-related tweets for their sentiment.

1. The dataset holds 11,932 documents annotated with 3 labels:

```python
sentiments = {
    "LABEL_0": "Bearish",
    "LABEL_1": "Bullish",
    "LABEL_2": "Neutral"
}
```



The data was collected using the Twitter API. The current dataset supports the multi-class classification task.

### Task: Sentiment Analysis

- Data Splits

There are 2 splits: train and validation. Below are the statistics:

| Dataset Split | Number of Instances in Split |
| :-----------: | :--------------------------: |
|     Train     |            9,938             |
|  Validation   |            2,486             |

