import pandas as pd
import re
import os

# 定义文件路径
TRAIN_INPUT_FILE = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/external/zeroshot/twitter-financial-news-sentiment/sent_train.csv'
VALID_INPUT_FILE = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/external/zeroshot/twitter-financial-news-sentiment/sent_valid.csv'

TRAIN_OUTPUT_FILE = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/external/zeroshot/twitter-financial-news-sentiment/cleaned_train.csv'
VALID_OUTPUT_FILE = '/Users/heyujie/Documents/cuhksz-all-sync/code/mds5020_group_project/task2/data/external/zeroshot/twitter-financial-news-sentiment/cleaned_valid.csv'

# 数据清洗函数
def clean_text(text):
    # 转换为字符串类型
    text = str(text)

    # 清除所有引号（包括开头、结尾和中间）
    text = text.replace('"', '').replace("'", '')

    # 清除首尾空格
    text = text.strip()

    # 清除所有网络链接（包括中间的链接）
    text = re.sub(r'https?://\S+', '', text)

    # 清除"Read more:"及其后面内容
    text = re.sub(r'Read more:.*$', '', text, flags=re.IGNORECASE)

    # 清除"via"及其后面内容
    text = re.sub(r'\s+via\s+.*$', '', text, flags=re.IGNORECASE)

    # 清除@标记的用户
    text = re.sub(r'@\w+', '', text)

    # 清除#开头的话题
    text = re.sub(r'#\w+', '', text)

    # 清除所有$符号
    text = text.replace('$', '')

    # 清除开头的股票代码（格式：XXX - 或 XXX:）
    text = re.sub(r'^[A-Z]+(?:\s+[A-Z]+)*\s*[-:]\s*', '', text)

    # 清除结尾的股票代码（格式如：" GPS"）
    text = re.sub(r'\s+[A-Z]+$', '', text)

    # 清除可能的多余分隔符和空格
    text = re.sub(r'^[-:\s]+', '', text)
    text = re.sub(r'[-:\s]+$', '', text)
    text = re.sub(r'\s+', ' ', text)

    # 清除可能的多余符号
    text = re.sub(r'[`~]+', '', text)

    # 再次清除可能产生的额外空格
    text = text.strip()

    return text

def process_file(input_file, output_file):
    # 读取数据
    df = pd.read_csv(input_file)

    # 过滤掉中立类别（label=2）
    df = df[df['label'] != 2]

    # 清洗文本
    df['text'] = df['text'].apply(clean_text)

    # 转换标签：0→-1，1→1
    df['sentiment'] = df['label'].replace({0: -1, 1: 1})

    # 添加doc_id字段
    df['doc_id'] = range(1, len(df) + 1)

    # 重命名text列为news_title
    df.rename(columns={'text': 'news_title'}, inplace=True)

    # 选择需要的列
    df = df[['doc_id', 'news_title', 'sentiment']]

    # 保存清洗后的数据
    df.to_csv(output_file, index=False)

    print(f"清洗完成！已保存到 {output_file}")
    print(f"清洗后的数据量：{len(df)} 条")

if __name__ == "__main__":
    # print("开始清洗训练集...")
    # process_file(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE)

    print("\n开始清洗验证集...")
    process_file(VALID_INPUT_FILE, VALID_OUTPUT_FILE)

    print("\n所有数据清洗完成！")