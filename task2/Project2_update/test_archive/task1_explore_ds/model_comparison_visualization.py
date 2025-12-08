# [file name]: model_comparison_visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def visualize_model_comparison(benchmark_results):
    """可视化模型比较结果"""
    # 准备数据
    models = list(benchmark_results.keys())
    mean_scores = [benchmark_results[m]['mean_f1'] for m in models]
    std_scores = [benchmark_results[m]['std_f1'] for m in models]

    # 创建DataFrame
    df_scores = pd.DataFrame({
        'Model': models,
        'Mean F1 Score': mean_scores,
        'Std': std_scores
    }).sort_values('Mean F1 Score', ascending=True)

    # 绘制横向条形图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df_scores['Model'], df_scores['Mean F1 Score'],
                    xerr=df_scores['Std'], capsize=5, alpha=0.7)

    # 添加数值标签
    for i, (score, std) in enumerate(zip(df_scores['Mean F1 Score'], df_scores['Std'])):
        plt.text(score + 0.01, i, f'{score:.4f} (±{std:.4f})',
                va='center', fontsize=10)

    plt.xlabel('加权F1分数')
    plt.title('模型性能比较')
    plt.xlim([0, 1.1])
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df_scores

def analyze_error_patterns(model_pipeline, X_test, y_test):
    """分析错误模式"""
    # 预测
    y_pred = model_pipeline.predict(X_test)

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'text': X_test,
        'true': y_test,
        'pred': y_pred,
        'correct': y_test == y_pred
    })

    # 分析错误类型
    errors_df = results_df[~results_df['correct']]

    if len(errors_df) > 0:
        print(f"\n错误分析 ({len(errors_df)} 个错误样本):")

        # 按错误类型分组
        error_types = errors_df.groupby(['true', 'pred']).size().reset_index()
        error_types.columns = ['真实标签', '预测标签', '数量']

        print("\n错误类型分布:")
        print(error_types)

        # 可视化错误矩阵
        plt.figure(figsize=(8, 6))

        # 创建错误矩阵
        error_matrix = error_types.pivot(index='真实标签',
                                        columns='预测标签',
                                        values='数量').fillna(0)

        sns.heatmap(error_matrix, annot=True, fmt='g', cmap='Reds')
        plt.title('错误分类矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        plt.savefig('error_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 显示典型错误示例
        print("\n典型错误示例:")
        for (true_label, pred_label), group in errors_df.groupby(['true', 'pred']):
            if len(group) > 0:
                print(f"\n真实标签={true_label}, 预测标签={pred_label}:")
                for i in range(min(2, len(group))):
                    text = group.iloc[i]['text']
                    print(f"  示例 {i+1}: {text[:100]}...")

    return results_df, errors_df

def visualize_feature_importance(model_pipeline, top_n=20):
    """可视化特征重要性"""
    if 'clf' in model_pipeline.named_steps and hasattr(model_pipeline.named_steps['clf'], 'coef_'):
        vectorizer = model_pipeline.named_steps['tfidf']
        classifier = model_pipeline.named_steps['clf']

        # 获取特征名称
        feature_names = vectorizer.get_feature_names_out()

        # 获取系数
        if len(classifier.coef_.shape) == 1:
            # 二分类
            coefficients = classifier.coef_[0]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coef': np.abs(coefficients)
            }).sort_values('abs_coef', ascending=False).head(top_n)

            # 绘制特征重要性
            plt.figure(figsize=(12, 8))
            colors = ['red' if coef < 0 else 'green' for coef in importance_df['coefficient']]
            plt.barh(range(len(importance_df)), importance_df['abs_coef'], color=colors)
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('特征重要性 (系数绝对值)')
            plt.title(f'Top {top_n} 特征重要性 (红色:负面, 绿色:正面)')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()

            # 打印最重要的特征
            print(f"\n最重要的 {top_n} 个特征:")
            for idx, row in importance_df.iterrows():
                sentiment = "负面" if row['coefficient'] < 0 else "正面"
                print(f"  {row['feature']}: {row['coefficient']:.4f} ({sentiment})")

    return None

def main():
    # 这里可以加载之前保存的结果进行可视化
    print("模型对比和可视化")
    print("=" * 60)

    # 示例：如果你有benchmark_results，可以这样可视化
    # benchmark_results = joblib.load('benchmark_results.joblib')
    # df_scores = visualize_model_comparison(benchmark_results)

    print("请先运行 improved_sentiment_model.py 生成结果")
    print("然后使用生成的结果进行可视化分析")

if __name__ == "__main__":
    main()
