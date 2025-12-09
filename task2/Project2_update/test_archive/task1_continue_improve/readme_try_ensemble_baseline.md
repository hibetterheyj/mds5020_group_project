# try_ensemble_baseline.py

1. 创建了集成分类器脚本 ：

   - 创建了 try_ensemble_baseline.py 文件，用于实现和测试不同类型的集成分类器
   - 基于 classifier_comparison_summary.md 中最好的10个分类器的最佳参数构建集成模型
2. 实现了三种集成分类器 ：

   - StackingClassifier ：使用ExtraTreesClassifier作为最终分类器
   - VotingClassifier(hard) ：基于硬投票的集成方法
   - VotingClassifier(soft) ：基于软投票的集成方法
3. 解决了技术问题 ：

   - 由于LinearSVC和SGDClassifier不支持概率预测，使用 CalibratedClassifierCV 包装它们以获得概率输出
   - 确保所有集成分类器都能正常工作
4. 运行结果 ：

   - StackingClassifier ：F1分数为0.8207
   - VotingClassifier_Hard ：F1分数为0.8246
   - VotingClassifier_Soft ：F1分数为0.8195
5. 保存结果 ：

   - 所有集成分类器的结果都已保存到 res 目录中，格式与 run_all_classifiers.py 一致
VotingClassifier_Hard表现最好，F1分数达到0.8246，优于之前最好的单一分类器ExtraTreesClassifier（0.8194）。这表明集成分类器确实可以提高模型的性能。