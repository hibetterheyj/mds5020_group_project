# update log

## change app.py endpoint format

aigc reference: https://chat.deepseek.com/a/chat/s/d077723d-6e50-4e8d-9c1c-948e6e4dca5e

## 📝 **额外建议：**

1. **测试你的模型输出**：

   ```python
   # 检查情感分析模型的输出值范围
   print("模型可能的输出值:", model.classes_)
   ```

2. **确保映射正确**：

   - 情感分析：`-1`=负面，`1`=正面
   - 主题分类：确保18个类别都正确映射到1-18

3. **添加错误处理**：

   ```python
   @app.errorhandler(500)
   def handle_error(e):
       return {"error": "Internal server error"}, 500
   ```

4. **测试API**：

   ```bash
   curl -X POST http://localhost:5724/predict_sentiment \
        -H "Content-Type: application/json" \
        -d '{"news_text": "这是一条测试新闻"}'
   ```
