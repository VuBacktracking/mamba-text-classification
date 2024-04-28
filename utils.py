import numpy as np
import evaluate

# Tải module "accuracy" từ thư viện evaluate.
accuracy = evaluate.load("accuracy")
# Tạo chức năng tiền xử lý để mã hóa văn bản và cắt bớt các chuỗi không dài hơn độ dài đầu vào tối đa của mã thông báo
def preprocess_function(tokenizer, examples):
    samples = tokenizer(examples["text"], truncation = True)
    samples.pop('attention_mask')
    return samples

def compute_metrics(eval_pred):
    predictions, labels =eval_pred
    # Lấy chỉ số của lớp có xác suất cao nhất trong predictions.
    predictions = np.argmax(predictions, axis = 1)
    
    # Sử dụng module "accuracy" để tính độ chính xác dựa trên predictions và labels.
    return accuracy.compute(predictions=predictions , references=labels)