from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_model")

model.push_to_hub("riddlee/fake-review-detector-bert")
tokenizer.push_to_hub("riddlee/fake-review-detector-bert")