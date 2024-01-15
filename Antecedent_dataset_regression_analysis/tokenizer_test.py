from transformers import DistilBertTokenizerFast, DistilBertModel, LongformerTokenizerFast, LongformerModel

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
test_string = "None None None None None None presenting weather forecast presenting weather forecast sign language interpreting answering questions None None flying kite presenting weather forecast presenting weather forecast presenting weather forecast presenting weather forecast sign language interpreting None None None None None None None sign language interpreting None None None None None None presenting weather forecast sign language interpreting playing ukulele playing ukulele playing ukulele playing ukul"
test_tokens = tokenizer(test_string, return_tensors='pt', truncation=True, padding='max_length', max_length=500)

print(test_tokens['input_ids'][0])
print(test_tokens['input_ids'])
