import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model1 = AutoModelForSeq2SeqLM.from_pretrained("/Users/lihao/Desktop/semeval/T5-model-checkpoint/my_model_final-T5large-with-rouge-max_len=40")
tokenizer1 = AutoTokenizer.from_pretrained('/Users/lihao/Desktop/semeval/T5-model-checkpoint/my_model_final-T5large-with-rouge-max_len=40')

model2 = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-large")
tokenizer2 = AutoTokenizer.from_pretrained("google-t5/t5-large")
# print(model.config)

# text = "translate English to French: How many times have the Los Angeles Dodgers lost the World Series?"
text = "translate English to French: I watched the movie 'The Shawshank Redemption' last night."
inputs1 = tokenizer1(text,return_tensors="pt")
inputs2 = tokenizer2(text,return_tensors="pt")

output1 = model1.generate(inputs1['input_ids'],max_length=100,do_sample=True,top_k=10)
output2 = model2.generate(inputs2['input_ids'],max_length=100,do_sample=True,top_k=10)

decode_output1 = tokenizer1.decode(output1[0],skip_special_tokens=True)
decode_output2 = tokenizer2.decode(output2[0],skip_special_tokens=True)

print("fine-tuning model output: ",decode_output1)
# fine-tuning model output:  Combien de fois les Dodgers de Los Angeles ont-ils perdu en Série mondiale ?
print("origin model output: ",decode_output2)
# origin model output:  Comment nombreux sont les Los Angeles Dodgers à perdre la série du World?

# target:"J'ai regardé le film 'Les Évadés' hier soir."