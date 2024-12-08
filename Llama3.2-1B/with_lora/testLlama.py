# from transformers import AutoTokenizer
# # 加载 Llama3.2 的分词器
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
#
# # 输入的句子
# text = "This is a simple sentence that is around ninety characters long for testing how many tokens it will be split into using the Llama3 tokenizer."
# print(f"Length of the sentence is: {len(text)}")
# print(f"nums of the words is: {len(text.split())}")
# # Length of the sentence is: 141
# # nums of the words is: 25
#
# # 对文本进行分词
# tokens = tokenizer.tokenize(text)
#
# # 打印 token 数量
# print(f"Number of tokens: {len(tokens)}")
# # Number of tokens: 28


# -------测试token数-------
#
# -----------准备peft及lora--------
from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')
# for name,param in model.named_parameters():
#     if 'q_proj' in name:
#         print(name,param.size())
# ---------------正式部分------------
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer
from peft import LoraConfig,get_peft_model
from  datasets import Dataset
from transformers import BitsAndBytesConfig

import evaluate
import numpy as np
import wandb




dataset = load_dataset("skyfuryLH/semeval2025")
lora_config = LoraConfig(
    r = 16,
    lora_alpha=32,
    target_modules=['q_proj','v_proj'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"      # 任务类型
)

# print(dataset)

train_val = dataset['train'].train_test_split(test_size=0.1)
train_data = train_val['train']
val_data = train_val['test']

# print(train_val)
# print(train_data)
# print(train_data[0])

os.environ['WANDB_API_KEY'] = "a464ce6c3b972e3e7090ac20839b9a1daac1b608"
os.environ["WANDB_PROJECT"] = "Llama3-model"
wandb.init()

column_name = train_data.column_names
model_id = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
).to("cuda")
model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

# metric = evaluate.load("sacrebleu")
bleuM = evaluate.load("sacrebleu")
rougeM = evaluate.load("rouge")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model)
translate_dict={
    "ar":"Arabic",
    "de":"German",
    "es":"Spanish",
    "fr":"French",
    "it":"Italian",
    "ja":"Japanese",
}

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#下面这句话必须加上
model.resize_token_embeddings(len(tokenizer))  # 更新模型的token嵌入层
# 因为添加了一个新的token，导致最后推理的时候加载模型维度对不上？
# 考虑如何处理

print(tokenizer.pad_token)

def apply_instruction_template(examples):
    # 现对train进行预处理
    inputs_source = [f'''### Instruction:\nTranslate the following text to {translate_dict[target_locale]}:\n{text}\n''' for target_locale,text in zip(examples['target_locale'],examples['source'])]
    # inputs_source = [f'''Translate the following text to {translate_dict[target_locale]}: {text}''' for target_locale,text in zip(examples['target_locale'],examples['source'])]
    return {"inputs_source":inputs_source}


def preprocess_function(examples):
    # #         '''
    # #         在需要处理一个批次样本时，可以通过设置 batched=True，让 map 传入一组样本的批次。
    # #         此时，自定义函数的参数会变成一个包含每列字段的字典，字段的值为一个列表，代表该批次中所有样本在对应列下的数据。
    # #         这样，用户可以灵活选择是否逐条或批量处理数据。
    # #         :param examples:
    # #         :return:
    # #         '''
    model_inputs = tokenizer(examples['inputs_source'], max_length=32, truncation=True, padding="max_length",
                             return_tensors="pt")
    labels = tokenizer(examples['target'], max_length=32, truncation=True, padding="max_length", return_tensors="pt")
    labels["input_ids"] = [
        [l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels["input_ids"]
    ]
    model_inputs['label'] = labels['input_ids']
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]
    #         print(preds.shape)
    #         batchsize,seq_len,vocab_size
    #         (100, 64, 32128)
    #         input()

    # 将 predictions 从概率转为 token IDs ?
    preds = np.argmax(preds, axis=-1)  # (batch_size, seq_len)

    # 需要额外处理label吗？
    # transformers 中，填充位置通常被标记为 -100，需要将这些位置替换为 tokenizer.pad_token_id，避免在解码时生成错误文本

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = rougeM.compute(predictions=decoded_preds, references=decoded_labels)
    reuslt_bleu = bleuM.compute(predictions=decoded_preds, references=decoded_labels)
    result['bleu'] = reuslt_bleu['score']

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    # result = {k: round(v, 4) for k, v in result.items()}
    return result

# 定义指令模板
instructed_train = train_data.map(apply_instruction_template, batched=True)
instructed_val = val_data.map(apply_instruction_template, batched=True)

tokenized_train = instructed_train.map(preprocess_function, batched=True,remove_columns=column_name)
tokenized_val = instructed_val.map(preprocess_function, batched=True,remove_columns=column_name)

# print(tokenized_train[0])

# 定义训练的超参数
training_args = Seq2SeqTrainingArguments(
    output_dir='./semeval-llama',  # output directory
    save_total_limit=3,  # 只保留最近的 3 个检查点
    #         report_to="wandb",
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=8,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs

    # 频繁的日志记录可能导致内存增加，特别是当模型需要频繁评估时。
    logging_steps=500,
    learning_rate=1e-5,  # Set learning rate
    evaluation_strategy="steps",

    eval_accumulation_steps=100,

    # 使用T5-base时遇到loss为NAN的问题，或许尝试不使用gradient_checkpointing
    # 尝试禁用gradient_checkpointing和FP16，跑出了结果

    # 过多裁剪会导致不能学习到有用的知识
    # max_grad_norm=1.0,  # 最大梯度裁剪，防止梯度爆炸

    save_strategy="epoch",

    # gradient_checkpointing=True,
    # fp16=True, # 开启混合精度
    # gradient_accumulation_steps=8,  # 累积 8 个小批次
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,

    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()

trainer.save_model("./my_model_final-Llama")  # 定义训练的超参数

