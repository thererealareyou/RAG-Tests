from datasets import Dataset
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# Загрузка CSV
df = pd.read_csv("dataset.csv", sep='|')
print(df.head())

# Преобразование в формат messages
def convert_to_messages(row):
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]}
    ]
    return {"messages": messages}

dataset = Dataset.from_pandas(df)
dataset = dataset.map(convert_to_messages)

# Загрузка модели и токенизатора
model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Формирование текста из messages
def format_chat(ex):
    messages = ex["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_chat)

# Настройка LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, peft_config)

# Параметры обучения
training_args = TrainingArguments(
    output_dir="./lora-qwen3-1.7b",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    learning_rate=5e-5,
    fp16=True,
    logging_dir="./logs",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config
)

trainer.train()

trainer.save_model()