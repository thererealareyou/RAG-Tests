# src/trainer.py

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


class SarcasmPhilosopherTrainer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        dataset_path: str = "data/dataset.csv",
        sep: str = "|",
        output_dir: str = "models/lora-sarcasm-philosopher",
        num_epochs: int = 5,
        learning_rate: float = 2e-4,
        per_device_batch_size: int = 1,
        gradient_accumulation_steps: int = 16,
        save_steps: int = 100,
        logging_steps: int = 10,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.sep = sep
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.save_steps = save_steps
        self.logging_steps = logging_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Загрузка модели и токенизатора
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.dataset = self._load_and_format_dataset()

        self._setup_lora()

        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=50,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            learning_rate=self.learning_rate,
            fp16=True,
            logging_dir=f"{self.output_dir}/logs",
            report_to=None
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset,
            peft_config=self.peft_config
        )

    def _load_and_format_dataset(self):
        df = pd.read_csv(self.dataset_path, sep=self.sep)

        def convert_to_messages(row):
            messages = [
                {"role": "system", "content": "Ты — саркастический философ. Отвечай с иронией, цинизмом и глубоким смыслом."},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
            return {"messages": messages}

        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(convert_to_messages)
        return dataset

    def _formatting_func(self, examples):
        texts = []
        for messages in examples["messages"]:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return texts

    def _setup_lora(self):
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def train(self):
        print("🚀 Начинаю обучение...")
        self.trainer.train()
        print("✅ Обучение завершено!")

    def save_model(self):
        self.trainer.save_model()
        self.trainer.tokenizer.save_pretrained(self.output_dir)
        print(f"💾 Модель сохранена в: {self.output_dir}")