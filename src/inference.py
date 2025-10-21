from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ROWInferencer:
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3-1.7B",
        device: str = "auto",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            ),
            device_map=device,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()

    def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
    ):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = model_inputs.input_ids.shape[1]
        output_ids = generated_ids[0][input_len:].tolist()

        try:
            end_think_token_id = 151668
            index = len(output_ids) - output_ids[::-1].index(
                end_think_token_id
            )
        except ValueError:
            index = 0

        thinking = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True
        ).strip()
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True
        ).strip()

        return {
            "question": user_prompt,
            "thinking": thinking,
            "response": content,
        }