import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class LlamaLLM(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", pretrained_model=None):
        super().__init__()
        
        if pretrained_model is not None:
            self.model = pretrained_model
            self.tokenizer = None # Assuming handled externally or not needed if model passed
        else:
            print(f"Loading LLM from {model_name}...")
            # Load with bf16 as requested
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Freeze LLM
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.config = self.model.config

    def forward(self, inputs_embeds, attention_mask=None, labels=None, input_ids=None):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate(self, inputs_embeds, **kwargs):
        return self.model.generate(inputs_embeds=inputs_embeds, **kwargs)
