import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class LlamaLLM(nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-7B-Instruct",
        pretrained_model=None,
        dtype: str | torch.dtype = "bfloat16",
        attn_implementation: str = "sdpa",
        low_cpu_mem_usage: bool = True,
    ):
        super().__init__()

        def _resolve_dtype(v: str | torch.dtype) -> torch.dtype:
            if isinstance(v, torch.dtype):
                return v
            s = str(v).strip().lower()
            if s in {"fp16", "float16", "half"}:
                return torch.float16
            if s in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if s in {"fp32", "float32"}:
                return torch.float32
            return torch.bfloat16

        if pretrained_model is not None:
            self.model = pretrained_model
            self.tokenizer = None  # handled externally when injected
        else:
            model_dtype = _resolve_dtype(dtype)
            print(f"Loading LLM from {model_name}...")
            kwargs = {
                "torch_dtype": model_dtype,
                "low_cpu_mem_usage": bool(low_cpu_mem_usage),
            }
            if attn_implementation:
                kwargs["attn_implementation"] = str(attn_implementation)

            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            except TypeError:
                # Fallback for older transformers that do not support some kwargs.
                kwargs.pop("attn_implementation", None)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                except TypeError:
                    kwargs.pop("low_cpu_mem_usage", None)
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

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
