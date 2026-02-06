from omnigraph.utils.env import setup_env

setup_env()

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class VicunaLLM(nn.Module):
    def __init__(self, model_name="lmsys/vicuna-7b-v1.3", pretrained_model=None):
        super().__init__()
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

if __name__ == "__main__":
    print("Initializing VicunaLLM...")
    try:
        try:
            model = VicunaLLM()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Real model load failed: {e}")
            print("Falling back to mock model for demonstration.")
            
            # Mock objects
            class MockConfig:
                vocab_size = 32000
                hidden_size = 4096
                
            class MockCausalLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = MockConfig()
                    self.lm_head = nn.Linear(4096, 32000)
                    
                def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                    batch_size, seq_len = input_ids.shape
                    logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
                    loss = None
                    if labels is not None:
                        loss = torch.tensor(2.5) # Dummy loss
                    return type('Output', (object,), {'logits': logits, 'loss': loss})()
                
                def generate(self, input_ids, max_new_tokens=20, **kwargs):
                    batch_size, seq_len = input_ids.shape
                    new_tokens = torch.randint(0, self.config.vocab_size, (batch_size, max_new_tokens))
                    return torch.cat([input_ids, new_tokens], dim=1)

            mock_llm = MockCausalLM()
            model = VicunaLLM(pretrained_model=mock_llm)

        # Zero-shot test
        print("Starting zero-shot test...")
        
        # Try to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3", use_fast=False)
            dummy_text = "Hello, how are you?"
            inputs = tokenizer(dummy_text, return_tensors="pt")
            print("Tokenizer loaded.")
        except Exception as e:
            print(f"Tokenizer load failed: {e}. Using manual dummy inputs.")
            inputs = {'input_ids': torch.randint(0, 32000, (1, 10)), 'attention_mask': torch.ones(1, 10)}

        # Run forward pass
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("Forward pass successful.")
        print("Logits shape:", outputs.logits.shape)
        
        # Run generate
        print("Testing generation...")
        generated_ids = model.generate(inputs['input_ids'], max_new_tokens=10)
        print("Generation successful.")
        print("Generated sequence length:", generated_ids.shape[1])
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()