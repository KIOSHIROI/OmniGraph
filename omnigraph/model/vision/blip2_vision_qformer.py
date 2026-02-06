from omnigraph.utils.env import setup_env

setup_env()

import torch
import torch.nn as nn
from transformers import Blip2Model, Blip2Config

class BLIP2VisionQFormer(nn.Module):
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-flan-t5-xl",
        num_query_tokens: int = 32,
        freeze_vision: bool = True,
        freeze_qformer: bool = False,
        pretrained_model=None
    ):
        super().__init__()

        # Load full BLIP-2 but we will only USE part of it
        if pretrained_model is not None:
            blip2 = pretrained_model
        else:
            blip2 = Blip2Model.from_pretrained(model_name)

        # ---- Vision Encoder ----
        self.vision_encoder = blip2.vision_model
        self.vision_ln = blip2.vision_model.post_layernorm

        # ---- Q-Former ----
        self.qformer = blip2.qformer

        # ---- BLIP2 language projection (optional) ----
        # Some BLIP-2 variants expose language_projection on the base model.
        self.language_projection = getattr(blip2, "language_projection", None)

        # Learnable query tokens
        self.query_tokens = nn.Parameter(
            blip2.query_tokens[:, :num_query_tokens, :].clone()
        )

        # Cache dims for downstream modules
        self.qformer_hidden_dim = self.query_tokens.size(-1)
        self.language_projection_out_dim = (
            self.language_projection.out_features
            if self.language_projection is not None and hasattr(self.language_projection, "out_features")
            else None
        )

        # Optional freezing
        if freeze_vision:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False

        if freeze_qformer:
            for p in self.qformer.parameters():
                p.requires_grad = False

    def forward(self, pixel_values):
        """
        pixel_values: (B, 3, H, W)
        return: vision_tokens (B, Nq, D)
        """

        # 1. Vision encoding
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            image_embeds = self.vision_ln(image_embeds)

        B = image_embeds.size(0)

        # 2. Expand query tokens
        query_tokens = self.query_tokens.expand(B, -1, -1)

        # 3. Q-Former cross-attention
        qformer_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=torch.ones(
                image_embeds.size()[:-1],
                dtype=torch.long,
                device=image_embeds.device,
            ),
            return_dict=True,
        )

        # Output raw visual representation (no projection)
        # Note: The output of QFormer's last_hidden_state is (B, Nq, 768).
        # In standard BLIP-2 architecture, there is usually a projection layer 
        # (language_projection) mapping this to the LLM's dimension.
        # However, the user states that QFormer output is already considered LLM-compatible 
        # or contains the projector implicitly in some contexts, but technically 
        # Blip2Model.qformer output is 768 dim.
        # If the user insists on "QFormer contains Projector", they might be referring to
        # Blip2ForConditionalGeneration which HAS a language_projection.
        # But here we are using Blip2Model which returns the raw QFormer output.
        # We will return the raw output as requested, and handle projection in the main model if needed.
        
        vision_tokens = qformer_outputs.last_hidden_state
        return vision_tokens

    def has_language_projection(self) -> bool:
        return self.language_projection is not None

    def get_qformer_hidden_dim(self) -> int:
        return self.qformer_hidden_dim

if __name__ == "__main__":
    print("Initializing BLIP2VisionQFormer...")
    try:
        try:
            model = BLIP2VisionQFormer()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Real model load failed: {e}")
            print("Falling back to mock model for demonstration.")
            
            # Mock objects to mimic Blip2Model structure
            class MockVisionModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.post_layernorm = nn.LayerNorm(1408) # ViT-g/14 usually has 1408
                def forward(self, pixel_values=None):
                    B = pixel_values.shape[0] if pixel_values is not None else 1
                    # ViT output: (B, 257, 1408)
                    return type('Output', (object,), {'last_hidden_state': torch.randn(B, 257, 1408)})()

            class MockQFormer(nn.Module):
                def __init__(self):
                    super().__init__()
                def forward(self, query_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, return_dict=True):
                    B, Nq, D = query_embeds.shape
                    # QFormer output: (B, Nq, 768)
                    return type('Output', (object,), {'last_hidden_state': torch.randn(B, Nq, 768)})()

            class MockBlip2Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.vision_model = MockVisionModel()
                    self.qformer = MockQFormer()
                    # 32 query tokens, dim 768
                    self.query_tokens = nn.Parameter(torch.randn(1, 32, 768))

            mock_blip2 = MockBlip2Model()
            model = BLIP2VisionQFormer(pretrained_model=mock_blip2)

        # Zero-shot test
        print("Starting zero-shot test...")
        
        # Create dummy pixel values
        # BLIP-2 usually expects (B, 3, 224, 224) or similar depending on processor, let's use standard
        dummy_pixel_values = torch.randn(1, 3, 224, 224)
        print(f"Input shape: {dummy_pixel_values.shape}")
        
        # Run inference
        model.eval()
        with torch.no_grad():
            vision_tokens = model(dummy_pixel_values)
            
        print("Inference successful.")
        print("Output vision tokens shape:", vision_tokens.shape) # Should be (1, 32, 768)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
