import torch
import torch.nn as nn
from hivemind.moe.server.layers.custom_experts import register_expert_class

COND_NUM_TOKENS = 128
CHANNELS = 3
HEIGHT = WIDTH = 256

STEPS = 50

def get_input_example(batch_size: int, *_unused):
    cond_tokens = torch.empty((batch_size, COND_NUM_TOKENS), dtype=torch.int64)
    initial_images = torch.empty((batch_size, 3, HEIGHT, WIDTH), dtype=torch.uint8)
    return (cond_tokens, initial_images)


@register_expert_class("ExampleModule", get_input_example)
class ExampleModule(nn.Module):
    def __init__(self):
        super().__init__()
        print("[DEBUGPRINT] INIT MY MODULE!!")

    def forward(self, cond_tokens: torch.LongTensor, initial_images: torch.ByteTensor):
        print(f"[DEBUGPRINT] RAN FORWARD PASS, {cond_tokens.shape}, {initial_images.shape}!!")
        output_images = initial_images * 10
        return (output_images.to(torch.uint8),) # note: output dtype is important since it affects bandwidth usage!
