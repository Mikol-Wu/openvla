import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prismatic.vla.diffusion_action_decoder import DiffusionActionDecoder


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask=None,
        pixel_values=None,
        use_cache=False,
        return_dict=True,
    ):
        # Deterministic logits: higher token ids -> higher logits
        bsz, seq_len = input_ids.shape
        base = torch.arange(self.vocab_size, device=input_ids.device, dtype=torch.float32)
        logits = base[None, None, :].expand(bsz, seq_len, self.vocab_size).contiguous()
        return SimpleNamespace(logits=logits)


def test_diffusion_decoder_shape_and_range():
    vocab_size = 100
    action_start, action_end = 80, 100
    bsz, prompt_len, action_dim = 2, 5, 7

    model = DummyModel(vocab_size)
    decoder = DiffusionActionDecoder(mask_token_id=0)

    input_ids = torch.randint(0, vocab_size, (bsz, prompt_len))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    action_ids = decoder.decode(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        action_dim=action_dim,
        steps=3,
        schedule="linear",
        action_vocab_start=action_start,
        action_vocab_end=action_end,
    )

    assert action_ids.shape == (bsz, action_dim)
    assert int(action_ids.min().item()) >= action_start
    assert int(action_ids.max().item()) < action_end


def test_diffusion_decoder_k1_matches_one_shot():
    vocab_size = 100
    action_start, action_end = 80, 100
    bsz, prompt_len, action_dim = 2, 5, 6

    model = DummyModel(vocab_size)
    decoder = DiffusionActionDecoder(mask_token_id=0)

    input_ids = torch.randint(0, vocab_size, (bsz, prompt_len))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    out_ids = decoder.decode(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=None,
        action_dim=action_dim,
        steps=1,
        schedule="linear",
        action_vocab_start=action_start,
        action_vocab_end=action_end,
    )

    # One-shot parallel prediction (single forward)
    action_mask = torch.full((bsz, action_dim), 0, dtype=torch.long)
    curr_ids = torch.cat([input_ids, action_mask], dim=1)
    curr_attn = torch.ones_like(curr_ids, dtype=torch.bool)
    logits = model(curr_ids, attention_mask=curr_attn).logits[:, -action_dim:, action_start:action_end]
    expected = logits.argmax(dim=-1) + action_start

    assert torch.equal(out_ids, expected)
