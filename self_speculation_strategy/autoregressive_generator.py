# autoregressive_generator.py
from dataclasses import dataclass
from typing import Any, List, Optional
import torch
import time

from device_utils import model_input_device
from generation_limits import limit_generation_to_context
from llava_next import count_llava_next_visual_tokens
from utils import (
    Env,
    apply_generation_constraints,
    forward,
    forward_divided,
    decode_next_token,
    GenerationResult,
    forward_multimodal_prefill,
)
from multimodal import (
    get_text_model,
    is_supported_multimodal,
    prepare_multimodal_inputs,
)

class AutoregressiveGenerator:
    def __init__(self, env: Env, coefficients: Optional[tuple] = None):
        self.env = env
        self.model = env.model
        self.coefficients = coefficients
        if coefficients:
             self.c_1, self.c_2, self.c_3 = coefficients
        else:
             self.c_1 = 0
             self.c_2 = 0
             self.c_3 = 0
        
        self.best_tpt_list = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        sample: bool = False,
        reference_output_ids: Optional[List[int]] = None,
        image: Optional[Any] = None,
        video: Optional[Any] = None,
        video_num_frames: int = 64,
        video_max_visual_tokens: int = 8192,
        llava_video_contact_sheets: int = 1,
    ) -> GenerationResult:
        """Generate text autoregressively."""
        multimodal_prefill_inputs = None
        if is_supported_multimodal(self.model):
            enc = prepare_multimodal_inputs(
                self.model,
                self.env.processor,
                prompt,
                image=image,
                video=video,
                video_num_frames=video_num_frames,
                video_max_visual_tokens=video_max_visual_tokens,
                llava_video_contact_sheets=llava_video_contact_sheets,
            )
            multimodal_prefill_inputs = enc
        else:
            enc = self.env.tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"][0].tolist()  # Convert to List[int] like Meta
        initial_len = len(input_ids)
        max_new_tokens, min_new_tokens = limit_generation_to_context(
            self.model,
            initial_len,
            max_new_tokens,
            min_new_tokens,
            label="AR",
        )
        
        # EOS token IDs
        eos_token_ids = []
        if self.env.eos_id is not None:
            eos_token_ids.append(self.env.eos_id)
        
        # Main generation loop (like Meta's generate_token_ids)
        past_key_values = None
        input_device = model_input_device(self.env.model)
        input_ids_tensor = enc["input_ids"].to(input_device)
        if multimodal_prefill_inputs is not None:
            del enc
        output_ids: List[int] = []
        
        self.best_tpt_list = []
        L = len(get_text_model(self.model).layers)

        total_start_time = time.perf_counter()
        for step in range(max_new_tokens):
            if multimodal_prefill_inputs is not None:
                logits, past_key_values = forward_multimodal_prefill(
                    self.model,
                    multimodal_prefill_inputs,
                )
                multimodal_prefill_inputs = None
            else:
                logits, past_key_values = forward(
                    self.model,
                    input_ids_tensor,
                    past_key_values,
                )
            logits = apply_generation_constraints(
                logits,
                output_ids,
                eos_token_ids=eos_token_ids,
                min_new_tokens=min_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )

            next_token, probabilities = decode_next_token(
                logits,
                token_idx = -1,
                sample=sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            
            # Convert to int if tensor
            next_token_id = next_token.item()
                
            # Check for EOS
            if next_token_id in eos_token_ids:
                break
            
            output_ids.append(next_token_id)
            
            # TPT Logging every 50 tokens
            if self.coefficients and (step + 1) % 50 == 0:
                current_length = initial_len + len(output_ids)
                t_mlp = self.c_1
                t_attn = self.c_2 * current_length + self.c_3
                
                cost = L * (t_attn + t_mlp)
                if cost > 0:
                    tpt = 1.0 / cost
                    self.best_tpt_list.append(tpt)

            
            # Update input_ids for next iteration (single token)
            input_ids_tensor = torch.tensor([[next_token_id]], device=input_device)
            
            if step % 1000 == 0:
                print(f"\r[AR] Generated {step} tokens...", end="", flush=True)
        total_time = time.perf_counter() - total_start_time
        
        # Decode generated tokens
        generated_text = self.env.tok.decode(output_ids, skip_special_tokens=True) if output_ids else ""
        
        avg_best_tpt = sum(self.best_tpt_list) / len(self.best_tpt_list) if self.best_tpt_list else None

        return GenerationResult(
            text=generated_text,
            num_output_tokens=len(output_ids),
            output_ids=output_ids,
            num_input_tokens=initial_len,
            num_visual_tokens=count_llava_next_visual_tokens(self.model, input_ids),
            total_time=total_time,
            avg_best_tpt=avg_best_tpt,
        )
