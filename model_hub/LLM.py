import sys
import time
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from tqdm import tqdm


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.timing_stats = {
            "decode_update_kv_cache": 0.0,
            "decode_attention": 0.0,
        }
        self.prefill_chunk_size = 1

    def _update_prefill_chunk_size(self, cfg):
        """Update prefill_chunk_size from config, falling back to batch_size."""
        chunk = None
        if isinstance(cfg, dict):
            chunk = cfg.get("prefill_chunk_size")
            if chunk is None:
                attn_cfg = cfg.get(getattr(self, "attention_type", None), {})
                if isinstance(attn_cfg, dict):
                    chunk = attn_cfg.get("prefill_chunk_size")
        if isinstance(chunk, int) and chunk > 0:
            self.prefill_chunk_size = min(chunk, self.batch_size)
        else:
            self.prefill_chunk_size = min(self.prefill_chunk_size or 1, self.batch_size)

    def sample_next_token(self, logits, temperature=1.0, top_p=0.0, top_k=0):
        """Sample next token with temperature, top-k, and top-p (nucleus) filtering."""
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)

        if temperature < 1e-5:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift right to keep the first token that exceeds the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)

        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print("[Warning] NaN or Inf in probs, falling back to argmax")
            return logits.argmax(dim=-1)

        prob_sum = probs.sum(dim=-1)
        if (prob_sum <= 0).any():
            print("[Warning] Invalid probability distribution, falling back to argmax")
            return logits.argmax(dim=-1)

        try:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        except RuntimeError as e:
            print(f"[Warning] Multinomial sampling failed: {e}, falling back to argmax")
            next_token = logits.argmax(dim=-1)

        return next_token

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        temp_hidden_states = hidden_states.clone()

        for start_idx in range(0, seq_len, 8192 // bsz):
            end_idx = min(seq_len, start_idx + 8192 // bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(
                temp_hidden_states[:, start_idx:end_idx, :],
                layer.input_layernorm_variance_epsilon,
                layer.input_layernorm_weight
            )

        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        torch.cuda.empty_cache()
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)

        key_states, value_states = self.kv_cache.prefill_update_kv_cache(
            query_states, key_states, value_states, layer_idx, start_bdx
        )
        torch.cuda.empty_cache()

        temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states
        torch.cuda.empty_cache()

        # num_heads * head_dim may differ from hidden_size (e.g. Qwen3 MoE)
        attn_out_dim = temp_attn_out.shape[-2] * temp_attn_out.shape[-1]
        temp_attn_out = temp_attn_out.reshape(bsz, seq_len, attn_out_dim)
        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, attn_out_dim)
        del temp_attn_out
        torch.cuda.empty_cache()

        residual = hidden_states.clone()

        for start_idx in range(0, seq_len, 8192 // bsz):
            end_idx = min(seq_len, start_idx + 8192 // bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(
                hidden_states[:, start_idx:end_idx, :],
                layer.post_attention_layernorm_variance_epsilon,
                layer.post_attention_layernorm_weight
            )
            hidden_states[:, start_idx:end_idx, :] = self.mlp(
                hidden_states[:, start_idx:end_idx, :], layer
            )

        hidden_states += residual

        del residual
        torch.cuda.empty_cache()

        return hidden_states

    def layer_decode(self, layer_idx, hidden_states):
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(
            hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight
        )

        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)

        start = time.perf_counter()
        nvtx.range_push("decode_update_kv_cache")
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        nvtx.range_pop()
        self.timing_stats["decode_update_kv_cache"] += time.perf_counter() - start

        attn_start = time.perf_counter()
        nvtx.range_push("decode_attention")
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        nvtx.range_pop()
        self.timing_stats["decode_attention"] += time.perf_counter() - attn_start

        # num_heads * head_dim may differ from hidden_size (e.g. Qwen3 MoE)
        attn_out_dim = attn_out.shape[-2] * attn_out.shape[-1]
        attn_out = attn_out.reshape(bsz, seq_len, attn_out_dim)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, attn_out_dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(
            hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight
        )
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states

    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        chunk_size = min(self.prefill_chunk_size or 1, bsz)
        for start_bdx in range(0, bsz, chunk_size):
            end_bdx = min(bsz, start_bdx + chunk_size)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])
            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    torch.cuda.empty_cache()
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]

        last_hidden_states = self.layernorm(
            last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight
        )
        logits = self.lm(last_hidden_states)

        return logits

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)

        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)

        return logits

    def inference(self, inputs_ids):
        outputs_ids = []
        output_ids = []

        print("Start prefilling ...")
        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()
        torch.cuda.synchronize()
        prefill_end = time.time()
        print(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s\n")

        print("Start decoding ...")
        decode_start = time.time()
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None
        actual_generated = 0

        disable_pbar = not sys.stderr.isatty()
        decode_pbar = tqdm(
            total=self.max_new_length - 1, desc="Decoding", unit="token",
            ncols=100, leave=False, disable=disable_pbar, file=sys.stderr
        )

        for step_idx in range(self.max_new_length - 1):
            decode_pbar.update(1)

            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            actual_generated += 1

            if not getattr(self, 'disable_early_stop', False):
                if eos_token_id is not None and output_ids[0].item() == eos_token_id:
                    print(f"\n[Early Stop] EOS token detected at step {step_idx + 1}")
                    break

        decode_pbar.close()

        torch.cuda.synchronize()
        decode_end = time.time()

        prefill_time = prefill_end - prefill_start
        decode_time = decode_end - decode_start
        total_time = prefill_time + decode_time

        avg_latency = round(decode_time * 1000 / actual_generated, 2) if actual_generated > 0 else 0
        throughput = round(self.batch_size * actual_generated / decode_time, 2) if decode_time > 0 else 0

        print(
            f"\nDecoding latency: {avg_latency} ms/step, "
            f"Throughput: {throughput} tokens/s, "
            f"Generated: {actual_generated} tokens\n"
        )

        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()

        timing_stats = {
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'actual_generated': actual_generated,
            'avg_latency_ms': avg_latency,
            'throughput': throughput,
        }

        return outputs_ids, timing_stats

    def inference_cot(self, inputs_ids):
        """Sampling-based inference for chain-of-thought generation."""
        outputs_ids = []
        output_ids = []

        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = self.sample_next_token(
            logits, self.temperature, self.top_p, self.top_k
        ).unsqueeze(-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        print(f"Prefilling latency: {round((prefill_end - prefill_start), 4)} s")

        decode_start = time.time()

        eos_token_id = self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None
        actual_generated = 0

        disable_pbar = not sys.stderr.isatty()
        decode_pbar = tqdm(
            total=self.max_new_length - 1, desc="Decoding", unit="token",
            ncols=100, leave=False, disable=disable_pbar, file=sys.stderr
        )

        for step_idx in range(self.max_new_length - 1):
            decode_pbar.update(1)

            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = self.sample_next_token(
                logits, self.temperature, self.top_p, self.top_k
            ).unsqueeze(-1)
            outputs_ids.append(output_ids)
            actual_generated += 1

            if not getattr(self, 'disable_early_stop', False):
                if eos_token_id is not None and output_ids[0, 0].item() == eos_token_id:
                    print(f"\n[Early Stop] EOS token detected at step {step_idx + 1}")
                    break

        decode_pbar.close()

        torch.cuda.synchronize()
        decode_end = time.time()
        decode_time = decode_end - decode_start
        avg_latency = round(decode_time * 1000 / actual_generated, 2) if actual_generated > 0 else 0
        throughput = round(self.batch_size * actual_generated / decode_time, 2) if decode_time > 0 else 0

        print(
            f"\nDecoding latency: {avg_latency} ms/token, "
            f"Throughput: {throughput} tokens/s, "
            f"Generated: {actual_generated} tokens\n"
        )

        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()

        prefill_time = prefill_end - prefill_start
        total_time = prefill_time + decode_time
        timing_stats = {
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'actual_generated': actual_generated,
            'avg_latency_ms': avg_latency,
            'throughput': throughput,
        }

        return outputs_ids, timing_stats

    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None,
                 temperature=0.0, top_p=0.0, top_k=0, disable_early_stop=False):
        """Run LLM inference (prefill + decode).

        Args:
            attention_type: Attention backend to use (e.g. 'PolarANN').
            inputs_ids: Input token ids tensor.
            attention_masks: Attention mask tensor.
            max_new_length: Maximum number of tokens to generate.
            attn_config: Optional KV cache configuration dict.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold (0.0 = disabled).
            top_k: Top-k sampling (0 = disabled).
            disable_early_stop: If True, ignore EOS tokens (for throughput benchmarks).
        """
        self.disable_early_stop = disable_early_stop

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
            f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        self.prefill_chunk_size = min(self.prefill_chunk_size or 1, self.batch_size)

        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks
        torch.cuda.empty_cache()

        print("Allocating KV cache ...")
        self.init_kv_cache(input_length, valid_start, attn_config)

        use_cot_mode = (temperature > 1e-5)

        if use_cot_mode:
            print(f"[Sampling mode] temperature={temperature}, top_p={top_p}, top_k={top_k}")
            outputs = self.inference_cot(inputs_ids)
        else:
            print("[Greedy mode] argmax decoding")
            outputs = self.inference(inputs_ids)

        return outputs
