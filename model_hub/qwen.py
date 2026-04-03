import gc
import re
import os
import json
import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from .LLM import LLM
from attn_hub import polar_prefill_attn, polar_decode_attn



class QwenLayer:
    """
    A class representing the Qwen layer.
    """

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device
    
    def init_layer(self, hf_qwen_layer):
        self.wq = hf_qwen_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_qwen_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_qwen_layer.self_attn.v_proj.weight.detach()
        
        # Check if bias exists (Qwen3 doesn't have bias)
        if hf_qwen_layer.self_attn.q_proj.bias is not None:
            self.bq = hf_qwen_layer.self_attn.q_proj.bias.detach()
            self.bk = hf_qwen_layer.self_attn.k_proj.bias.detach()
            self.bv = hf_qwen_layer.self_attn.v_proj.bias.detach()
            self.bqkv = torch.cat((self.bq, self.bk, self.bv), dim=0).to(self.device, non_blocking=True)
        else:
            self.bqkv = None
        
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_qwen_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)
        
        # Q/K per-head RMSNorm (Qwen3)
        if hasattr(hf_qwen_layer.self_attn, "q_norm"):
            self.q_norm_weight = hf_qwen_layer.self_attn.q_norm.weight.detach().to(self.device, non_blocking=True)
            self.q_norm_variance_epsilon = hf_qwen_layer.self_attn.q_norm.variance_epsilon
        else:
            self.q_norm_weight = None
            self.q_norm_variance_epsilon = None
        
        if hasattr(hf_qwen_layer.self_attn, "k_norm"):
            self.k_norm_weight = hf_qwen_layer.self_attn.k_norm.weight.detach().to(self.device, non_blocking=True)
            self.k_norm_variance_epsilon = hf_qwen_layer.self_attn.k_norm.variance_epsilon
        else:
            self.k_norm_weight = None
            self.k_norm_variance_epsilon = None
        
        if hasattr(hf_qwen_layer.mlp, 'gate_proj'):
            self.gate_proj = hf_qwen_layer.mlp.gate_proj.weight.detach()
            self.up_proj = hf_qwen_layer.mlp.up_proj.weight.detach()
            self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
            self.down_proj = hf_qwen_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)
            self.mlp_module = None
        else:
            # MoE: keep the original module for forward pass
            self.mlp_module = hf_qwen_layer.mlp
            self.gate_up_proj = None
            self.down_proj = None

        self.input_layernorm_weight = hf_qwen_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_qwen_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_qwen_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_qwen_layer.post_attention_layernorm.variance_epsilon

        # Clean up temporary variables
        del self.wq, self.wk, self.wv
        if hasattr(self, 'gate_proj'):
            del self.gate_proj, self.up_proj
        if hasattr(self, 'bq'):
            del self.bq, self.bk, self.bv


class QwenModel(LLM):
    """
    A class representing the Qwen model.
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        rope_scaling: dict = None,
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)
        self._rope_scaling = rope_scaling

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        # For Qwen3 MoE models, head_dim is explicitly set in config, not calculated
        self.head_dim = getattr(self.config, 'head_dim', self.hidden_size // self.num_heads)
        self.base = self.config.rope_theta
        self.max_position_embeddings = self.config.max_position_embeddings
        self.yarn_factor = 4
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]

        self.init_model()

    
    def _set_cos_sin_cache(self):
        """Build RoPE cos/sin cache using HF-provided inv_freq and attention_scaling."""
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        if hasattr(self, 'attention_scaling') and self.attention_scaling is not None:
            return freqs.cos() * self.attention_scaling, freqs.sin() * self.attention_scaling
        else:
            return freqs.cos(), freqs.sin()


    def init_model(self):
        if self.device_map != 'auto' and self.device_map.startswith('cuda:'):
            device_id = int(self.device_map.split(':')[1])
            torch.cuda.set_device(device_id)
        
        _extra = {}
        if self._rope_scaling is not None:
            _extra["rope_scaling"] = self._rope_scaling
            print(f"[YaRN] rope_scaling={self._rope_scaling}")

        if self.device_map != 'auto':
            hf_qwen = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                dtype=self.dtype, 
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
                **_extra,
            )
        else:
            hf_qwen = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                dtype=self.dtype, 
                trust_remote_code=True,
                device_map='auto',
                **_extra,
            )

        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'
        
        if self.device_map != "auto":   # single GPUs
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): self.device_map})

            self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_qwen.lm_head.weight.detach().to(self.device_map, non_blocking=True)

            self.norm_weight = hf_qwen.model.norm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
            self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for idx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
                qwen_layer = QwenLayer(idx, device=self.device_map)
                qwen_layer.init_layer(hf_qwen_layer)
                self.layers.append(qwen_layer)
                hf_qwen.model.layers[idx] = None

        else:                         # multi GPUs
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {}
            for ldx in range(0, self.num_layers):
                self.layer_mapping.update({str(ldx): f'cuda:{ldx // self.layer_interval}'})

            self.embed_tokens = hf_qwen.model.embed_tokens.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.lm_head = hf_qwen.lm_head.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)

            self.norm_weight = hf_qwen.model.norm.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.norm_variance_epsilon = hf_qwen.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.inv_freq = hf_qwen.model.rotary_emb.inv_freq.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.attention_scaling = hf_qwen.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for ldx, hf_qwen_layer in enumerate(hf_qwen.model.layers):
                qwen_layer = QwenLayer(ldx, device=self.layer_mapping[str(ldx)])
                qwen_layer.init_layer(hf_qwen_layer)
                self.layers.append(qwen_layer)
                hf_qwen.model.layers[ldx] = None

        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()
        
        self._try_compile_moe_layers()

    
    def _try_compile_moe_layers(self):
        """Attempt to torch.compile MoE layers for faster inference."""
        try:
            import torch._dynamo as dynamo
            has_moe = any(layer.mlp_module is not None for layer in self.layers)
            if not has_moe:
                return
            
            print("[INFO] Compiling MoE layers with torch.compile (mode='reduce-overhead')...")
            compile_count = 0
            for layer in self.layers:
                if layer.mlp_module is not None:
                    try:
                        layer._compiled_mlp = torch.compile(
                            layer.mlp_module, 
                            mode="reduce-overhead",
                            fullgraph=False,
                        )
                        compile_count += 1
                    except Exception as e:
                        print(f"[WARN] Failed to compile layer {layer.layer_idx}: {e}")
                        layer._compiled_mlp = None
                else:
                    layer._compiled_mlp = None
            
            if compile_count > 0:
                print(f"[INFO] Successfully prepared {compile_count} MoE layers for compilation")
                print("[INFO] First inference will be slower due to JIT compilation, subsequent runs will be faster")
        except Exception as e:
            print(f"[WARN] torch.compile not available or failed: {e}")
            for layer in self.layers:
                layer._compiled_mlp = None

    
    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        # Construct CONFIG_FILE path first (needed for error messages)
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
        CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
        MODEL_NAME = self.model_name.split("/")[-1]+'.json'
        CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME)
        
        if attn_config is None:
            with open(CONFIG_FILE, "r") as f:
                qwen_config = json.load(f)
        else:
            qwen_config = attn_config
        
        self._update_prefill_chunk_size(qwen_config)
        
        # If we are re-initializing between samples, drop previous kv_cache early to free GPU memory.
        if hasattr(self, "kv_cache") and self.kv_cache is not None:
            try:
                del self.kv_cache
            except Exception:
                self.kv_cache = None
            torch.cuda.empty_cache()

        if self.attention_type == 'PolarANN':
            from cache_hub.polar_cache import polar_cache
            polar_config = qwen_config.get(self.attention_type)
            
            if polar_config is None:
                raise ValueError(f"PolarANN configuration not found in config file. Please check {CONFIG_FILE}")

            polar_cache_kwargs = {
                "valid_start": valid_start,
                "layer_num": self.num_layers,
                "batch_size": self.batch_size,
                "max_length": self.max_new_length + real_input_length,
                "num_key_value_heads": self.num_key_value_heads,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "dtype": self.dtype,
                "layer_mapping": self.layer_mapping,
                "max_new_length": self.max_new_length,
                "sink_size": polar_config.get("sink_size", 4),
                "local_size": polar_config.get("local_size", 512),
                "core": polar_config["core"],
                "nprobe": polar_config["nprobe"],
                "cache_unit_size": polar_config["cache_unit_size"],
                "cache_cluster_num": polar_config["cache_cluster_num"],
                "num_gpus": self.num_gpus,
                "model_size": int(re.search(r'(\d+)[B]', self.model_name).group(1)),
                "enable_offload": polar_config.get("enable_offload", False),
                "dynamic_update_interval": polar_config.get("dynamic_update_interval", 128),
                "final_topk": polar_config.get("final_topk", 100),
            }
            
            if "codebook_path" in polar_config:
                polar_cache_kwargs["codebook_path"] = polar_config["codebook_path"]
            
            self.kv_cache = polar_cache(**polar_cache_kwargs)
            
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")

    
    def move(self):
        torch.cuda.empty_cache()

    
    def word_embedding(self, inputs_id):
        hidden_states = F.embedding(inputs_id, self.embed_tokens)
        return hidden_states

    
    def lm(self, hidden_states):
        logits = F.linear(hidden_states, self.lm_head).float()
        return logits


    def wqkv(self, hidden_states, layer):
        # Handle both with and without bias (Qwen3 doesn't have bias)
        qkv = F.linear(hidden_states, layer.wqkv, layer.bqkv if layer.bqkv is not None else None)
        # Calculate correct split sizes based on actual QKV dimensions
        # Query: num_heads * head_dim, Key/Value: num_key_value_heads * head_dim
        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        query_states, key_states, value_states = qkv.split([query_dim, kv_dim, kv_dim], dim=-1)
        
        if layer.q_norm_weight is not None:
            query_states = self.head_rmsnorm(query_states, layer.q_norm_weight, layer.q_norm_variance_epsilon, self.num_heads)
        if layer.k_norm_weight is not None:
            key_states = self.head_rmsnorm(key_states, layer.k_norm_weight, layer.k_norm_variance_epsilon, self.num_key_value_heads)
        return query_states, key_states, value_states

    
    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        hidden_states = F.linear(hidden_states, layer.wo)
        return hidden_states

    
    def prefill_attention(self, query_states, key_states, value_states):
        if self.attention_type == 'PolarANN':
            attn_out = polar_prefill_attn(query_states, key_states, value_states, causal=True)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out
    

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'PolarANN':
            attn_out = polar_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        return attn_out

    
    def mlp(self, hidden_states, layer):
        if layer.mlp_module is not None:
            if hasattr(layer, '_compiled_mlp') and layer._compiled_mlp is not None:
                result = layer._compiled_mlp(hidden_states)
            else:
                result = layer.mlp_module(hidden_states)
            
            if isinstance(result, tuple):
                hidden_states = result[0]
            else:
                hidden_states = result
        else:
            hidden_states = F.linear(hidden_states, layer.gate_up_proj)
            dim = hidden_states.shape[-1] // 2
            hidden_shape = (hidden_states.shape[:-1] + (dim,))
            out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
            flashinfer.activation.silu_and_mul(hidden_states, out)
            hidden_states = F.linear(out, layer.down_proj)
        return hidden_states 

    
    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx+1)] if str(ldx+1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)
        return hidden_states

    
    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        
        if hidden_states.dtype == torch.float32:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
            hidden_states = hidden_states * weight
            return hidden_states
        else:
            hidden_states = hidden_states.reshape(bsz * seq_len, dim)
            hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
            hidden_states = hidden_states.reshape(bsz, seq_len, dim)
            return hidden_states

    
    def head_rmsnorm(self, states, weight, epsilon, num_heads):
        """
        Apply RMSNorm per attention head (Qwen3 uses q_norm / k_norm on head_dim).
        Args:
            states: [bsz, seq_len, num_heads * head_dim] tensor
            weight: [head_dim]
        """
        if weight is None:
            return states
        bsz, seq_len, _ = states.shape
        reshaped = states.reshape(bsz, seq_len, num_heads, self.head_dim)
        
        if states.dtype == torch.float32:
            variance = reshaped.pow(2).mean(-1, keepdim=True)
            reshaped = reshaped * torch.rsqrt(variance + epsilon)
            reshaped = reshaped * weight
            return reshaped.reshape(bsz, seq_len, num_heads * self.head_dim)
        else:
            reshaped = reshaped.reshape(bsz * seq_len * num_heads, self.head_dim)
            reshaped = flashinfer.rmsnorm(reshaped, weight, epsilon)
            reshaped = reshaped.reshape(bsz, seq_len, num_heads, self.head_dim)
            return reshaped.reshape(bsz, seq_len, num_heads * self.head_dim)


    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, seq_len, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape
        
        if query_states.dtype == torch.float32:
            q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
            k = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            
            positions = position_ids[0]
            
            if hasattr(self, 'cos_cache') and hasattr(self, 'sin_cache'):
                cos = self.cos_cache[positions].to(query_states.dtype)
                sin = self.sin_cache[positions].to(query_states.dtype)
            else:
                cos_sin = self.cos_sin_cache[positions].to(query_states.dtype)
                half_dim = cos_sin.shape[-1] // 2
                cos = cos_sin[..., :half_dim]
                sin = cos_sin[..., half_dim:]
            
            rope_dim = cos.shape[-1]
            
            q_rot = q[..., :rope_dim]
            q_pass = q[..., rope_dim:]
            k_rot = k[..., :rope_dim]
            k_pass = k[..., rope_dim:]
            
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
            
            def rotate_half(x):
                x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)
            
            q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
            
            q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
            
            query_states = q_embed.view(bsz, seq_len, hidden_dim)
            key_states = k_embed.view(bsz, seq_len, kv_dim)
            return query_states, key_states
        else:
            query_states = query_states.contiguous().view(-1, hidden_dim)
            key_states = key_states.contiguous().view(-1, kv_dim)
            
            positions = position_ids.reshape(-1).to(torch.int32)
            
            _ = flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
                positions, query_states, key_states, 
                self.head_dim, self.cos_sin_cache, True
            )
            query_states = query_states.view(bsz, -1, hidden_dim)
            key_states = key_states.view(bsz, -1, kv_dim)
            return query_states, key_states


    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape

        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context+seq_len].unsqueeze(0).repeat(bsz, 1)
        
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, position_ids)

        return query_states, key_states
