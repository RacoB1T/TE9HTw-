from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from packaging.version import Version

import torch
import transformers
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)


# 论文称 LoRA 微调 attention modules。
# 对 Llama-2、Llama-3 和 Mistral，注意力线性层通常对应这四个模块。
ATTENTION_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


def _get_torch_dtype(training_args: Any) -> Optional[torch.dtype]:
    """Infer model loading dtype from Hugging Face TrainingArguments."""
    if getattr(training_args, "bf16", False):
        return torch.bfloat16

    if getattr(training_args, "fp16", False):
        return torch.float16

    return None


def _configure_position_encoding(config: Any, model_args: Any) -> Any:
    """
    Apply optional RoPE/context-window settings.

    Important:
    ModelArguments.max_position_embeddings defaults to 10 in the
    released code. We must not overwrite an existing 4K/8K/80K
    context window with that default value.
    """
    requested_max_position = getattr(
        model_args,
        "max_position_embeddings",
        None,
    )

    original_max_position = getattr(
        config,
        "max_position_embeddings",
        None,
    )

    # Only expand the context window. Do not accidentally shrink it to 10.
    if (
        requested_max_position is not None
        and requested_max_position > 0
        and (
            original_max_position is None
            or requested_max_position > original_max_position
        )
    ):
        logger.info(
            "Set max_position_embeddings: %s -> %s",
            original_max_position,
            requested_max_position,
        )
        config.max_position_embeddings = int(requested_max_position)

    rope_theta = getattr(model_args, "rope_theta", None)
    if rope_theta is not None:
        logger.info("Set rope_theta to %s", rope_theta)
        config.rope_theta = float(rope_theta)

    rope_type = getattr(model_args, "rope_type", None)
    rope_factor = getattr(model_args, "factor", None)

    if rope_type is not None:
        if rope_factor is None or float(rope_factor) <= 0:
            raise ValueError(
                "A positive --factor is required when --rope_type is set."
            )

        if Version(transformers.__version__) >= Version("4.44.0"):
            config.rope_scaling = {
                "rope_type": str(rope_type),
                "factor": float(rope_factor),
            }
        else:
            config.rope_scaling = {
                "type": str(rope_type),
                "factor": float(rope_factor),
            }

        logger.info("Set rope_scaling to %s", config.rope_scaling)

    return config


def _prepare_tokenizer(
    model_name_or_path: str,
    training_args: Any,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=getattr(training_args, "cache_dir", None),
        model_max_length=getattr(
            training_args,
            "max_seq_length",
            8192,
        ),
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Llama 系列通常没有独立 pad token。
    # 使用 EOS 避免无意义地扩大词表。
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens(
                {"pad_token": "<|pad|>"}
            )

    return tokenizer


def _load_base_model(
    model_name_or_path: str,
    config: Any,
    training_args: Any,
) -> PreTrainedModel:
    load_kwargs = {
        "config": config,
        "cache_dir": getattr(training_args, "cache_dir", None),
        "trust_remote_code": True,
        # 使用 DeepSpeed 时不要设置 device_map="auto"。
        "low_cpu_mem_usage": True,
    }

    torch_dtype = _get_torch_dtype(training_args)
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )

    return model


def _apply_lora(
    model: PreTrainedModel,
    model_args: Any,
    training_args: Any,
) -> PreTrainedModel:
    peft_model_path = getattr(model_args, "peft_model_path", None)

    if peft_model_path:
        logger.info(
            "Load trainable PEFT adapter from %s",
            peft_model_path,
        )

        model = PeftModelForCausalLM.from_pretrained(
            model,
            peft_model_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=int(getattr(model_args, "lora_r", 32)),
            lora_alpha=int(
                getattr(model_args, "lora_alpha", 16)
            ),
            target_modules=ATTENTION_LORA_TARGETS,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        logger.info("LoRA configuration: %s", lora_config)
        model = get_peft_model(model, lora_config)

    # 仓库 TrainingArguments 默认是 "embed, norm"。
    # 即除 LoRA 参数外，Embedding 和归一化层也参加训练。
    trainable_names = [
        name.strip()
        for name in getattr(
            training_args,
            "trainable_params",
            "embed, norm",
        ).split(",")
        if name.strip()
    ]

    for parameter_name, parameter in model.named_parameters():
        if any(key in parameter_name for key in trainable_names):
            parameter.requires_grad_(True)

    return model


def _prepare_gradient_checkpointing(
    model: PreTrainedModel,
    training_args: Any,
) -> None:
    if not getattr(training_args, "gradient_checkpointing", False):
        return

    model.config.use_cache = False

    # PEFT + gradient checkpointing 时，需要让输入 Embedding 输出保留梯度。
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(
            module: torch.nn.Module,
            inputs: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(
            make_inputs_require_grad
        )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def _print_trainable_parameters(model: PreTrainedModel) -> None:
    total_parameters = 0
    trainable_parameters = 0

    # data_ptr 去重可以避免 tied embedding 被重复统计。
    visited = set()

    for parameter in model.parameters():
        pointer = parameter.data_ptr()
        if pointer in visited:
            continue

        visited.add(pointer)
        total_parameters += parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter.numel()

    ratio = (
        100.0 * trainable_parameters / total_parameters
        if total_parameters > 0
        else 0.0
    )

    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable_parameters:,}",
        f"{total_parameters:,}",
        ratio,
    )

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def create_and_prepare_model(
    model_name_or_path: str,
    training_args: Any,
    model_args: Any,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure the model used by LOGO.

    Responsibilities inferred from the released scripts:
      1. Load and optionally modify model configuration;
      2. Load tokenizer and causal language model;
      3. Apply/load LoRA when low_rank_training=True;
      4. Unfreeze embedding/norm parameters;
      5. Prepare gradient checkpointing;
      6. Return model and tokenizer.
    """
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=getattr(training_args, "cache_dir", None),
        trust_remote_code=True,
    )

    config = _configure_position_encoding(
        config=config,
        model_args=model_args,
    )

    tokenizer = _prepare_tokenizer(
        model_name_or_path=model_name_or_path,
        training_args=training_args,
    )

    model = _load_base_model(
        model_name_or_path=model_name_or_path,
        config=config,
        training_args=training_args,
    )

    # 只有新增了 pad token 时才需要扩展词表。
    embedding_size = model.get_input_embeddings().num_embeddings
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if getattr(training_args, "low_rank_training", True):
        model = _apply_lora(
            model=model,
            model_args=model_args,
            training_args=training_args,
        )

    _prepare_gradient_checkpointing(
        model=model,
        training_args=training_args,
    )

    _print_trainable_parameters(model)

    return model, tokenizer