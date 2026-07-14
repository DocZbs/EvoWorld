# Qwen3-VL evaluator backend

The public inference baseline is exactly `Qwen/Qwen3-VL-4B-Instruct`. EvoWorld
passes ordered, validated local trajectory frames and one strict evaluation prompt
through an isolated JSONL process. The model must return exactly one JSON object:

```json
{"score": 0.9, "label": "NONE", "rationale": "task completed"}
```

`score` is a finite JSON number in `[0, 1]`; `label` is one of `TSM`, `PF`,
`WMF`, or `NONE`; and `rationale` is nonblank, trimmed, and at most 1,000
characters. Markdown fences, surrounding prose, duplicate keys, non-RFC numeric
constants, unknown fields, and implicit coercions are rejected. Model-supplied
metadata is rejected unless the caller explicitly allowlists every metadata field.

## Backend environment

Install Qwen inference dependencies in the separate backend environment, not in
the lightweight EvoWorld core environment:

```text
transformers>=4.57,<5
accelerate>=1,<2    # required by automatic device placement
peft>=0.17,<1       # required only for a local LoRA adapter checkpoint
torch              # version selected for the target CUDA/CPU runtime
Pillow
```

Qwen's official repository states that Qwen3-VL requires Transformers 4.57 or
newer and demonstrates `AutoModelForImageTextToText`, `AutoProcessor`,
`apply_chat_template`, `generate`, input-prefix trimming, and `batch_decode`:
https://github.com/QwenLM/Qwen3-VL#using-qwen3-vl-to-chat

The Transformers multimodal chat-template guide documents ordered image/text
content blocks and tokenized processor output:
https://huggingface.co/docs/transformers/chat_templating_multimodal

PEFT documents loading a saved local adapter directory over an initialized base
model with `PeftModel.from_pretrained(..., is_trainable=False)`:
https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftModel.from_pretrained

The bridge bounds `max_new_tokens`, requires an explicit `device` and `dtype`, and
sets `do_sample=False`. Automatic device placement uses the declared Accelerate
range; explicit `cpu` or `cuda` placement avoids `device_map`. It imports
Transformers, Accelerate, and PEFT only after both the parent process and child
process have independently decoded and validated every frame and checked all
checkpoint paths. The parent adapter sends paths, identifiers, and text
through JSONL; it never embeds media bytes in JSON. Every diagnosis records a
stable request ID and binds the task, scene, trajectory, and selected checkpoint.

## Base model and fine-tuned checkpoint boundary

With no adapter path, the checkpoint provenance is
`Qwen/Qwen3-VL-4B-Instruct`. A local adapter must be an existing directory under
the configured checkpoint root and contain `adapter_config.json`; the bridge loads
the public base first and then attaches that adapter through PEFT.

Training task T6 exports a method-aligned `ms-swift==4.3.1` LoRA baseline and can
produce the local adapter consumed here. That recipe is a configurable public
baseline for real and manually curated synthetic trajectory examples. The exact
evaluator dataset, checkpoint, and hyperparameters used in the original experiment
are not public, so this repository does not claim that the T6 recipe reproduces
those unpublished details.
