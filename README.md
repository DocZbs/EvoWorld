# EvoWorld

EvoWorld is a world-model-centric framework for routing modular robot policies,
simulating candidate actions, executing a selected policy, and using Qwen3-VL
diagnosis to select targeted updates.

The public repository includes deterministic orchestration, pinned Ctrl-World,
OpenPI, and Isaac GR00T submodules, backend bridges, data contracts, and plan-first
fine-tuning. Real experiments require private data, checkpoints, calibration, and
robot infrastructure.

## Quick start

```bash
git clone --recursive https://github.com/DocZbs/EvoWorld.git
cd EvoWorld
git submodule update --init --recursive
uv sync --python 3.11 --extra dev
uv run python scripts/verify_submodules.py
uv run evoworld validate --config configs/dry_run.toml
uv run evoworld run --config configs/dry_run.toml --rounds 1
uv run evoworld resume --run-dir artifacts/runs/<run-id>
```

The method follows **Task Generation**, **Contextual Scene Anchoring**,
**Strategic Routing and Imaginative Simulation**, and **VLM Diagnosis and
Recursive Evolution**.

## Fine-tuning

Fine-tuning is explicit and plan-first:

```bash
cp configs/finetune/pi05.example.toml /absolute/path/pi05.toml
uv run evoworld finetune plan --config /absolute/path/pi05.toml --target pi05
uv run evoworld finetune execute --job <job-from-plan-json>
```

Supported targets are `pi05`, `pi0`, `pi0_fast`, `groot_n16`, `ctrl_world`, and
`qwen_evaluator`. See [fine-tuning](docs/fine_tuning.md) for data and GPU
requirements.

## Documentation

- [Architecture](docs/architecture.md)
- [Data contracts](docs/data.md)
- [Real runtime](docs/backends/real_composition.md)
- [OpenPI](docs/backends/openpi.md), [GR00T](docs/backends/groot.md),
  [Ctrl-World](docs/backends/ctrl_world.md), and [Qwen3-VL](docs/backends/qwen_vl.md)
- [Project article](site/static/pdfs/evoworld.pdf)

## Limitations

The repository does not ship private trajectories, annotations, trained weights,
projector calibration, or robot control code. Deterministic mode runs locally;
real training and execution require the documented Linux/CUDA environments and
operator-provided safety controls.

## Citation

```bibtex
@inproceedings{zhang2026evoworld,
  title={EvoWorld: A World-Model-Centric Framework for Continuous Self-Evolution of Modular Embodied Skills},
  author={Boshi Zhang and Sen Cui and {BaoHuaYin} and Youyi Kou and Junyu Wu and Zuo Pu and Tao Xue and Zhikang Chen and Shanshan Wei and Min Zhang and Miao Liu and Changshui Zhang and {Zhang Tao}},
  booktitle={European Conference on Computer Vision},
  year={2026}
}
```
