# EvoWorld Upstream Integrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pin three official upstream repositories and expose isolated, testable policy, world-model, and evaluator adapters without modifying upstream code.

**Architecture:** Offline command builders and schema validators live in the core package. Heavy imports run only inside backend-specific bridges or official service clients, connected by request-correlated JSONL, WebSocket, or ZeroMQ transports.

**Tech Stack:** Git submodules, Python 3.11, subprocess JSONL, openpi WebSocket/msgpack, GR00T ZeroMQ/msgpack, pytest.

## Global Constraints

- Pin Ctrl-World to `99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d`.
- Pin openpi to `15a9616a00943ada6c20a0f158e3adb39df2ccac`.
- Pin Isaac-GR00T to `ead52833afbbf4243f8cd5e7664f48a94de03b19` (`n1.6-release`).
- Initialize and audit nested submodules recursively.
- Never edit a file inside a submodule from the parent repository.
- Default tests stay offline and do not import CUDA-heavy packages.
- Use argv arrays; never concatenate task text or paths into shell commands.
- New filenames must not contain `paper`.
- Keep every commit local; do not push.

---

### Task I1: Add and pin three recursive Git submodules

**Files:**
- Create: `.gitmodules`
- Create: `third_party/ctrl_world` gitlink
- Create: `third_party/openpi` gitlink
- Create: `third_party/isaac_groot` gitlink

**Interfaces:**
- Produces: immutable upstream source locations used by every real adapter and trainer.

- [ ] **Step 1: Record the expected clean baseline**

Run: `test ! -e .gitmodules && test ! -e third_party`

Expected: exit zero before submodules are added.

- [ ] **Step 2: Add official repositories and detach at approved commits**

```bash
git submodule add https://github.com/Robert-gyj/Ctrl-World.git third_party/ctrl_world
git submodule add https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git submodule add https://github.com/NVIDIA/Isaac-GR00T.git third_party/isaac_groot
git -C third_party/ctrl_world checkout --detach 99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d
git -C third_party/openpi checkout --detach 15a9616a00943ada6c20a0f158e3adb39df2ccac
git -C third_party/isaac_groot checkout --detach ead52833afbbf4243f8cd5e7664f48a94de03b19
git submodule update --init --recursive
```

Expected: every checkout succeeds; openpi and GR00T nested submodules initialize.

- [ ] **Step 3: Verify exact pins and recursive cleanliness**

```bash
test "$(git -C third_party/ctrl_world rev-parse HEAD)" = 99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d
test "$(git -C third_party/openpi rev-parse HEAD)" = 15a9616a00943ada6c20a0f158e3adb39df2ccac
test "$(git -C third_party/isaac_groot rev-parse HEAD)" = ead52833afbbf4243f8cd5e7664f48a94de03b19
test -z "$(git submodule status --recursive | sed -n '/^[-+U]/p')"
```

Expected: all assertions exit zero.

- [ ] **Step 4: Commit gitlinks and module metadata**

```bash
git add .gitmodules third_party/ctrl_world third_party/openpi third_party/isaac_groot
git commit -m "chore: pin EvoWorld upstream submodules"
```

### Task I2: Verify submodule revisions and cleanliness

**Files:**
- Create: `src/evoworld/adapters/revisions.py`
- Create: `scripts/verify_submodules.py`
- Test: `tests/contract/test_submodule_revisions.py`
- Modify: `configs/reproduction.toml`

**Interfaces:**
- Produces: `EXPECTED_REVISIONS`, `inspect_submodules(repo_root)`, and CLI exit code.

- [ ] **Step 1: Write failing revision tests**

```python
def test_revision_audit_rejects_wrong_head(tmp_path: Path) -> None:
    runner = FakeGitRunner(heads={"third_party/openpi": "wrong"}, dirty=set())
    with pytest.raises(SubmoduleRevisionError, match="openpi"):
        inspect_submodules(tmp_path, runner=runner)


def test_revision_audit_rejects_dirty_nested_module(tmp_path: Path) -> None:
    runner = FakeGitRunner(heads=EXPECTED_REVISIONS, dirty={"third_party/openpi/third_party/libero"})
    with pytest.raises(SubmoduleDirtyError, match="libero"):
        inspect_submodules(tmp_path, runner=runner)
```

- [ ] **Step 2: Verify tests fail before the auditor exists**

Run: `uv run pytest tests/contract/test_submodule_revisions.py -v`

Expected: FAIL importing revision functions.

- [ ] **Step 3: Implement exact and recursive audit**

```python
EXPECTED_REVISIONS = {
    "third_party/ctrl_world": "99fb20683fd79dfa6d0c6feb9d49c6c55eecd50d",
    "third_party/openpi": "15a9616a00943ada6c20a0f158e3adb39df2ccac",
    "third_party/isaac_groot": "ead52833afbbf4243f8cd5e7664f48a94de03b19",
}
```

Run `git -C <path> rev-parse HEAD`, `git -C <path> status --porcelain`, and
`git submodule status --recursive`. Return revisions for run manifests; raise
structured errors for missing, wrong, dirty, uninitialized, or conflicted modules.

- [ ] **Step 4: Verify tests and the local checkout**

Run: `uv run pytest tests/contract/test_submodule_revisions.py -v && uv run python scripts/verify_submodules.py`

Expected: contract tests and the recursive local audit exit zero.

- [ ] **Step 5: Commit the revision gate**

```bash
git add src/evoworld/adapters/revisions.py scripts/verify_submodules.py tests/contract/test_submodule_revisions.py configs/reproduction.toml
git commit -m "feat: verify pinned backend revisions"
```

### Task I3: JSONL process protocol

**Files:**
- Create: `src/evoworld/adapters/process.py`
- Create: `integrations/common/jsonl_bridge.py`
- Test: `tests/contract/test_process_protocol.py`

**Interfaces:**
- Produces: `BridgeRequest`, `BridgeResponse`, `ProcessBridge.request`, and `serve(handler)`.

- [ ] **Step 1: Write failing success and failure tests**

```python
def test_process_bridge_correlates_response_id(tmp_path: Path) -> None:
    bridge = ProcessBridge([sys.executable, fixture_bridge("success")], cwd=tmp_path)
    response = bridge.request(BridgeRequest("req-1", "act", tmp_path, {}), timeout=2.0)
    assert response.request_id == "req-1"
    assert response.status == "ok"


@pytest.mark.parametrize("mode", ["malformed", "wrong-id", "nonzero", "timeout"])
def test_process_bridge_reports_structured_failure(tmp_path: Path, mode: str) -> None:
    with pytest.raises(BridgeError):
        ProcessBridge([sys.executable, fixture_bridge(mode)], cwd=tmp_path).request(request(tmp_path), 0.1)
```

- [ ] **Step 2: Verify tests fail before transport exists**

Run: `uv run pytest tests/contract/test_process_protocol.py -v`

Expected: FAIL importing `ProcessBridge`.

- [ ] **Step 3: Implement strict JSONL transport**

```python
@dataclass(frozen=True, slots=True)
class BridgeRequest:
    request_id: str
    operation: str
    artifact_dir: Path
    payload: Mapping[str, JsonValue]
    protocol_version: int = 1


class ProcessBridge:
    def request(self, request: BridgeRequest, timeout: float) -> BridgeResponse:
        completed = subprocess.run(self.argv, input=encode(request), text=True,
                                   capture_output=True, cwd=self.cwd, timeout=timeout,
                                   check=False, env=self.env)
        return decode_checked(completed, expected_id=request.request_id)
```

Require absolute artifact paths and JSON values. Treat stdout contamination, wrong
IDs, timeouts, and nonzero exits as distinct errors. Keep protocol output on stdout,
logs on stderr, and full logs under the artifact directory.

- [ ] **Step 4: Verify all protocol cases**

Run: `uv run pytest tests/contract/test_process_protocol.py -v && uv run ruff check src/evoworld/adapters/process.py integrations/common/jsonl_bridge.py tests/contract/test_process_protocol.py && uv run mypy src integrations`

Expected: success and every structured failure case pass.

- [ ] **Step 5: Commit the process boundary**

```bash
git add src/evoworld/adapters/process.py integrations/common/jsonl_bridge.py tests/contract/test_process_protocol.py
git commit -m "feat: add the backend process protocol"
```

### Task I4: openpi policy service adapter

**Files:**
- Create: `src/evoworld/adapters/openpi.py`
- Create: `integrations/openpi/policy_bridge.py`
- Test: `tests/contract/test_openpi_adapter.py`
- Create: `docs/backends/openpi.md`

**Interfaces:**
- Produces: `OpenPiPolicyAdapter`, `build_openpi_serve_command`, and observation/action mappers.

- [ ] **Step 1: Write failing command and mapping tests**

```python
def test_openpi_serve_command_is_argument_safe() -> None:
    command = build_openpi_serve_command(Path("/repo"), "pi05_droid", Path("/ckpt"), 8000)
    assert command == ["uv", "run", "scripts/serve_policy.py", "policy:checkpoint",
                       "--policy.config=pi05_droid", "--policy.dir=/ckpt", "--port=8000"]


def test_openpi_actions_map_to_domain_chunk() -> None:
    chunk = map_openpi_response("pi05", {"actions": np.zeros((15, 32))})
    assert (len(chunk.actions), len(chunk.actions[0])) == (15, 32)
```

- [ ] **Step 2: Verify adapter tests fail**

Run: `uv run pytest tests/contract/test_openpi_adapter.py -v`

Expected: FAIL importing openpi adapter.

- [ ] **Step 3: Implement lazy service integration**

Validate `/healthz`, map DROID observation keys and prompt, call the official client
inside `policy_bridge.py`, and map NumPy actions to `ActionChunk`. Validate pi05
`(15,32)`, pi0 horizon 10, and pi0-FAST `(10,8)`. Do not import openpi in core tests.

- [ ] **Step 4: Verify offline contracts**

Run: `uv run pytest tests/contract/test_openpi_adapter.py -v && uv run ruff check src/evoworld/adapters/openpi.py integrations/openpi/policy_bridge.py tests/contract/test_openpi_adapter.py && uv run mypy src integrations`

Expected: command, health, schema, and action mapping tests pass.

- [ ] **Step 5: Commit openpi integration**

```bash
git add src/evoworld/adapters/openpi.py integrations/openpi/policy_bridge.py tests/contract/test_openpi_adapter.py docs/backends/openpi.md
git commit -m "feat: adapt openpi policy services"
```

### Task I5: GR00T-N1.6 policy service adapter

**Files:**
- Create: `src/evoworld/adapters/groot.py`
- Create: `integrations/groot/policy_bridge.py`
- Test: `tests/contract/test_groot_adapter.py`
- Create: `docs/backends/groot.md`

**Interfaces:**
- Produces: `GrootPolicyAdapter`, `build_groot_server_command`, and nested modality mappers.

- [ ] **Step 1: Write failing server and modality tests**

```python
def test_groot_server_command_uses_n16_checkpoint() -> None:
    command = build_groot_server_command(Path("/repo"), Path("/ckpt"), "NEW_EMBODIMENT", 5555)
    assert "gr00t/eval/run_gr00t_server.py" in command
    assert command[command.index("--model-path") + 1] == "/ckpt"


def test_groot_response_requires_btd_actions() -> None:
    with pytest.raises(AdapterSchemaError, match="B, T, D"):
        map_groot_response("groot", {"actions": np.zeros((16, 7))})
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/contract/test_groot_adapter.py -v`

Expected: FAIL importing GR00T adapter.

- [ ] **Step 3: Implement the lazy ZeroMQ bridge**

Map inputs to nested `video`, `state`, and `language`; use official
`PolicyClient.ping`, `get_modality_config`, and `get_action`; validate float32
`(B,T,D)` and select batch zero. Keep pyzmq/msgpack imports inside the bridge.

- [ ] **Step 4: Verify GR00T contracts**

Run: `uv run pytest tests/contract/test_groot_adapter.py -v && uv run ruff check src/evoworld/adapters/groot.py integrations/groot/policy_bridge.py tests/contract/test_groot_adapter.py && uv run mypy src integrations`

Expected: command, ping, modality, and action-shape tests pass offline.

- [ ] **Step 5: Commit GR00T integration**

```bash
git add src/evoworld/adapters/groot.py integrations/groot/policy_bridge.py tests/contract/test_groot_adapter.py docs/backends/groot.md
git commit -m "feat: adapt GR00T-N1.6 policy services"
```

### Task I6: Ctrl-World rollout bridge

**Files:**
- Create: `src/evoworld/adapters/ctrl_world.py`
- Create: `integrations/ctrl_world/rollout_bridge.py`
- Test: `tests/contract/test_ctrl_world_adapter.py`
- Create: `docs/backends/ctrl_world.md`

**Interfaces:**
- Produces: `CtrlWorldAdapter.imagine`, command builders, and 7-D action/artifact validators.

- [ ] **Step 1: Write failing action and artifact tests**

```python
def test_ctrl_world_requires_seven_dimensional_actions() -> None:
    with pytest.raises(AdapterSchemaError, match="7-D"):
        validate_ctrl_actions(ActionChunk("pi05", ((0.0,) * 8,), (0.0,), "joint"))


def test_rollout_bridge_returns_absolute_frames(tmp_path: Path) -> None:
    rollout = map_ctrl_response(fixture_ctrl_response(tmp_path, frame_count=16))
    assert all(path.is_absolute() for path in rollout.frame_paths)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/contract/test_ctrl_world_adapter.py -v`

Expected: FAIL importing Ctrl-World adapter.

- [ ] **Step 3: Implement the narrow upstream bridge**

Validate DROID layout and 7-D Cartesian conditioning. Inside the backend environment,
set the required upstream module-global args, construct its official agent, call the
world-model forward method, and emit frame/video artifacts through JSONL. Require
all model, dataset, checkpoint, and output paths; never copy `/cephfs` defaults.

- [ ] **Step 4: Verify Ctrl-World contracts**

Run: `uv run pytest tests/contract/test_ctrl_world_adapter.py -v && uv run ruff check src/evoworld/adapters/ctrl_world.py integrations/ctrl_world/rollout_bridge.py tests/contract/test_ctrl_world_adapter.py && uv run mypy src integrations`

Expected: action, path, artifact, and structured-error tests pass offline.

- [ ] **Step 5: Commit Ctrl-World integration**

```bash
git add src/evoworld/adapters/ctrl_world.py integrations/ctrl_world/rollout_bridge.py tests/contract/test_ctrl_world_adapter.py docs/backends/ctrl_world.md
git commit -m "feat: adapt Ctrl-World imagined rollouts"
```

### Task I7: Qwen3-VL evaluator bridge

**Files:**
- Create: `src/evoworld/adapters/qwen_vl.py`
- Create: `integrations/qwen_vl/evaluator_bridge.py`
- Test: `tests/contract/test_qwen_evaluator_adapter.py`
- Create: `docs/backends/qwen_vl.md`

**Interfaces:**
- Produces: `QwenVlEvaluator.evaluate`, strict prompt/response parser, and bridge command.

- [ ] **Step 1: Write failing strict-response tests**

```python
def test_qwen_response_maps_to_diagnosis() -> None:
    diagnosis = parse_evaluator_output('{"score":0.2,"label":"PF","rationale":"grasp missed"}')
    assert diagnosis.score == 0.2
    assert diagnosis.label is DiagnosisLabel.PF


@pytest.mark.parametrize("text", ["score: 0.5", '{"score":"NaN","label":"NONE","rationale":"x"}',
                                  '{"score":0.5,"label":"OTHER","rationale":"x"}'])
def test_qwen_response_rejects_non_contract_output(text: str) -> None:
    with pytest.raises(EvaluatorOutputError):
        parse_evaluator_output(text)
```

- [ ] **Step 2: Verify tests fail**

Run: `uv run pytest tests/contract/test_qwen_evaluator_adapter.py -v`

Expected: FAIL importing Qwen evaluator adapter.

- [ ] **Step 3: Implement prompt and inference bridge**

Prompt includes task instruction, ordered frames/video, grounded scene context, and
the exact JSON schema. Bridge lazily loads `Qwen/Qwen3-VL-4B-Instruct` or an adapter
checkpoint. Parser accepts one JSON object with score `[0,1]`, an approved label,
and nonblank rationale.

- [ ] **Step 4: Run the integrations quality gate**

Run: `uv run pytest tests/contract -v && uv run ruff check src/evoworld/adapters integrations tests/contract && uv run mypy src integrations && uv run python scripts/verify_submodules.py`

Expected: all offline contracts and the local revision audit pass.

- [ ] **Step 5: Commit evaluator integration**

```bash
git add src/evoworld/adapters/qwen_vl.py integrations/qwen_vl/evaluator_bridge.py tests/contract/test_qwen_evaluator_adapter.py docs/backends/qwen_vl.md
git commit -m "feat: adapt the Qwen3-VL evaluator"
```
