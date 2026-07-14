# openpi policy service

EvoWorld pins `third_party/openpi` to commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`. The submodule is never edited.

Start the pinned service from its own checkout with an argv produced by
`build_openpi_server_command` (`build_openpi_serve_command` remains a compatibility
alias); pass the top-level `project_root` so the builder can verify that the checkout
is the exact audited gitlink. The command launches EvoWorld's checked-in wrapper,
passes the expected OpenPi pin, and selects the checkout as the explicit uv project.
The checkpoint directory is explicit, absolute, canonical, and free of symlinks. An
externally managed service lifecycle must expose `GET /healthz` on the configured host
and port before the bridge opens the official `openpi_client` WebSocket client. The
bridge checks it with a bounded two-second timeout for every isolated request and requires the exact
`200 OK` body `OK\n`.

Both the command builder and process bridge reject fake, symlinked, dirty,
uninitialized, or wrong-revision recursive submodule trees. The process bridge
repeats this audit immediately before every child launch, closing the gap between
configuration and execution.

`build_openpi_process_bridge` fixes an absolute Python executable and bridge script,
uses the openpi checkout as `cwd`, passes only an explicit environment, and confines
process logs to the configured artifact root. Its allowed media root is also fixed
on the command line and repeated in each request. Both sides resolve every image,
reject missing files and symlink escapes, and pass only absolute contained paths over
JSON. After health succeeds, the backend lazily loads the two images and uses the
pinned `openpi_client.image_tools.resize_with_pad` helper at `224 x 224` before
calling the official client.

Before loading, the wrapper recursively inventories and hashes the actual checkpoint.
It loads the policy through the exact pinned `get_config` and `create_trained_policy`
signatures used by upstream `serve_policy.py`, then verifies the manifest again before
opening the websocket server. The server metadata preserves the policy's real metadata
and adds `checkpoint_identity` (the resolved absolute checkpoint path) and
`checkpoint_digest` (the combined manifest SHA-256). Runtime configuration must use
those exact values; names such as `pi05-runtime` and placeholder digests are not valid
checkpoint identities. The bridge reads and validates this independent websocket
metadata handshake before inference. The official response also contains the pinned
policy's `policy_timing` object and the websocket server's `server_timing` object. The
bridge validates their exact finite numeric shapes, discards both, and returns only
actions to the core; identity is never accepted from the inference response.

The core package does not import openpi or NumPy. Joint state is exactly seven finite
values and gripper state is exactly one finite value. The official DROID output
transform returns the first eight action dimensions, so public service actions are
pi0.5 `(15, 8)`, pi0 `(10, 8)`, and pi0-FAST `(10, 8)`. The 32-dimensional pi0.5
setting is the model's internal training width, not the DROID service output width.
The core bridge result contains only `actions`; any other bridge-result field is
rejected.

Fine-tuning configuration and its no-shuffle data-loader bridge are documented with
the training implementation; this service document deliberately covers inference
lifecycle only.
## Fine-tuning

EvoWorld pins openpi to `15a9616a00943ada6c20a0f158e3adb39df2ccac` and keeps that
checkout unmodified. The method-aligned overrides are 5,000 steps, eight loader
workers, `shuffle=false`, `chunk-000` storage, and joint-velocity actions. pi0.5
uses `(action_dim=32, horizon=15, batch=16)`; pi0 uses `(horizon=10, batch=8)`
while retaining its upstream action dimension; pi0-FAST uses
`(action_dim=8, horizon=10, batch=8, max_tokens=180)`.

The executable bridge owns config construction in-process, so generated names are
never passed to the upstream registry. A training job records an absolute bridge
path and all of `config-path`, expert, manifest hash, curated dataset root, output
directory, and stable job id. The `train` operation always computes normalization,
reconstructs the same official `TrainConfig` so it loads those statistics, and then
runs training. Normalization can also be run independently:

```text
uv run --project <openpi-root> python <absolute-bridge-path> normalize \
  --openpi-root <absolute-pinned-openpi-root> \
  --config-path <absolute-openpi-toml> --expert pi05 \
  --manifest-path <absolute-manifest-json> --manifest-sha256 <sha256> \
  --dataset-root <absolute-lerobot-dataset-root> \
  --output-dir <absolute-output> --job-id <stable-job-id>
```

The data factory is the pinned official `LeRobotDROIDDataConfig`, whose transforms
consume joint-velocity actions; it replaces the Libero factories in the pi0 and
pi0-FAST base recipes and points at the complete LeRobot dataset root. That root is
not the chunk directory: it contains `meta/info.json`, task/episode JSONL metadata,
and `data/chunk-000/episode_*.parquet`. This is an input prerequisite: the bridge
does not convert, subset, or synthesize LeRobot data. The T2 curation stage must
produce a train-only root plus `meta/evoworld_manifest.json`, with exactly one
episode per manifest sample and no retained evaluation episodes. Every sample must
be diagnosed `PF`, use split `train`, and target the selected expert.

EvoWorld first checks the pinned v2.1 metadata, official DROID image/state features,
raw 8D joint-velocity actions, episode horizon, one-to-one sample IDs, contained
artifact paths, and content hashes. Inside the isolated pinned openpi environment,
the bridge then opens every episode with Arrow, checks real schemas and row counts,
constructs the pinned `LeRobotDataset`, and reads an actual horizon-aligned sample
before normalization or training starts.

Before official imports, the child process validates `--openpi-root` contains the
pinned scripts and package tree and that Git HEAD is exactly `15a9616`. It then
prepends only that root and its `src` directory to module lookup. EvoWorld is loaded
directly from the absolute bridge's repository root; it is not installed into the
openpi environment, so EvoWorld's NumPy 2 dependency cannot alter openpi's pinned
dependency solution. The bridge keeps
the base recipe's weight loader, optimizer, schedule, freeze filters, preprocessing,
loss, EMA, and checkpoint implementation.

The pinned `scripts/train.py` main function forces `shuffle=True`, so EvoWorld never
calls that main function. It imports the pinned `init_train_state` and `train_step`
helpers at process execution time, locally composes the official JAX sharding,
loader, and checkpoint APIs, passes `shuffle=False`, and saves the final checkpoint
at step `4999` after exactly 5,000 updates. The upstream checkout remains unmodified.
