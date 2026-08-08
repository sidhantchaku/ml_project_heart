# AgentCore Runtime deployment (CardioRisk-AI)

**Status as of this writing: prepared and validated locally. Not deployed to
AWS.** This directory contains everything needed to deploy, but actual
deployment requires AWS credentials and a decision to spend, which only you
can authorize and run.

## Tooling actually verified in this environment (not from memory)

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12.10 | |
| boto3 / botocore | 1.43.65 | |
| `bedrock-agentcore` (SDK) | 1.21.0 | Provides `BedrockAgentCoreApp` / `@app.entrypoint`, used in `agent_entrypoint.py` |
| `bedrock-agentcore-starter-toolkit` (CLI, `agentcore` command) | 0.3.11 | **Its own `--help` output states this CLI is deprecated** and recommends the newer `@aws/agentcore` npm package for new work |
| `@aws/agentcore` (npm, official successor CLI) | 0.26.0 published (confirmed via `npm view`) | Could not retrieve `--help` output in this sandbox (`npx @aws/agentcore --help` timed out with no output -- likely a registry/network restriction here, not a real tooling failure) |

**Practical conclusion:** the commands below use the still-installed and
functional `agentcore` (starter-toolkit) CLI, since it works in this
environment and is still AWS-published. Before you actually deploy, re-check
whether `@aws/agentcore` is preferred/required in your environment --
the deprecation notice is real and current, and the ecosystem may have moved
further by the time you deploy.

One additional real finding: `agentcore deploy --help` crashed in this
Windows sandbox with `UnicodeEncodeError: 'charmap' codec can't encode
character '\U0001f680'` -- a console-encoding bug in the CLI's own emoji
output on cp1252 terminals, not a problem with this project's code. Running
with a UTF-8-capable terminal (or `PYTHONIOENCODING=utf-8`) should avoid it.

## Deployment method: direct code deployment

Chosen over container deployment because:
- `heart_model.pkl` (975 bytes) and `scaler.pkl` (1263 bytes) are tiny and
  packagable directly -- no need for S3 or a container image just to carry
  two small pickle files.
- Every dependency (FastAPI, boto3, LangGraph, scikit-learn, joblib,
  bedrock-agentcore) installs from standard wheels; nothing here needs a
  custom OS-level binary.
- `agentcore configure` explicitly supports `--deployment-type
  direct_code_deploy` with a `--runtime` (Python version) and `--s3` bucket
  for staging the code package.

Container deployment (`--deployment-type container`) is documented in
`agentcore configure --help` as the fallback if direct deployment turns out
to be unsuitable (e.g. a future native dependency that direct deploy can't
package) -- not needed here.

## Known artifact/version caveat (carried over from Phase 2)

`heart_model.pkl`/`scaler.pkl` were pickled with scikit-learn **1.7.2**; this
environment has scikit-learn **1.9.0** installed, producing an
`InconsistentVersionWarning` on load (see `tools/risk_prediction.py`).
Predictions have been verified to still match a manual reference computation
(see `tests/test_prediction.py`), but pin `scikit-learn==1.7.2` in
`requirements-agentcore.txt` (and the main `requirements.txt`) if you want to
eliminate the warning before deploying, or retrain the model against 1.9.0.
Neither was done in this phase, per "do not modify stable functionality
unnecessarily."

## What gets deployed

- `infrastructure/agentcore/agent_entrypoint.py` -- the AgentCore entry point
  (`@app.entrypoint`), a thin wrapper.
- `infrastructure/agentcore/runtime_adapter.py` -- validates the payload and
  calls the existing `agent.graph.invoke_cardio_graph()`.
- The existing `agent/`, `api/schemas.py`, `services/`, `tools/`, `config/`
  packages, plus `heart_model.pkl` / `scaler.pkl` -- all unchanged, reused
  as-is.
- `infrastructure/agentcore/requirements-agentcore.txt` as the deployment's
  dependency list (kept separate from the Vercel app's `requirements.txt`).

Nothing here duplicates model loading, preprocessing, retrieval, generation,
safety, or LangGraph node logic -- see `runtime_adapter.py`'s docstring.

## Commands

All commands below are what you would run **from the repository root**,
**after** you have valid AWS credentials configured (`aws configure` or
equivalent) and have reviewed `deployment/iam_policy_example.json` and
`deployment/trust_policy_example.json`. None of these were run against real
AWS in this session -- see the main Phase 6 report for why.

### 1. Authentication and identity verification
```bash
aws sts get-caller-identity
aws configure get region
```
Confirms which AWS account/identity and region you're about to act in
*before* creating anything.

### 2. Local runtime testing (no AWS calls unless Bedrock is enabled)
```bash
python scripts/invoke_agentcore_local.py
```
or, to run the actual AgentCore local dev server:
```bash
agentcore configure --create --entrypoint infrastructure/agentcore/agent_entrypoint.py \
  --name cardiorisk-ai --deployment-type direct_code_deploy --non-interactive
agentcore dev
```

### 3. Agent configuration
```bash
agentcore configure --create \
  --entrypoint infrastructure/agentcore/agent_entrypoint.py \
  --name cardiorisk-ai \
  --deployment-type direct_code_deploy \
  --runtime PYTHON_3_12 \
  --requirements-file infrastructure/agentcore/requirements-agentcore.txt \
  --execution-role <ARN of the least-privilege role in deployment/iam_policy_example.json> \
  --region <your-region> \
  --non-interactive
```
Confirm `PYTHON_3_12` is an accepted `--runtime` value at deploy time --
this environment only showed `PYTHON_3_10`/`PYTHON_3_11` as *examples* in
`--help`, not an exhaustive list; re-check against whatever this project's
actual runtime is at deploy time.

### 4. Deployment
```bash
agentcore deploy
```
**Do not run this until every item in the "Before deploying" checklist in
the main Phase 6 report is actually true.**

### 5. Runtime status inspection
```bash
agentcore status --agent cardiorisk-ai
```

### 6. Invocation
```bash
agentcore invoke '{"patient_input": {"age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3}, "user_message": "Explain what this risk estimate means."}'
```

### 7. Log inspection
```bash
agentcore obs --agent cardiorisk-ai
```
(or the equivalent CloudWatch Logs console/CLI query for the runtime's log
group -- exact log group name is assigned at deploy time.)

### 8. Cleanup (destructive -- never run automatically)
```bash
agentcore destroy --agent cardiorisk-ai
```
This removes the AgentCore endpoint and runtime. Run it yourself,
deliberately, when you're done -- it is never invoked by any script in this
repository.

## Scope decisions (per Phase 6 instructions)

- **No AgentCore Gateway**: the existing prediction tool is already an
  internal LangGraph tool, not an MCP tool, and there is no current
  requirement for external tool discovery/exposure via Gateway. Documented
  here as a **future enhancement** if a use case for exposing tools to other
  agents/consumers via MCP ever emerges -- not implemented.
- **No AgentCore Memory**: each request is stateless (see `agent/state.py`);
  no persistent conversational memory is required or added.
- **No multi-agent architecture**: one LangGraph workflow, one entry point.
- **No Docker**: direct code deployment covers this project's needs.
