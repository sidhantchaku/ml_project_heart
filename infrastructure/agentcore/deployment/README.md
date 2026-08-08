# IAM policy examples (CardioRisk-AI AgentCore Runtime)

**These are examples for review, not production-certified policies.** Replace
every `<PLACEHOLDER>` before use, have someone review them, and test in a
non-production account first. Neither file has been applied to any real AWS
account as part of this project.

## `trust_policy_example.json`
The trust policy for the AgentCore Runtime **execution role** -- who is
allowed to assume this role. Scoped to the `bedrock-agentcore.amazonaws.com`
service principal, further restricted with `aws:SourceAccount` and
`aws:SourceArn` conditions so only this specific account's AgentCore runtimes
can assume it.

## `iam_policy_example.json`
The **permissions** policy attached to that execution role -- what the
running agent process is allowed to do once AgentCore has assumed the role.

| Statement | Scope | Notes |
|---|---|---|
| `InvokeSelectedBedrockFoundationModelOnly` | One specific foundation model ARN | `bedrock:InvokeModel(WithResponseStream)` |
| `RetrieveFromSpecificKnowledgeBaseOnly` | One specific Knowledge Base ARN | `bedrock-agent-runtime:Retrieve` |
| `WriteLogsToThisAgentsLogGroupOnly` | This agent's own CloudWatch log group prefix | Standard logging actions |
| `EmitAgentCoreObservabilityTraces` | `Resource: "*"` -- **cannot be scoped further** | `xray:PutTraceSegments`/`PutTelemetryRecords` do not support resource-level ARN scoping in IAM; this is an AWS API limitation, not a choice made here. Every other statement in this policy is resource-scoped. |

No statement uses `"Action": "*"`. Only the X-Ray statement above uses
`"Resource": "*"`, and only because the action itself doesn't support finer
scoping.

## `vercel_invoker_policy_example.json`

A **third, separate** identity: the credentials Vercel's server-side
FastAPI code (`services/agentcore_client.py`) uses to *call* the deployed
runtime -- distinct from both the runtime's own execution role above and
whoever deploys it. This is the credential set described in the main
README's "Vercel credential strategy" section: a dedicated IAM user (or role,
if your Vercel setup supports federation) that can do exactly one thing --
`bedrock-agentcore:InvokeAgentRuntime` against this one runtime ARN -- and
nothing else. It cannot create/delete/modify AWS resources, cannot read S3,
cannot touch IAM, and is never exposed to the browser (it lives only in
Vercel's encrypted server-side environment variables).

## Not included here: the *deployer's* permissions

The policies above are for the runtime's own execution role. Separately, the
IAM identity that actually **runs** `agentcore configure`/`agentcore deploy`
(you, or a CI role) needs its own, broader set of control-plane permissions
-- e.g. `bedrock-agentcore:CreateAgentRuntime`, `iam:PassRole` (to hand the
execution role to the service), ECR/S3 permissions for staging the deployment
package, etc. Those permissions target resources that don't exist yet at
policy-authoring time (the runtime itself), so they can't be fully
resource-scoped in advance the way the execution role's policy above can.
Use AWS's own current AgentCore IAM documentation for that policy when you're
ready to deploy, and prefer scoping by resource-name-prefix (e.g.
`cardiorisk-ai*`) wherever the API allows it.

## S3 (not used in this deployment)

This project packages `heart_model.pkl`/`scaler.pkl` directly into the
AgentCore direct-code-deploy package (see `../README.md`) rather than loading
them from S3 at runtime, so no S3 read permissions are included in the
execution role policy above. If a future change moves the model artifacts to
S3, add a narrowly-scoped `s3:GetObject` statement limited to the exact
object ARNs (not the bucket), and ensure the bucket itself is never made
public.
