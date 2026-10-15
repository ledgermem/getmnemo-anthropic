# getmnemo-anthropic

Mnemo helper for the Anthropic SDK. Wraps `messages.create` with a single
`memory` tool (search + add), prompt caching on the system block, and
optional extended thinking — the recommended setup for any long-lived Claude
assistant.

## Install

```bash
npm install getmnemo-anthropic @anthropic-ai/sdk
```

`getmnemo` (the core SDK, `^0.6`) is pulled in automatically as a dependency —
no separate pin needed. You only need `@anthropic-ai/sdk` because you construct
the Anthropic client yourself and pass it in.

Get your Mnemo API key from
[app.mnemohq.com/settings/api-keys](https://app.mnemohq.com/settings/api-keys).

## Quickstart (30 seconds)

```ts
import Anthropic from "@anthropic-ai/sdk";
import { withMemoryTool } from "getmnemo-anthropic";

const agent = withMemoryTool({
  client: new Anthropic(),
  apiKey: process.env.GETMNEMO_API_KEY!,
  containerTag: "user:jane",
  model: "claude-sonnet-4-6",
  maxTokens: 8192, // must exceed thinkingBudgetTokens
  thinkingBudgetTokens: 4000,
});

const result = await agent.run({
  system: "You are a personal assistant. Use memory before answering.",
  messages: [{ role: "user", content: "What did I tell you about my schedule?" }],
  metadata: { userId: "user-42" },
});

console.log(result.text);
console.log("memory tool calls:", result.memoryToolCalls);
```

Already have a configured `Mnemo` instance? Pass it via `getmnemo` instead of
`apiKey` — both forms work:

```ts
import { Mnemo } from "getmnemo";

const agent = withMemoryTool({
  client: new Anthropic(),
  getmnemo: new Mnemo({ apiKey: process.env.GETMNEMO_API_KEY! }),
  containerTag: "user:jane",
});
```

## Containers

`containerTag` is required. It scopes every memory read/write and is the
tenant-isolation boundary — supply it server-side (e.g. `"user:<your-user-id>"`).
It is deliberately absent from the model-facing tool schema, so a model (or a
prompt-injection attacker) can never pick another tenant's container. Override
it per `run()` call via `run({ containerTag })` when one wrapper serves many
users.

Searching a container that nothing has written to yet returns no matches: the
wrapper handles the API's 404 ("Container not found") for brand-new containers
and hands the model `{ results: [] }` instead of failing the tool call.

## What the wrapper does

- Exposes one tool `memory` with `action: "search" | "add"` — the model
  decides when to recall and when to remember.
- Runs the tool-use loop for you (default cap: 6 iterations).
- Adds `cache_control: { type: "ephemeral" }` to the system prompt so
  long stable instructions hit the prompt cache.
- Optionally enables extended thinking with a configurable token budget.
- Returns the final transcript so you can persist it elsewhere.

## API

`withMemoryTool(options)` returns a wrapper with a single `run(input)` method.

### `withMemoryTool(options)`

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `client` | `Anthropic` | — | An `@anthropic-ai/sdk` client (anything with `messages.create`). |
| `getmnemo` | `Mnemo` | — | A `Mnemo` instance from `getmnemo`. Optional — omit it and pass `apiKey` instead. |
| `apiKey` | `string` | `GETMNEMO_API_KEY` | Mnemo API key the wrapper uses to construct its own client when `getmnemo` is not passed. |
| `containerTag` | `string` | — | **Required.** Tenant-isolation boundary for all memory reads/writes (e.g. `"user:jane"`). |
| `model` | `string` | `"claude-sonnet-4-6"` | Default model; override per call. |
| `maxTokens` | `number` | `1024` | Default `max_tokens`; override per call. |
| `searchLimit` | `number` | `5` | Default top-k for memory search (clamped to 1–50). |
| `cacheSystem` | `boolean` | `true` | Wrap the system prompt with `cache_control: { type: "ephemeral" }`. |
| `thinkingBudgetTokens` | `number` | — | Enable extended thinking with this token budget. `maxTokens` must be greater (checked at construction and per run). |
| `metadata` | `Record<string, unknown>` | `{}` | Static metadata merged into every persisted memory. |
| `maxIterations` | `number` | `6` | Hard cap on tool-use loop iterations. |

### `run(input)`

| Field | Type | Description |
| --- | --- | --- |
| `messages` | `ChatMessage[]` | Conversation so far (required). |
| `system` | `string` | Optional system prompt. |
| `model` | `string` | Per-call model override. |
| `maxTokens` | `number` | Per-call `max_tokens` override. |
| `metadata` | `Record<string, unknown>` | Per-call metadata merged into persisted memories. |
| `containerTag` | `string` | Per-call override of the configured container tag (still server-supplied, never model input). |

Returns:

| Field | Type | Description |
| --- | --- | --- |
| `text` | `string` | Final assistant text, concatenated across blocks. |
| `responses` | `MessagesCreateResult[]` | Every raw API response from the tool-use loop. |
| `memoryToolCalls` | `number` | How many times the memory tool ran. |
| `messages` | `ChatMessage[]` | Final messages array after the loop — store this as your transcript. |

The package also exports `MEMORY_TOOL_NAME` (the literal tool name, `"memory"`)
and the `WithMemoryTool*` / `WithMemoryRun*` types.

## Links

- Docs: [mnemohq.com](https://mnemohq.com)
- API keys: [app.mnemohq.com/settings/api-keys](https://app.mnemohq.com/settings/api-keys)

## License

MIT
