# getmnemo-anthropic

Mnemo helper for the Anthropic SDK. Wraps `messages.create` with a single
`memory` tool (search + add), prompt caching on the system block, and
optional extended thinking — the recommended setup for any long-lived Claude
assistant.

## Install

```bash
npm install getmnemo-anthropic @anthropic-ai/sdk
```

`getmnemo` (the core SDK) is pulled in automatically as a dependency. You only
need `@anthropic-ai/sdk` because you construct the Anthropic client yourself and
pass it in.

Get your Mnemo API key and workspace ID from
[app.mnemohq.com/settings/api-keys](https://app.mnemohq.com/settings/api-keys).

## Quickstart (30 seconds)

```ts
import Anthropic from "@anthropic-ai/sdk";
import { Mnemo } from "getmnemo";
import { withMemoryTool } from "getmnemo-anthropic";

const claude = new Anthropic();
const memory = new Mnemo({
  apiKey: process.env.GETMNEMO_API_KEY!,
  workspaceId: process.env.GETMNEMO_WORKSPACE_ID!,
});

const agent = withMemoryTool({
  client: claude,
  getmnemo: memory,
  model: "claude-sonnet-4-6",
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
| `getmnemo` | `Mnemo` | — | A `Mnemo` instance from `getmnemo`. |
| `model` | `string` | `"claude-sonnet-4-6"` | Default model; override per call. |
| `maxTokens` | `number` | `1024` | Default `max_tokens`; override per call. |
| `searchLimit` | `number` | `5` | Default top-k for memory search (clamped to 1–50). |
| `cacheSystem` | `boolean` | `true` | Wrap the system prompt with `cache_control: { type: "ephemeral" }`. |
| `thinkingBudgetTokens` | `number` | — | Enable extended thinking with this token budget. |
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
