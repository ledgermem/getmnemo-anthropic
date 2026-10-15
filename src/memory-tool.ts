import { Mnemo } from "getmnemo";

/**
 * Structural subset of `Anthropic` we depend on — keeps the peer dep loose.
 *
 * `create` is typed loosely on purpose: the real SDK's `messages.create` is an
 * overloaded function with strict param types, which no narrower signature
 * here can match. `params: never` accepts any such function; the single call
 * site below narrows it to the shape we actually send/read.
 */
export interface AnthropicLike {
  messages: {
    create: (params: never) => PromiseLike<unknown>;
  };
}

type CreateFn = (
  params: Record<string, unknown>,
) => PromiseLike<MessagesCreateResult>;

interface MessagesCreateResult {
  id: string;
  content: Array<MessageBlock>;
  stop_reason: string | null;
  usage?: Record<string, number>;
  [key: string]: unknown;
}

type MessageBlock =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string }
  | {
      type: "tool_use";
      id: string;
      name: string;
      input: Record<string, unknown>;
    };

interface ChatMessage {
  role: "user" | "assistant";
  content: string | Array<Record<string, unknown>>;
}

export interface WithMemoryToolOptions {
  client: AnthropicLike;
  /**
   * A `Mnemo` instance from `getmnemo`. Optional — when omitted, the wrapper
   * constructs its own client from `apiKey` (or `GETMNEMO_API_KEY`).
   */
  getmnemo?: Mnemo;
  /**
   * Mnemo API key, used only when no `getmnemo` client is passed. Falls back
   * to `process.env.GETMNEMO_API_KEY`.
   */
  apiKey?: string;
  /**
   * REQUIRED. The container tag (e.g. `"user:jane"`) that scopes every memory
   * read/write for this wrapper. This is the TENANT-ISOLATION BOUNDARY and is
   * supplied by YOU (the developer/server) — it is deliberately NOT part of the
   * model-facing tool schema, so a model (or a prompt-injection attacker) can
   * never read or write another tenant's memories by emitting a different tag.
   * Can be overridden per `run()` via `WithMemoryRunInput.containerTag`.
   */
  containerTag: string;
  /** Defaults to claude-sonnet-4-6. Override per call too. */
  model?: string;
  /**
   * Defaults to 1024. With `thinkingBudgetTokens` set it must be strictly
   * greater than the budget (Anthropic rejects the request otherwise).
   */
  maxTokens?: number;
  /** Default top-k for memory search. */
  searchLimit?: number;
  /** Wrap the system prompt with `cache_control: { type: "ephemeral" }`. */
  cacheSystem?: boolean;
  /** Enable extended thinking with this token budget. */
  thinkingBudgetTokens?: number;
  /** Extra static metadata merged into every persisted memory. */
  metadata?: Record<string, unknown>;
  /** Hard cap on tool-use loop iterations. */
  maxIterations?: number;
}

export interface WithMemoryRunInput {
  system?: string;
  messages: ChatMessage[];
  /** Per-call overrides. */
  model?: string;
  maxTokens?: number;
  metadata?: Record<string, unknown>;
  /**
   * Per-call override of the configured `containerTag`. Still developer/server
   * supplied (a `run()` argument) — NOT something the model controls.
   */
  containerTag?: string;
}

export interface WithMemoryRunResult {
  /** Final assistant text concatenated across blocks. */
  text: string;
  /** All raw API responses produced during the tool-use loop. */
  responses: MessagesCreateResult[];
  /** Number of memory tool invocations. */
  memoryToolCalls: number;
  /** Final messages array after tool loop (useful for storing transcript). */
  messages: ChatMessage[];
}

export interface WithMemoryToolWrapper {
  run(input: WithMemoryRunInput): Promise<WithMemoryRunResult>;
}

export const MEMORY_TOOL_NAME = "memory";

const DEFAULT_MAX_TOKENS = 1024;

function assertThinkingFits(
  maxTokens: number,
  thinkingBudgetTokens: number | undefined,
): void {
  if (thinkingBudgetTokens && maxTokens <= thinkingBudgetTokens) {
    throw new Error(
      `withMemoryTool: maxTokens (${maxTokens}) must be greater than ` +
        `thinkingBudgetTokens (${thinkingBudgetTokens}) — Anthropic rejects ` +
        "budget_tokens >= max_tokens. Raise maxTokens (e.g. 8192).",
    );
  }
}

/**
 * Wrap an `Anthropic` client so a single `memory` tool (with `search`/`add`
 * actions) is exposed to the model and the tool-use loop is handled for you.
 *
 * Also opts into prompt caching of the system prompt and (optionally) extended
 * thinking — both are recommended for any long-lived assistant.
 */
export function withMemoryTool(
  options: WithMemoryToolOptions,
): WithMemoryToolWrapper {
  if (!options.containerTag || !options.containerTag.trim()) {
    // Fail fast: an empty container tag would either 400 at the API or — worse —
    // silently fall back to a wrong scope. The isolation boundary must be set.
    throw new Error(
      "withMemoryTool: `containerTag` is required and identifies the tenant " +
        "scope for all memory reads/writes. Supply it server-side.",
    );
  }
  assertThinkingFits(options.maxTokens ?? DEFAULT_MAX_TOKENS, options.thinkingBudgetTokens);
  const mnemo = resolveMnemo(options);
  const createMessage = options.client.messages.create.bind(
    options.client.messages,
  ) as unknown as CreateFn;
  const defaults = {
    containerTag: options.containerTag,
    model: options.model ?? "claude-sonnet-4-6",
    maxTokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
    searchLimit: options.searchLimit ?? 5,
    cacheSystem: options.cacheSystem ?? true,
    thinkingBudgetTokens: options.thinkingBudgetTokens,
    metadata: options.metadata ?? {},
    maxIterations: options.maxIterations ?? 6,
  };

  const tool = {
    name: MEMORY_TOOL_NAME,
    description:
      "Long-term memory store. Use action='search' to recall facts about the user before answering, and action='add' to save a new fact worth remembering across conversations.",
    input_schema: {
      type: "object" as const,
      // additionalProperties:false lets Anthropic enforce strict JSON schema
      // for tool inputs — without it the model can pass arbitrary keys
      // (including keys that look like trusted metadata) and we silently
      // accept them.
      //
      // SECURITY: `containerTag`/`scope` are DELIBERATELY ABSENT from this
      // schema. The container tag is the tenant-isolation boundary; it is
      // injected by the wrapper from server config (`defaults.containerTag`)
      // and is never a field the model can fill. Do NOT add it here.
      additionalProperties: false,
      properties: {
        action: {
          type: "string",
          enum: ["search", "add"],
          description: "Whether to read from or write to memory.",
        },
        query: {
          type: "string",
          description: "Required when action='search'. Natural-language query.",
        },
        content: {
          type: "string",
          description: "Required when action='add'. The fact to remember.",
        },
        limit: {
          type: "integer",
          minimum: 1,
          maximum: 50,
          description: "Optional max results for search.",
        },
        tags: {
          type: "object",
          description: "Optional metadata merged into the stored memory.",
          additionalProperties: true,
        },
      },
      required: ["action"],
    },
  };

  return {
    async run(input: WithMemoryRunInput): Promise<WithMemoryRunResult> {
      const messages: ChatMessage[] = [...input.messages];
      const responses: MessagesCreateResult[] = [];
      let toolCalls = 0;

      // Resolve the tenant scope once per run. Per-call override is still a
      // server-supplied `run()` argument (like `metadata`), never model input.
      const runContainerTag =
        input.containerTag && input.containerTag.trim()
          ? input.containerTag
          : defaults.containerTag;

      const maxTokens = input.maxTokens ?? defaults.maxTokens;
      assertThinkingFits(maxTokens, defaults.thinkingBudgetTokens);
      const system = buildSystem(input.system, defaults.cacheSystem);
      const baseParams: Record<string, unknown> = {
        model: input.model ?? defaults.model,
        max_tokens: maxTokens,
        tools: [tool],
        ...(system ? { system } : {}),
      };
      if (defaults.thinkingBudgetTokens) {
        baseParams.thinking = {
          type: "enabled",
          budget_tokens: defaults.thinkingBudgetTokens,
        };
      }

      for (let i = 0; i < defaults.maxIterations; i++) {
        const resp = await createMessage({
          ...baseParams,
          messages,
        });
        responses.push(resp);

        // Anthropic requires every tool_use block in an assistant turn to be
        // answered with a matching tool_result in the next user turn. Collect
        // ALL tool_use blocks (not just memory ones) so the message list stays
        // valid; non-memory tools get a structured error result.
        const allToolUses = resp.content.filter(
          (b): b is Extract<MessageBlock, { type: "tool_use" }> =>
            b.type === "tool_use",
        );

        messages.push({ role: "assistant", content: resp.content });

        if (resp.stop_reason !== "tool_use" || allToolUses.length === 0) {
          return finalize(messages, responses, toolCalls);
        }

        const toolResults: Array<Record<string, unknown>> = [];
        for (const use of allToolUses) {
          if (use.name !== MEMORY_TOOL_NAME) {
            toolResults.push({
              type: "tool_result",
              tool_use_id: use.id,
              is_error: true,
              content: JSON.stringify({
                error: `Unknown tool: ${use.name}. Only '${MEMORY_TOOL_NAME}' is available.`,
              }),
            });
            continue;
          }
          toolCalls += 1;
          const result = await runMemoryTool(
            use.input,
            mnemo,
            // Server-controlled container tag — the tenant boundary. Injected
            // here; never sourced from `use.input` (model output).
            runContainerTag,
            // Trusted server metadata (input.metadata, defaults.metadata)
            // wins over any tool input fields a model could fabricate.
            { ...defaults.metadata, ...(input.metadata ?? {}) },
            defaults.searchLimit,
          );
          toolResults.push({
            type: "tool_result",
            tool_use_id: use.id,
            content: JSON.stringify(result),
          });
        }
        messages.push({ role: "user", content: toolResults });
      }

      return finalize(messages, responses, toolCalls);
    },
  };
}

function finalize(
  messages: ChatMessage[],
  responses: MessagesCreateResult[],
  memoryToolCalls: number,
): WithMemoryRunResult {
  const last = responses[responses.length - 1];
  const text = last
    ? last.content
        .filter((b): b is Extract<MessageBlock, { type: "text" }> => b.type === "text")
        .map((b) => b.text)
        .join("\n")
        .trim()
    : "";
  return { text, responses, memoryToolCalls, messages };
}

function resolveMnemo(options: WithMemoryToolOptions): Mnemo {
  if (options.getmnemo) return options.getmnemo;
  const apiKey = options.apiKey ?? process.env.GETMNEMO_API_KEY;
  if (!apiKey) {
    throw new Error(
      "withMemoryTool: pass a `getmnemo` client, or an `apiKey` (or set " +
        "GETMNEMO_API_KEY) so the wrapper can construct one.",
    );
  }
  return new Mnemo({ apiKey });
}

function buildSystem(
  system: string | undefined,
  cache: boolean,
): Array<Record<string, unknown>> | undefined {
  if (!system) return undefined;
  if (!cache) return [{ type: "text", text: system }];
  return [
    { type: "text", text: system, cache_control: { type: "ephemeral" } },
  ];
}

async function runMemoryTool(
  raw: Record<string, unknown>,
  client: Mnemo,
  containerTag: string,
  baseMetadata: Record<string, unknown>,
  defaultLimit: number,
): Promise<unknown> {
  const action = String(raw.action ?? "");
  if (action === "search") {
    const query = String(raw.query ?? "").trim();
    if (!query) return { error: "query is required for search" };
    // Clamp the limit into [1, 50] regardless of what the model asked for —
    // a hostile/buggy tool input could otherwise pass a huge value (or 0)
    // that bypasses the JSON-schema bounds and torches the context window.
    const requested = Number(raw.limit ?? defaultLimit);
    const limit = Number.isFinite(requested)
      ? Math.min(50, Math.max(1, Math.floor(requested)))
      : defaultLimit;
    // Core 0.6.0 surface: `{ q, containerTag, limit }`. The container tag is
    // injected from server config, never from `raw` (model output).
    let response;
    try {
      response = await client.search({ q: query, containerTag, limit });
    } catch (err: unknown) {
      // A brand-new container has no memories yet — the API answers a search
      // against a not-yet-written container with 404 "Container not found".
      // For the memory tool that means "nothing remembered yet", not a fatal
      // error → return empty results instead of crashing the tool loop.
      if (isContainerNotFound(err)) {
        return { results: [] };
      }
      throw err;
    }
    // confirmed against prod 2026-06-16. Core `search()` → SearchResponse, whose
    // primary hits live in `results` (NOT `hits`). Each is a SearchHit:
    // { resultType, memoryId, scopeKey, content, metadata, memoryType, polarity,
    //   score, createdAt, updatedAt }. Surface the most useful fields back to
    // Claude — content + score drive its recall decision — and keep memoryId so
    // a follow-up turn can reference a specific memory.
    const hits = response.results ?? [];
    const results = hits.map((h) => ({
      memoryId: h.memoryId,
      content: h.content,
      score: h.score,
      memoryType: h.memoryType,
      polarity: h.polarity,
    }));
    return { results };
  }
  if (action === "add") {
    const content = String(raw.content ?? "").trim();
    if (!content) return { error: "content is required for add" };
    const tags =
      typeof raw.tags === "object" && raw.tags
        ? (raw.tags as Record<string, unknown>)
        : {};
    // Model-supplied `tags` are merged FIRST so trusted server-controlled
    // baseMetadata (userId, workspaceId, etc.) cannot be overwritten by an
    // LLM via prompt injection — e.g. a model that emits
    // tags={"userId":"victim"} would otherwise reattribute the memory.
    // Core 0.6.0 surface: `{ content, containerTag, metadata }`.
    const response = await client.add({
      content,
      containerTag,
      metadata: { ...tags, ...baseMetadata },
    });
    // confirmed against prod 2026-06-16. Core `add()` → AddResponse:
    // `{ scopeKey, scope, items: Memory[] }`. Hand the stored item id(s) and
    // content back to Claude so it can confirm what was saved.
    const items = (response.items ?? []).map((m) => ({
      id: m.id,
      content: m.content,
    }));
    return { saved: items.length, items };
  }
  return { error: `unknown action: ${action}` };
}

/**
 * True only for the API's 404 "Container not found" (a brand-new container
 * nothing has been written to). Other 404s are rethrown.
 */
function isContainerNotFound(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  if ((err as { status?: unknown }).status !== 404) return false;
  const body = (err as { body?: unknown }).body;
  const code =
    body && typeof body === "object" && "code" in body
      ? String((body as { code?: unknown }).code)
      : "";
  return /container/i.test(err.message) || /container/i.test(code);
}
