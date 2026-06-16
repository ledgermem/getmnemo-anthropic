import { describe, expect, it, vi } from "vitest";
import { withMemoryTool, MEMORY_TOOL_NAME } from "./memory-tool.js";

const CONTAINER_TAG = "user:jane";

function fakeMnemo() {
  return {
    // Core 0.2.0 search() resolves a SearchResponse — primary hits in `results`.
    search: vi.fn().mockResolvedValue({
      results: [
        {
          resultType: "memory",
          memoryId: "m1",
          scopeKey: "user:jane",
          content: "user prefers dark mode",
          metadata: null,
          memoryType: "preference",
          polarity: "positive",
          score: 0.9,
          createdAt: "2026-06-16T00:00:00Z",
          updatedAt: "2026-06-16T00:00:00Z",
        },
      ],
    }),
    // Core 0.2.0 add() resolves an AddResponse `{ scopeKey, scope, items }`.
    add: vi.fn().mockResolvedValue({
      scopeKey: "user:jane",
      scope: { type: "user", id: "jane" },
      items: [{ id: "m2", content: "stored" }],
    }),
    update: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
    list: vi.fn(),
  } as any;
}

describe("withMemoryTool", () => {
  it("short-circuits when the model returns text without tool use", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "Hi there." }],
        }),
      },
    };
    const wrap = withMemoryTool({ client, getmnemo: fakeMnemo(), containerTag: CONTAINER_TAG });
    const out = await wrap.run({
      system: "You are helpful.",
      messages: [{ role: "user", content: "Hello" }],
    });
    expect(out.text).toBe("Hi there.");
    expect(out.memoryToolCalls).toBe(0);
    expect(client.messages.create).toHaveBeenCalledOnce();
  });

  it("runs the memory tool loop and feeds tool_result back", async () => {
    const getmnemo = fakeMnemo();
    const client = {
      messages: {
        create: vi
          .fn()
          .mockResolvedValueOnce({
            id: "r1",
            stop_reason: "tool_use",
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: MEMORY_TOOL_NAME,
                input: { action: "search", query: "ui prefs" },
              },
            ],
          })
          .mockResolvedValueOnce({
            id: "r2",
            stop_reason: "end_turn",
            content: [{ type: "text", text: "You prefer dark mode." }],
          }),
      },
    };
    const wrap = withMemoryTool({ client, getmnemo, containerTag: CONTAINER_TAG });
    const out = await wrap.run({
      messages: [{ role: "user", content: "What do I prefer?" }],
    });
    expect(out.text).toBe("You prefer dark mode.");
    expect(out.memoryToolCalls).toBe(1);
    // Core 0.2.0 surface: object arg with `q`, server-injected `containerTag`,
    // and the clamped limit. The model never supplied the container tag.
    expect(getmnemo.search).toHaveBeenCalledWith({
      q: "ui prefs",
      containerTag: CONTAINER_TAG,
      limit: 5,
    });
    // The tool_result turn must have been fed back. `messages` is one shared
    // array (the run mutates it in place), so `.at(-1)` is the final assistant
    // text turn by inspection time — assert the tool_result turn by position
    // instead: [user, assistant(tool_use), user(tool_result), assistant(text)].
    const finalMessages = out.messages;
    const toolResultMsg = finalMessages[2];
    expect(toolResultMsg.role).toBe("user");
    expect((toolResultMsg.content as any[])[0].type).toBe("tool_result");
  });

  it("opts into prompt caching on the system block by default", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "ok" }],
        }),
      },
    };
    const wrap = withMemoryTool({ client, getmnemo: fakeMnemo(), containerTag: CONTAINER_TAG });
    await wrap.run({
      system: "stable instructions",
      messages: [{ role: "user", content: "hi" }],
    });
    const params = client.messages.create.mock.calls[0]![0] as any;
    expect(params.system[0].cache_control).toEqual({ type: "ephemeral" });
  });

  it("enables extended thinking when configured", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "ok" }],
        }),
      },
    };
    const wrap = withMemoryTool({
      client,
      getmnemo: fakeMnemo(),
      containerTag: CONTAINER_TAG,
      thinkingBudgetTokens: 4000,
    });
    await wrap.run({ messages: [{ role: "user", content: "hi" }] });
    const params = client.messages.create.mock.calls[0]![0] as any;
    expect(params.thinking).toEqual({ type: "enabled", budget_tokens: 4000 });
  });

  it("throws when no containerTag is configured", () => {
    expect(() =>
      withMemoryTool({
        client: { messages: { create: vi.fn() } },
        getmnemo: fakeMnemo(),
        containerTag: "  ",
      }),
    ).toThrow(/containerTag/);
  });

  it("never exposes containerTag/scope in the model-facing tool schema", async () => {
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r1",
          stop_reason: "end_turn",
          content: [{ type: "text", text: "ok" }],
        }),
      },
    };
    const wrap = withMemoryTool({
      client,
      getmnemo: fakeMnemo(),
      containerTag: CONTAINER_TAG,
    });
    await wrap.run({ messages: [{ role: "user", content: "hi" }] });
    const params = client.messages.create.mock.calls[0]![0] as any;
    const props = params.tools[0].input_schema.properties;
    expect(Object.keys(props).sort()).toEqual([
      "action",
      "content",
      "limit",
      "query",
      "tags",
    ]);
    expect(props.containerTag).toBeUndefined();
    expect(props.scope).toBeUndefined();
    expect(params.tools[0].input_schema.additionalProperties).toBe(false);
  });

  it("injects the server containerTag and ignores any tag a model emits", async () => {
    const getmnemo = fakeMnemo();
    const client = {
      messages: {
        create: vi
          .fn()
          .mockResolvedValueOnce({
            id: "r1",
            stop_reason: "tool_use",
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: MEMORY_TOOL_NAME,
                // Hostile/confused model tries to smuggle a foreign tenant tag.
                input: {
                  action: "add",
                  content: "secret",
                  containerTag: "user:attacker",
                  scope: { type: "user", id: "attacker" },
                },
              },
            ],
          })
          .mockResolvedValueOnce({
            id: "r2",
            stop_reason: "end_turn",
            content: [{ type: "text", text: "done" }],
          }),
      },
    };
    const wrap = withMemoryTool({ client, getmnemo, containerTag: CONTAINER_TAG });
    await wrap.run({ messages: [{ role: "user", content: "remember this" }] });
    const addArg = getmnemo.add.mock.calls[0]![0];
    expect(addArg.containerTag).toBe(CONTAINER_TAG);
    expect(addArg.content).toBe("secret");
  });

  it("honors a per-run containerTag override", async () => {
    const getmnemo = fakeMnemo();
    const client = {
      messages: {
        create: vi
          .fn()
          .mockResolvedValueOnce({
            id: "r1",
            stop_reason: "tool_use",
            content: [
              {
                type: "tool_use",
                id: "tu_1",
                name: MEMORY_TOOL_NAME,
                input: { action: "search", query: "x" },
              },
            ],
          })
          .mockResolvedValueOnce({
            id: "r2",
            stop_reason: "end_turn",
            content: [{ type: "text", text: "ok" }],
          }),
      },
    };
    const wrap = withMemoryTool({ client, getmnemo, containerTag: CONTAINER_TAG });
    await wrap.run({
      messages: [{ role: "user", content: "hi" }],
      containerTag: "user:bob",
    });
    expect(getmnemo.search.mock.calls[0]![0].containerTag).toBe("user:bob");
  });

  it("respects maxIterations and stops the loop", async () => {
    const getmnemo = fakeMnemo();
    const client = {
      messages: {
        create: vi.fn().mockResolvedValue({
          id: "r",
          stop_reason: "tool_use",
          content: [
            {
              type: "tool_use",
              id: "tu",
              name: MEMORY_TOOL_NAME,
              input: { action: "search", query: "x" },
            },
          ],
        }),
      },
    };
    const wrap = withMemoryTool({
      client,
      getmnemo,
      containerTag: CONTAINER_TAG,
      maxIterations: 2,
    });
    const out = await wrap.run({
      messages: [{ role: "user", content: "loop" }],
    });
    expect(client.messages.create).toHaveBeenCalledTimes(2);
    expect(out.memoryToolCalls).toBe(2);
  });
});
