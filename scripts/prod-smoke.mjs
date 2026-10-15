#!/usr/bin/env node
/**
 * CI prod smoke gate for `getmnemo-anthropic` (the Anthropic SDK helper).
 *
 * A REAL round-trip against PRODUCTION with NO LLM in the loop. `withMemoryTool`
 * is driven by a FAKE `AnthropicLike` client that returns scripted `tool_use`
 * blocks (search on a fresh container, add, search, final text). The wrapper's
 * real tool-dispatch loop executes them against the live Mnemo API, and we
 * assert on the `tool_result` blocks it feeds back. This proves the whole
 * dispatch → core `getmnemo` add/search → prod chain without spending any
 * model-provider credits.
 *
 * (maxTokens vs thinkingBudgetTokens validation is unit-tested, not here.)
 *
 * Steps:
 *   (1) search on a never-written container → tool_result { results: [] }
 *       (404 "Container not found" tolerance).
 *   (2) add a unique nonce fact → tool_result { saved >= 1, items[].id }.
 *   (3) search round-trips the nonce back in tool_result.results[].content.
 *   (4) cross-container isolation: a sibling containerTag does not see it.
 *
 * Cleanup (delete with containerTag) is best-effort and NEVER fatal.
 *
 * Required env:
 *   MNEMO_API_KEY        scoped test key (needs delete scope for cleanup)
 *   MNEMO_WORKSPACE_ID   throwaway test workspace id
 *   MNEMO_TEST_CONTAINER base containerTag, e.g. "ci-smoke"
 */

import { Mnemo } from 'getmnemo'
import { withMemoryTool, MEMORY_TOOL_NAME } from '../dist/index.js'

const SEARCH_ATTEMPTS = 5
const SEARCH_RETRY_MS = 2_000

function fail(msg) {
  console.error(`\n[smoke] FAIL: ${msg}`)
  process.exit(1)
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

let seq = 0
const toolUse = (input) => ({
  id: `msg_smoke_${++seq}`,
  stop_reason: 'tool_use',
  content: [{ type: 'tool_use', id: `toolu_smoke_${seq}`, name: MEMORY_TOOL_NAME, input }],
})
const finalText = (text) => ({
  id: `msg_smoke_${++seq}`,
  stop_reason: 'end_turn',
  content: [{ type: 'text', text }],
})

/**
 * Fake AnthropicLike: replays `script` one response per messages.create call
 * and records the params it was sent. Throws if the loop asks for more.
 */
function fakeClient(script) {
  const queue = [...script]
  const calls = []
  return {
    calls,
    messages: {
      create: async (params) => {
        calls.push(params)
        const next = queue.shift()
        if (!next) throw new Error('fake client: script exhausted')
        return next
      },
    },
  }
}

/** Parsed tool_result payloads in order, from the run's final messages. */
function toolResults(run) {
  const out = []
  for (const m of run.messages ?? []) {
    if (m.role !== 'user' || !Array.isArray(m.content)) continue
    for (const b of m.content) {
      if (b?.type !== 'tool_result') continue
      if (b.is_error) fail(`tool_result is_error: ${b.content}`)
      out.push(JSON.parse(b.content))
    }
  }
  return out
}

const hasNonce = (res, nonce) =>
  (res?.results ?? []).some((r) => String(r?.content ?? '').includes(nonce))

async function main() {
  const apiKey = process.env.MNEMO_API_KEY
  const workspaceId = process.env.MNEMO_WORKSPACE_ID
  const base = process.env.MNEMO_TEST_CONTAINER

  if (!apiKey) fail('MNEMO_API_KEY is not set')
  if (!workspaceId) fail('MNEMO_WORKSPACE_ID is not set')
  if (!base) fail('MNEMO_TEST_CONTAINER is not set')

  const nonce = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  const fruit = `dragonfruit-${nonce}`
  const colon = base.indexOf(':')
  const cidBase = colon >= 0 ? base.slice(colon + 1) : base
  const containerTag = `user:${cidBase}-${nonce}`
  const otherTag = `user:${cidBase}-${nonce}-other`

  const getmnemo = new Mnemo({ apiKey, workspaceId })
  const wrap = (client) => withMemoryTool({ client, getmnemo, containerTag })

  console.log('[smoke] run nonce:    ', nonce)
  console.log('[smoke] container tag:', containerTag)

  const createdIds = []

  try {
    // ---- (1)+(2): fresh-container search, then add ---------------------
    const c1 = fakeClient([
      toolUse({ action: 'search', query: 'favorite fruit' }),
      toolUse({ action: 'add', content: `My favorite fruit is ${fruit}.` }),
      finalText('Saved.'),
    ])
    const storeRun = await wrap(c1).run({
      messages: [{ role: 'user', content: `Remember my favorite fruit is ${fruit}.` }],
    })
    if (storeRun.memoryToolCalls !== 2) {
      fail(`(1/2) expected 2 memory tool calls, got ${storeRun.memoryToolCalls}`)
    }
    const tools = c1.calls[0]?.tools ?? []
    if (tools.length !== 1 || tools[0]?.name !== MEMORY_TOOL_NAME) {
      fail(`tool definition not sent as '${MEMORY_TOOL_NAME}': ${JSON.stringify(tools)}`)
    }
    if (storeRun.text !== 'Saved.') fail(`final text not surfaced: ${JSON.stringify(storeRun.text)}`)

    const [emptyRes, addRes] = toolResults(storeRun)
    if (!Array.isArray(emptyRes?.results) || emptyRes.results.length !== 0) {
      fail(`(1) expected { results: [] } on fresh container, got ${JSON.stringify(emptyRes)}`)
    }
    console.log('[smoke] OK (1): fresh-container search → { results: [] }')

    for (const item of addRes?.items ?? []) if (item?.id) createdIds.push(item.id)
    if (!(addRes?.saved >= 1) || createdIds.length === 0) {
      fail(`(2) add tool_result has no saved items: ${JSON.stringify(addRes)}`)
    }
    console.log(`[smoke] OK (2): add saved ${createdIds.length} item(s)`)

    // ---- (3): search round-trips the nonce (retry for indexing) ---------
    let lastRes
    let found = false
    for (let i = 0; i < SEARCH_ATTEMPTS && !found; i++) {
      if (i > 0) await sleep(SEARCH_RETRY_MS)
      const run = await wrap(
        fakeClient([toolUse({ action: 'search', query: "What's my favorite fruit?", limit: 10 }), finalText('ok')]),
      ).run({ messages: [{ role: 'user', content: "What's my favorite fruit?" }] })
      ;[lastRes] = toolResults(run)
      found = hasNonce(lastRes, fruit)
    }
    if (!found) fail(`(3) "${fruit}" not in search tool_result: ${JSON.stringify(lastRes)}`)
    console.log('[smoke] OK (3): search tool_result round-tripped the written content')

    // ---- (4): cross-container isolation -------------------------------
    const isoRun = await wrap(
      fakeClient([toolUse({ action: 'search', query: "What's my favorite fruit?", limit: 10 }), finalText('ok')]),
    ).run({ containerTag: otherTag, messages: [{ role: 'user', content: 'fruit?' }] })
    const [isoRes] = toolResults(isoRun)
    if (hasNonce(isoRes, fruit)) fail('(4) sibling container saw the nonce — tenant isolation broken')
    console.log('[smoke] OK (4): sibling container did not see the write')
  } finally {
    for (const id of createdIds) {
      try {
        await getmnemo.delete(id, { containerTag })
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        console.warn(`[smoke] WARN: cleanup delete failed for memory ${id}: ${msg}`)
      }
    }
  }

  console.log('\n[smoke] PASS: tool dispatch add + search round-trip green (no LLM).')
  process.exit(0)
}

main().catch((err) => {
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err)
  fail(`unexpected error during smoke run:\n${msg}`)
})
