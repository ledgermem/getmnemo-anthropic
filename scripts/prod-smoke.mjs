#!/usr/bin/env node
/**
 * CI prod smoke gate for `getmnemo-anthropic` (the Anthropic SDK helper).
 *
 * Unlike the core SDK's smoke (which exercises add/search directly), this
 * drives a REAL LLM round-trip: it wires this package's `withMemoryTool` to a
 * live Anthropic client + a live Mnemo client and verifies that Claude actually
 * (a) FIRES the memory tool to STORE a fact, then (b) RECALLS it in a later turn.
 * If the tool never fires, or the fact can't be recalled, publish is blocked.
 *
 * The publish workflow gates `publish` on `needs: smoke`, so a red run here
 * blocks the release.
 *
 * Exit codes:
 *   0  the tool fired on the store turn AND the fact was recalled.
 *   1  missing env, the memory tool never fired, the recall turn lost the fact,
 *      or any unexpected error.
 *
 * Required env:
 *   MNEMO_API_KEY        scoped test key (needs delete scope for cleanup)
 *   MNEMO_WORKSPACE_ID   throwaway test workspace id
 *   MNEMO_TEST_CONTAINER base containerTag, e.g. "ci-smoke"
 *   ANTHROPIC_API_KEY    Anthropic API key for the live model call
 */

import { Mnemo } from 'getmnemo'
import Anthropic from '@anthropic-ai/sdk'
import { withMemoryTool } from '../dist/index.js'

// Exact, real model id — do NOT invent a variant. Haiku keeps the smoke cheap.
const MODEL = 'claude-haiku-4-5'
const MAX_TOKENS = 512

function fail(msg) {
  console.error(`\n[smoke] FAIL: ${msg}`)
  process.exit(1)
}

/** Pull every tool_result content string out of the run's message transcript. */
function toolResultText(runResult) {
  const out = []
  for (const m of runResult.messages ?? []) {
    if (m.role !== 'user' || !Array.isArray(m.content)) continue
    for (const block of m.content) {
      if (block && block.type === 'tool_result' && typeof block.content === 'string') {
        out.push(block.content)
      }
    }
  }
  return out.join('\n')
}

async function main() {
  const apiKey = process.env.MNEMO_API_KEY
  const workspaceId = process.env.MNEMO_WORKSPACE_ID
  const base = process.env.MNEMO_TEST_CONTAINER
  const anthropicApiKey = process.env.ANTHROPIC_API_KEY

  if (!apiKey) fail('MNEMO_API_KEY is not set')
  if (!workspaceId) fail('MNEMO_WORKSPACE_ID is not set')
  if (!base) fail('MNEMO_TEST_CONTAINER is not set')
  if (!anthropicApiKey) fail('ANTHROPIC_API_KEY is not set')

  // Unique per-run nonce so concurrent / re-run smokes never collide and a
  // recalled value can't be a stale leftover from a prior run.
  const nonce = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
  const fruit = `dragonfruit-${nonce}`
  // Per spec the model-facing container is "user:${base}-${nonce}". Strip any
  // "type:" prefix off the base id so we don't end up with "user:user:...".
  const colon = base.indexOf(':')
  const cidBase = colon >= 0 ? base.slice(colon + 1) : base
  const containerTag = `user:${cidBase}-${nonce}`

  const getmnemo = new Mnemo({ apiKey, workspaceId })
  const client = new Anthropic({ apiKey: anthropicApiKey })

  const assistant = withMemoryTool({
    client,
    getmnemo,
    containerTag,
    model: MODEL,
    maxTokens: MAX_TOKENS,
  })

  console.log('[smoke] run nonce:    ', nonce)
  console.log('[smoke] container tag:', containerTag)
  console.log('[smoke] model:        ', MODEL)
  console.log('[smoke] fruit codeword:', fruit)

  // Track created memory ids for cleanup even if an assertion throws.
  const createdIds = []

  try {
    // ---- STEP A: a message that should make Claude STORE the fact ----------
    const storeResult = await assistant.run({
      system:
        'You are a helpful assistant with a long-term memory tool. ' +
        'When the user tells you a durable personal fact, save it with the memory ' +
        "tool (action='add') so you can recall it later.",
      messages: [
        {
          role: 'user',
          content: `Please remember that my favorite fruit is ${fruit}.`,
        },
      ],
    })

    if (!(storeResult.memoryToolCalls > 0)) {
      fail(
        'store turn did not fire the memory tool ' +
          `(memoryToolCalls=${storeResult.memoryToolCalls}). ` +
          `assistant text: ${JSON.stringify(storeResult.text)}`,
      )
    }
    console.log(
      `[smoke] OK store: memory tool fired ${storeResult.memoryToolCalls}x on the store turn`,
    )

    // Capture stored ids from the tool_result payloads for cleanup. The
    // wrapper's add path returns { saved, items: [{ id, content }] } as a JSON
    // string in each tool_result block. Best-effort — cleanup is non-fatal.
    for (const m of storeResult.messages ?? []) {
      if (m.role !== 'user' || !Array.isArray(m.content)) continue
      for (const block of m.content) {
        if (!block || block.type !== 'tool_result' || typeof block.content !== 'string') continue
        let parsed
        try {
          parsed = JSON.parse(block.content)
        } catch {
          continue
        }
        for (const item of parsed?.items ?? []) {
          if (item?.id) createdIds.push(item.id)
        }
      }
    }

    // ---- STEP B: a later turn that should RECALL the stored fact -----------
    const recallResult = await assistant.run({
      system:
        'You are a helpful assistant with a long-term memory tool. ' +
        "Before answering questions about the user, search your memory (action='search') " +
        'and base your answer only on what you find.',
      messages: [
        {
          role: 'user',
          content: "What's my favorite fruit? Answer with the exact value you have stored.",
        },
      ],
    })

    const haystack = `${recallResult.text}\n${toolResultText(recallResult)}`
    if (!haystack.includes(fruit)) {
      fail(
        `recall turn did not surface "${fruit}". ` +
          `memoryToolCalls=${recallResult.memoryToolCalls}; ` +
          `assistant text: ${JSON.stringify(recallResult.text)}`,
      )
    }
    console.log(
      `[smoke] OK recall: "${fruit}" found in the recall turn ` +
        `(memoryToolCalls=${recallResult.memoryToolCalls})`,
    )
  } finally {
    // ---- CLEANUP: best-effort delete; failure warns, never fatal ----------
    for (const id of createdIds) {
      try {
        await getmnemo.delete(id)
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        console.warn(`[smoke] WARN: cleanup delete failed for memory ${id}: ${msg}`)
      }
    }
  }

  console.log('\n[smoke] PASS: memory tool fired on store + fact recalled.')
  process.exit(0)
}

main().catch((err) => {
  const msg = err instanceof Error ? (err.stack ?? err.message) : String(err)
  fail(`unexpected error during smoke run:\n${msg}`)
})
