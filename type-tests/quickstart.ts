// Compile-only check: the README quickstart must type-check against the real
// @anthropic-ai/sdk client. Run with `npm run test:types`; never executed.
import Anthropic from "@anthropic-ai/sdk";
import { withMemoryTool } from "../src/index.js";

export const agent = withMemoryTool({
  client: new Anthropic(),
  apiKey: "k",
  containerTag: "user:jane",
  model: "claude-sonnet-4-6",
  maxTokens: 8192,
  thinkingBudgetTokens: 4000,
});
