import { strict as assert } from "node:assert";
import { test } from "node:test";
import { trySolve } from "./solver.ts";

test("solves natural Vietnamese arithmetic wording", () => {
  assert.equal(trySolve("1+1 bằng mấy?")?.answer, "2");
  assert.equal(trySolve("Hãy tính 2*3")?.answer, "6");
  assert.equal(trySolve("25% của 200")?.answer, "50");
});
