import { strict as assert } from "node:assert";
import { test } from "node:test";
import { bm25, searchTokens } from "./retrieve.ts";

test("removes common Vietnamese question words but keeps meaning words", () => {
  assert.deepEqual(searchTokens("Trí tuệ nhân tạo là gì?"), ["trí", "tuệ", "nhân", "tạo"]);
});

test("finds the matching Vietnamese knowledge entry", () => {
  const hits = bm25("Trí tuệ nhân tạo là gì?", [
    { id: "ai", body: "Trí tuệ nhân tạo là gì? Có ba ứng dụng thực tế nào?" },
    { id: "git", body: "Git là công cụ quản lý mã nguồn." },
  ], 2);
  assert.equal(hits[0]?.id, "ai");
});
