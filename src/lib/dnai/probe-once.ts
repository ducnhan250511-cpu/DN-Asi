import knowledgeJson from "./knowledge.json";
import { bm25 } from "./retrieve";

const knowledge = knowledgeJson as { id: string; body: string }[];
const docs = knowledge.map((a) => ({ id: a.id, body: a.body }));
for (const q of ["Xin chào", "DNA là gì?", "Bạn tên gì?", "Thủ đô nước ta là đâu?", "Quang hợp diễn ra ở đâu?", "for và while khác nhau ra sao?"]) {
  const hits = bm25(q, docs, 3);
  console.log("\n", q);
  for (const h of hits) console.log(h.score.toFixed(3), h.id);
}
