from src.chunking import ChunkingStrategyComparator

text = open("data/output.md", encoding="utf-8").read()
comp = ChunkingStrategyComparator().compare(text, chunk_size=600)
for k, v in comp.items():
    print(f"{k}: Count={v['count']}, AvgLength={v['avg_length']:.2f}")
