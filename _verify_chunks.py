import json

chunks = json.loads(open("output/chunks/all_chunks.json", encoding="utf-8").read())
ncert = [c for c in chunks if c.get("source") == "ncert"]
pyq = [c for c in chunks if c.get("source") == "pyq"]

print(f"Total: {len(chunks)} chunks ({len(ncert)} NCERT, {len(pyq)} PYQ)")
print()

# Show a few NCERT chunks with heading context
for i in [0, 5, 20, 50]:
    if i < len(ncert):
        c = ncert[i]
        print(f"NCERT Chunk #{i}:")
        print(f"  ID:      {c['chunk_id']}")
        print(f"  Heading: {c.get('heading_context', 'NONE')}")
        print(f"  File:    {c.get('file_name', '?')}")
        has_h = "[" in c["text"][:5]
        print(f"  Has [heading] prefix: {has_h}")
        print(f"  Text:    {c['text'][:130]}...")
        print()

# Count context coverage
with_context = sum(1 for c in ncert if c.get("heading_context"))
print(f"Heading context coverage: {with_context}/{len(ncert)} NCERT chunks ({100*with_context/max(1,len(ncert)):.0f}%)")
