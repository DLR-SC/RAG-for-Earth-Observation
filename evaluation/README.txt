Nicht nur prüfen ob Ergebnisse auf die Query passen, auch prüfen ob zero-shot nicht komplett am halluzinieren ist.

Context Relevance
- Is the retrieved content relevenat to the query?

Groundedness (Faithfulness)
- Is the response supported by the context?

Answer Relevance
- Is the answer relevant to the query?

https://arxiv.org/pdf/2306.05685
https://huggingface.co/learn/cookbook/llm_judge


Is zero shot allowed to have a similar system prompt and same temperature?

If eval is done with gpt-4, it should probably not be used as RAG component to avaoid Self-enhancement bias
    gpt 3.5 oder irgendwas auf together.ai?


- ArangoDB contains very specific data and thus is not good at answering broader questions
- Current impelementation limits RAG to provided context. Probably improves if contraint is removed, but how about hallucinations?