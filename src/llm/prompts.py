WEEKLY_REPORT_PROMPT = """You are an AI analyst for an industrial waste incineration facility in Japan.
Based on the following waste composition data, respond in this exact format:
TREND: [one sentence about the dominant waste type and overall volume]
ANOMALY: [one sentence about anomalies detected, or "No anomalies detected this week"]
ACTION: [one sentence operational recommendation for facility managers]

Data:
{context}

Analysis:"""

ANOMALY_EXPLANATION_PROMPT = """You are an operations analyst at an industrial waste incineration facility in Japan.
A statistical anomaly was detected on {date}.

Anomaly details:
- Total annotations: {total_count}
- Anomaly score: {anomaly_score} (more negative = more anomalous)
- Statistical finding: {explanation}

Write a 2-sentence operational explanation for facility managers that:
1. Describes what happened in plain language
2. Suggests a possible operational cause or recommended action

Explanation:"""

RAG_PROMPT = """You are an AI analyst for an industrial waste incineration facility in Japan.
Answer the following question based only on the provided context.
If the answer is not in the context, say "I don't have enough data to answer this."
Be concise and professional.

Context:
{context}

Question: {question}

Answer:"""
