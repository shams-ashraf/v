import requests
import os
import time
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from typing import List, Dict, Tuple

from intent_detect import detect_intent_semantic, apply_document_weights, rerank_chunks, INTENT_DEFINITIONS

load_dotenv()

GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8")

]

GROQ_API_KEYS = [key for key in GROQ_API_KEYS if key]

if not GROQ_API_KEYS:
    raise ValueError("No GROQ API keys found! Please set GROQ_API_KEY_1, GROQ_API_KEY_2, and/or GROQ_API_KEY_3")

GROQ_MODEL = "llama-3.3-70b-versatile"

current_key_index = 0
GROQ_RATE_LIMIT_UNTIL = [0] * len(GROQ_API_KEYS)

MAX_TOTAL_TOKENS = 6000
MAX_OUTPUT_TOKENS = 1000
MIN_CHUNK_SIZE_TOKENS = 200
MAX_CONTEXT_TOKENS = 1800

MAX_CITED_SOURCES = 3
MEMORY_SUMMARY_TOKENS = 100

def get_next_available_key() -> Tuple[str, int]:
    global current_key_index
    
    now = time.time()
    
    for _ in range(len(GROQ_API_KEYS)):
        if now >= GROQ_RATE_LIMIT_UNTIL[current_key_index]:
            key = GROQ_API_KEYS[current_key_index]
            index = current_key_index
            current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
            return key, index
        
        current_key_index = (current_key_index + 1) % len(GROQ_API_KEYS)
    
    earliest_available = min(GROQ_RATE_LIMIT_UNTIL)
    wait_seconds = int(earliest_available - now)
    return None, wait_seconds

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def compress_to_memory(content: str, max_tokens: int = MEMORY_SUMMARY_TOKENS) -> str:
    target_chars = max_tokens * 4
    
    if len(content) <= target_chars:
        return content
    
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
    
    if len(sentences) <= 2:
        return content[:target_chars]
    
    memory = f"{sentences[0]}. [...] {sentences[-1]}."
    
    if len(memory) > target_chars:
        memory = memory[:target_chars] + "..."
    
    return memory

def compress_chat_history(chat_history, max_items=3):
    if not chat_history:
        return ""

    recent_pairs = []
    for i in range(len(chat_history) - 1, -1, -1):
        if chat_history[i]["role"] == "user":
            user_msg = chat_history[i]["content"]
            if i + 1 < len(chat_history) and chat_history[i + 1]["role"] == "assistant":
                assistant_msg = chat_history[i + 1]["content"]
                recent_pairs.insert(0, (user_msg, assistant_msg))
                if len(recent_pairs) >= max_items:
                    break

    if recent_pairs:
        summary = ["=== Previous Conversation ==="]
        for idx, (q, a) in enumerate(recent_pairs, 1):
            summary.append(f"\nUser Question {idx}: {q}")
            if len(a) > 500:
                summary.append(f"Assistant Answer {idx}: {a[:500]}... [truncated]")
            else:
                summary.append(f"Assistant Answer {idx}: {a}")
        summary.append("\n=== End of Previous Conversation ===\n")
        return "\n".join(summary)
    
    return ""

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )

def get_system_prompt(language, lightweight=False):
    if lightweight:
        return (
            "You are an assistant for the Master Biomedical Engineering (MBE) program.\n"
            "You MUST answer ONLY using the provided document excerpts.\n"
            "If the answer is NOT explicitly stated in the documents, reply exactly:\n"
            "\"No sufficient information is available in the provided documents.\""
        )

    return """You are an assistant for the Master Biomedical Engineering (MBE) program.

STRICT RULES (NO EXCEPTIONS):
1. You MUST answer ONLY using the provided document excerpts.
2. DO NOT use any external knowledge, assumptions, or general academic knowledge.
3. DO NOT infer, guess, summarize beyond what is explicitly written.
4. If the documents do NOT clearly contain the answer, you MUST reply exactly with:
   "No sufficient information is available in the provided documents."
5. Every factual statement MUST be directly supported by a cited document and page number.
6. If a question asks for something not present (even partially), do NOT answer it.

This is a document-grounded academic assistant. Accuracy is more important than completeness.
"""

def trim_system_prompt(system_prompt: str, max_tokens: int) -> str:
    prompt_tokens = estimate_tokens(system_prompt)
    
    if prompt_tokens <= max_tokens:
        return system_prompt
    
    target_chars = max_tokens * 4
    trimmed = system_prompt[:target_chars]
    
    return trimmed

def check_if_answer_insufficient(answer: str, language: str) -> bool:
    answer_lower = answer.lower().strip()
    
    if len(answer_lower) < 50:
        return True
    
    insufficient_patterns = [
        "no information", "not found", "cannot find", "insufficient", 
        "not available", "i don't see", "no relevant", "no data",
        "not mentioned", "does not","No sufficient information in the available documents."
    ]
    
    for p in insufficient_patterns:
        if p in answer_lower:
            return True

    return False

def check_if_answer_too_brief(answer: str, min_sentences=3) -> bool:
    sentences = [s.strip() for s in answer.split('.') if s.strip() and len(s.strip()) > 10]
    return len(sentences) < min_sentences or len(answer) < 120

def separate_tables_and_text(chunks):
    tables = []
    texts = []
    
    for chunk in chunks:
        chunk_type = chunk["metadata"].get("type", "text")
        if chunk_type in ["table_with_context", "table"]:
            tables.append(chunk)
        else:
            texts.append(chunk)
    
    return tables, texts

def filter_chunks_by_size(chunks: List[Dict], min_tokens: int = MIN_CHUNK_SIZE_TOKENS) -> List[Dict]:
    filtered = []
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk["content"])
        chunk_type = chunk["metadata"].get("type", "text")
        
        if chunk_type in ["table_with_context", "table"]:
            filtered.append(chunk)
            continue
        
        if chunk_tokens >= min_tokens:
            filtered.append(chunk)
    
    return filtered

def calculate_texts_per_batch(intent: str, debug: bool = False) -> int:    
    complex_intents = ["modules_list_overview", "exam_regulations", "module_details_content"]
    
    if intent in complex_intents:
        texts_per_batch = 3
    else:
        texts_per_batch = 2
    
    return texts_per_batch

def drop_lowest_weight_chunk(used_chunks: List[Dict], debug: bool = False) -> List[Dict]:
    text_chunks = [c for c in used_chunks if c.get("type", "text") not in ["table_with_context", "table"]]
    
    if len(text_chunks) <= 2:
        return used_chunks
    
    droppable = [c for c in text_chunks if not c.get("is_cited", False)]
    
    if not droppable:
        return used_chunks
    
    lowest_chunk = min(droppable, key=lambda c: c.get("doc_weight", 0.5))
    
    filtered = [c for c in used_chunks if c != lowest_chunk]
    
    return filtered

def calculate_tables_per_iteration(weighted_tables: List[Dict], intent: str, debug: bool = False) -> int:
    if not weighted_tables:
        return 0
        
    max_weight = 0
    high_weight_count = 0
    
    for table in weighted_tables:
        doc_weight = table.get("doc_weight", 0.5)
        
        if doc_weight > max_weight:
            max_weight = doc_weight
        
        if doc_weight >= 2.5:
            high_weight_count += 1
    
    if max_weight >= 2.5:
        tables_per_iter = 2
    elif max_weight >= 1.5:
        tables_per_iter = 1
    elif max_weight > 0:
        tables_per_iter = 1
    else:
        tables_per_iter = 0
    
    return tables_per_iter

def search_with_smart_intent(collection, query, detected_intent: str, n_texts=4, debug=False):
    all_table_pages = []
    try:
        table_results = collection.query(
            query_texts=[query],
            n_results=20,
            where={"type": {"$in": ["table_with_context", "table"]}}
        )
        
        table_chunks = []
        for doc, meta in zip(table_results["documents"][0], table_results["metadatas"][0]):
            table_chunks.append({"content": doc, "metadata": meta})
        
        weighted_tables = apply_document_weights(table_chunks, detected_intent, debug=debug)
        
        reranked_tables = rerank_chunks(
            weighted_tables, 
            top_k=len(weighted_tables),
            debug=debug
        )
        
        for chunk in reranked_tables:
            all_table_pages.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "doc_weight": chunk.get("doc_weight", 0.5),
                "source": chunk.get("source", "Unknown")
            })
        
    except Exception as e:
        pass
    
    all_texts = []
    try:
        text_results = collection.query(
            query_texts=[query],
            n_results=n_texts * 3,
            where={"type": {"$nin": ["table_with_context", "table"]}}
        )
        
        text_chunks = []
        for doc, meta in zip(text_results["documents"][0], text_results["metadatas"][0]):
            chunk_type = meta.get("type", "text")
            if chunk_type not in ["table_with_context", "table"]:
                text_chunks.append({"content": doc, "metadata": meta})
        
        weighted_texts = apply_document_weights(text_chunks, detected_intent, debug=debug)
        
        reranked_texts = rerank_chunks(weighted_texts, top_k=n_texts * 2, debug=debug)
        
        filtered_texts = filter_chunks_by_size(reranked_texts, MIN_CHUNK_SIZE_TOKENS)
        final_texts = filtered_texts[:n_texts]
        
        for chunk in final_texts:
            all_texts.append({
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "doc_weight": chunk.get("doc_weight", 0.5)
            })
        
    except Exception as e:
        pass
    
    return all_table_pages, all_texts

def extract_used_sources_from_answer(answer: str, used_chunks: list) -> list:
    actually_used = []
    answer_lower = answer.lower()
    
    for chunk in used_chunks:
        if isinstance(chunk, dict):
            if "metadata" in chunk:
                source = chunk["metadata"].get("source", "")
                page = chunk["metadata"].get("page", "")
            else:
                source = chunk.get("source", "")
                page = chunk.get("page", "")
        else:
            continue
        
        patterns = [
            f"page {page}",
            f"p{page}",
            f"p.{page}",
            f"({source}",
            str(source).lower().split('/')[-1].replace('.pdf', ''),
        ]
        
        found = False
        for pattern in patterns:
            if str(pattern).lower() in answer_lower:
                found = True
                break
        
        if found:
            actually_used.append(chunk)
    
    return actually_used

def check_source_diversity(cumulative_sources: List[Dict], new_sources: List[Dict]) -> bool:
    if not cumulative_sources:
        return True
    
    existing_keys = set()
    for src in cumulative_sources:
        key = f"{src.get('source', '')}_{src.get('page', '')}"
        existing_keys.add(key)
    
    new_count = 0
    for src in new_sources:
        key = f"{src.get('source', '')}_{src.get('page', '')}"
        if key not in existing_keys:
            new_count += 1
    
    diversity_ratio = new_count / len(new_sources) if new_sources else 0
    return diversity_ratio >= 0.5

def answer_question_with_groq(query, relevant_chunks, chat_history=None, user_language="en", 
                               collection=None, max_iterations=5, 
                               texts_per_batch=4,):
    
    detected_intent = detect_intent_semantic(query, debug=False)
    
    if not collection:
        table_pages, texts = separate_tables_and_text(relevant_chunks)
        all_table_pages = table_pages
        all_texts = texts
    else:
        all_table_pages, all_texts = search_with_smart_intent(
            collection, 
            query, 
            detected_intent=detected_intent,
            n_texts=max_iterations * texts_per_batch,
            debug=False
        )
    
    conversation_summary = compress_chat_history(chat_history, max_items=2)
    
    iteration = 1
    text_offset = 0
    table_offset = 0
    
    cumulative_cited_sources = []
    total_session_tokens = 0
    last_answer = ""
    
    tables_per_iteration = calculate_tables_per_iteration(all_table_pages, detected_intent, debug=False)
    texts_per_batch = calculate_texts_per_batch(detected_intent, debug=False)
    
    while iteration <= max_iterations:
        context_parts = []
        used_chunks = []
        
        if cumulative_cited_sources:
            active_cited = cumulative_cited_sources[-MAX_CITED_SOURCES:]
            
            for cited_chunk in active_cited:
                source = cited_chunk["source"]
                page = cited_chunk["page"]
                chunk_type = cited_chunk.get("type", "text")
                
                if iteration > 1:
                    content = compress_to_memory(cited_chunk["content"])
                    prefix = "📌 MEMORY"
                else:
                    content = cited_chunk["content"]
                    prefix = "🔥 CITED"
                
                is_table = chunk_type in ["table_with_context", "table"]
                type_marker = "[TABLE]" if is_table else "[TEXT]"
                
                context_parts.append(f"[{prefix} {type_marker} | {source} p{page}]\n{content}")
                used_chunks.append({**cited_chunk, "is_cited": True})
        
        if all_table_pages and table_offset < len(all_table_pages):
            current_tables = all_table_pages[table_offset:table_offset + tables_per_iteration]
            
            if current_tables:
                for tbl in current_tables:
                    source = tbl["metadata"].get("source", "Unknown")
                    page = tbl["metadata"].get("page", "N/A")
                    
                    context_parts.append(f"[📊 TABLE [NEW] | {source} p{page}]\n{tbl['content']}")
                    used_chunks.append({
                        "source": source, 
                        "page": page, 
                        "content": tbl["content"], 
                        "type": "table_with_context",
                        "doc_weight": tbl.get("doc_weight", 0.5)
                    })
                
                table_offset += len(current_tables)
        
        current_texts = all_texts[text_offset:text_offset + texts_per_batch]
        
        if current_texts:
            for chunk in current_texts:
                source = chunk["metadata"].get("source", "Unknown")
                page = chunk["metadata"].get("page", "N/A")
                content = chunk["content"]
                
                context_parts.append(f"[📄 TEXT [NEW] | {source} p{page}]\n{content}")
                used_chunks.append({
                    "source": source, 
                    "page": page, 
                    "content": content, 
                    "type": "text",
                    "doc_weight": chunk.get("doc_weight", 0.5)
                })
        
        if not context_parts:
            if last_answer:
                return last_answer, used_chunks
            else:
                return "No information available in the documents about this topic.", []
        
        context = "\n\n---\n\n".join(context_parts)
        context_tokens = estimate_tokens(context)
        
        if context_tokens > MAX_CONTEXT_TOKENS:
            used_chunks = drop_lowest_weight_chunk(used_chunks, debug=False)
            
            context_parts = []
            for chunk in used_chunks:
                source = chunk.get("source", "Unknown")
                page = chunk.get("page", "N/A")
                content = chunk.get("content", "")
                chunk_type = chunk.get("type", "text")
                
                is_table = chunk_type in ["table_with_context", "table"]
                type_marker = "[TABLE]" if is_table else "[TEXT]"
                marker = "🔥 CITED" if chunk.get("is_cited", False) else "📄 NEW"
                
                context_parts.append(f"[{marker} {type_marker} | {source} p{page}]\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            context_tokens = estimate_tokens(context)
        
        context_tokens = estimate_tokens(context)
        history_tokens = estimate_tokens(conversation_summary)
        query_tokens = estimate_tokens(query)
        
        system_prompt = get_system_prompt(user_language, lightweight=(iteration > 1))
        system_tokens = estimate_tokens(system_prompt)
        
        input_tokens = context_tokens + history_tokens + query_tokens + system_tokens
        estimated_total = input_tokens + MAX_OUTPUT_TOKENS
        
        if estimated_total > MAX_TOTAL_TOKENS:
            overage = estimated_total - MAX_TOTAL_TOKENS
            
            new_system_max = max(50, system_tokens - overage - 100)
            system_prompt = trim_system_prompt(system_prompt, new_system_max)
            system_tokens = estimate_tokens(system_prompt)
            
            input_tokens = context_tokens + history_tokens + query_tokens + system_tokens
            estimated_total = input_tokens + MAX_OUTPUT_TOKENS
        
        user_content = f"""{conversation_summary if conversation_summary else ''}

AVAILABLE DOCUMENT SOURCES:
{context}

CURRENT QUESTION: {query}

Instructions:
- Answer comprehensively using ALL provided sources
- Always cite sources clearly (e.g., "According to [Document], page X...")
- If insufficient info: "No sufficient information in the available documents."
"""

        data = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.05,
            "max_tokens": MAX_OUTPUT_TOKENS,
        }

        try:
            api_key, key_index = get_next_available_key()
            
            if api_key is None:
                return (f"All API keys are rate limited. Please wait {key_index} seconds.", [])
            
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=data,
                timeout=60
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()
            last_answer = answer
            
            output_tokens = estimate_tokens(answer)
            actual_total = input_tokens + output_tokens
            total_session_tokens += actual_total
            
            is_insufficient = check_if_answer_insufficient(answer, user_language)
            is_too_brief = check_if_answer_too_brief(answer, min_sentences=3)
            
            if not is_insufficient and not is_too_brief:
                return answer, used_chunks
            
            actually_used_sources = extract_used_sources_from_answer(answer, used_chunks)
            
            if iteration >= 2:
                has_diversity = check_source_diversity(cumulative_cited_sources, actually_used_sources)
                if not has_diversity:
                    return last_answer, used_chunks
            
            if actually_used_sources:
                for src in actually_used_sources:
                    is_duplicate = False
                    for existing in cumulative_cited_sources:
                        if (existing["source"] == src.get("source") and 
                            existing["page"] == src.get("page")):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        cumulative_cited_sources.append(src)
            
            if iteration >= max_iterations:
                return answer, used_chunks
            
            text_offset += texts_per_batch
            
            if text_offset >= len(all_texts):
                return answer, used_chunks
            
            iteration += 1

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait_time = int(retry_after) if retry_after else 60
                GROQ_RATE_LIMIT_UNTIL[key_index] = time.time() + wait_time
                
                api_key, new_key_index = get_next_available_key()
                if api_key is None:
                    return (f"All API keys are rate limited. Please wait {new_key_index} seconds.", [])
                continue
            return (f"Error: {str(e)}", [])
        except Exception as e:
            return (f"Error: {str(e)}", [])
    
    return last_answer, used_chunks