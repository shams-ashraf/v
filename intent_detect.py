import re
from typing import List, Dict, Tuple
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import numpy as np

SEMANTIC_MODEL = SentenceTransformer('intfloat/multilingual-e5-large')

INTENT_DEFINITIONS = {
    "modules_list_overview": {
        "semantic_query": "What modules courses curriculum electives are available in the semester program?",
        "description": "User asking about list of modules, courses, electives, curriculum structure,semester plans,semesters",
        "important_docs": {
            "SPO_MBE": 3,
            "Studien- und Prüfungsplan": 3.0,
            "Study Guide": 2.5,
            "Module Handbook": 1.5  
        },
        "default_weight": 0.6
    },
    
    "module_details_content": {
        "semantic_query": "Explain detailed content objectives requirements workload of specific module course",
        "description": "User asking about details, content, learning objectives of a specific module",
        "important_docs": {
            "Module Handbook": 3.0,  
            "ModulHandbook": 3.0,
            "SPO_MBE": 2.5, 
            "Studien- und Prüfungsplan": 2.5
        },
        "default_weight": 0.5
    },
    
    "thesis_registration": {
        "semantic_query": "How to register submit master thesis requirements deadline supervisor",
        "description": "User asking about thesis registration, requirements, deadlines, submission",
        "important_docs": {
            "SPO_MBE": 2.5,
            "Prüfungsordnung": 2.0,
            "Guide_for_writing": 2.5,
            "Notes_on_final_theses": 3.0  
        },
        "default_weight": 0.5
    },
    
    "exam_regulations": {
        "semantic_query": "Exam rules grading passing failing retake attempts assessment",
        "description": "User asking about exam regulations, grading, retakes",
        "important_docs": {
            "SPO_MBE": 3.0,
            "Prüfungsordnung": 3.0,
            "Studien- und Prüfungsplan": 2.0
        },
        "default_weight": 0.6
    },
    
    "admission_requirements": {
        "semantic_query": "Admission requirements application prerequisites qualifications language bachelor degree",
        "description": "User asking about admission, application, prerequisites",
        "important_docs": {
            "SPO_MBE": 2.5,
            "Zulassungsordnung": 3.0
        },
        "default_weight": 0.6
    },
    
    "writing_guidelines": {
        "semantic_query": "How to write format structure citations references bibliography layout thesis paper",
        "description": "User asking about writing, formatting, citations",
        "important_docs": {
            "Guide_for_writing": 3.0,
            "Notes_on_final_theses": 3.0,
            "SPO_MBE": 1.5
        },
        "default_weight": 0.5
    },
    
    "general": {
        "semantic_query": "",
        "description": "General query that doesn't fit other categories",
        "important_docs": {},
        "default_weight": 1.0
    }
}

def detect_intent_semantic(query: str, debug: bool = False) -> str:
    query_embedding = SEMANTIC_MODEL.encode(query, convert_to_tensor=True)
    
    intent_scores = {}
    
    for intent_name, intent_config in INTENT_DEFINITIONS.items():
        if intent_name == "general" or not intent_config["semantic_query"]:
            continue
        
        intent_embedding = SEMANTIC_MODEL.encode(
            intent_config["semantic_query"], 
            convert_to_tensor=True
        )
        
        similarity = float(np.dot(query_embedding, intent_embedding) / 
                          (np.linalg.norm(query_embedding) * np.linalg.norm(intent_embedding)))
        
        intent_scores[intent_name] = similarity
    
    if intent_scores:
        detected_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
        max_score = intent_scores[detected_intent]
        
        if max_score < 0.3:
            if debug:
                print(f"\n   ⚠️ Low similarity ({max_score:.4f}) -> using 'general'")
            return "general"
        
        return detected_intent
    
    return "general"

def apply_document_weights(chunks: List[Dict], intent: str, debug: bool = False) -> List[Dict]:
    intent_config = INTENT_DEFINITIONS.get(intent, INTENT_DEFINITIONS["general"])
    important_docs = intent_config["important_docs"]
    default_weight = intent_config["default_weight"]
    
    if debug:
        print(f"\n⚖️ Applying Document Weights for Intent: '{intent}'")
        print(f"   Important Docs: {list(important_docs.keys())}")
        print(f"   Default Weight: {default_weight}")
    
    weighted_chunks = []
    weight_stats = defaultdict(int)
    
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            if "metadata" in chunk:
                source = chunk["metadata"].get("source", "")
                content = chunk.get("content", "")
                metadata = chunk["metadata"]
            else:
                source = chunk.get("source", "")
                content = chunk.get("content", "")
                metadata = chunk
        else:
            continue
        
        doc_weight = default_weight
        matched_doc = None
        
        for doc_pattern, weight in important_docs.items():
            if doc_pattern.lower() in source.lower():
                doc_weight = weight
                matched_doc = doc_pattern
                break
        
        base_similarity = 1.0 - (idx * 0.5 / max(len(chunks), 1))
        weighted_score = base_similarity * doc_weight
        
        weighted_chunk = {
            "content": content,
            "metadata": metadata,
            "source": source,
            "page": metadata.get("page", "N/A"),
            "type": metadata.get("type", "text"),
            "base_similarity": base_similarity,
            "doc_weight": doc_weight,
            "weighted_score": weighted_score,
            "matched_doc_pattern": matched_doc
        }
        
        weighted_chunks.append(weighted_chunk)
        weight_stats[doc_weight] += 1
    
    if debug:
        print(f"\n   📊 Weight Distribution:")
        for weight, count in sorted(weight_stats.items(), reverse=True):
            print(f"      {weight:.1f}x: {count} chunks")
    
    return weighted_chunks

def rerank_chunks(weighted_chunks: List[Dict], top_k: int = 15, debug: bool = False) -> List[Dict]:
    sorted_chunks = sorted(weighted_chunks, key=lambda x: x["weighted_score"], reverse=True)
    top_chunks = sorted_chunks[:top_k]
    
    if debug:
        print(f"\n📈 Re-ranking Results (Top {top_k}):")
        for i, chunk in enumerate(top_chunks[:5], 1):
            print(f"   {i}. {chunk['source']} p{chunk['page']}")
            print(f"      Base: {chunk['base_similarity']:.3f} × Weight: {chunk['doc_weight']:.1f} = {chunk['weighted_score']:.3f}")
            if chunk['matched_doc_pattern']:
                print(f"      ✨ Matched: {chunk['matched_doc_pattern']}")
    
    return top_chunks

def smart_retrieve_chunks(collection, query: str, n_results: int = 20, top_k: int = 15, debug: bool = False) -> Tuple[List[Dict], str]:
    
    detected_intent = detect_intent_semantic(query, debug=debug)    
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        
        retrieved_chunks = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({"content": doc, "metadata": meta})
                
    except Exception as e:
        return [], detected_intent
    
    weighted_chunks = apply_document_weights(retrieved_chunks, detected_intent, debug=debug)
    
    final_chunks = rerank_chunks(weighted_chunks, top_k=top_k, debug=debug)
    
    return final_chunks, detected_intent