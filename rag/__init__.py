# RAG (Retrieval-Augmented Generation) система

# Новые легковесные компоненты (рекомендуется)
from .hybrid_rag import HybridRAGSystem, SearchMethod, HybridSearchResult, get_hybrid_rag, search_hybrid, add_documents_hybrid
from .lightweight_embeddings import EmbeddingProvider, FastEmbedProvider, TFIDFProvider, RandomProvider, get_embedding_provider, embed_texts, embed_query
from .sqlite_vector_store import SQLiteVectorStore, SearchResult as VectorSearchResult
from .minimal_rag import MinimalRAG, MinimalDocument, MinimalSearchResult, get_minimal_rag, search_minimal, add_to_minimal_rag
# Оптимизированные компоненты с высокой производительностью
from .optimized_rag import OptimizedRAGManager, SearchResult as OptimizedSearchResult, SearchMetrics as OptimizedSearchMetrics

# Legacy компоненты (deprecated - использовать новые)
from .rag_manager import RAGManager, SearchResult, SearchMetrics

__all__ = [
    # Новые компоненты (рекомендуется)
    'HybridRAGSystem', 'SearchMethod', 'HybridSearchResult',
    'get_hybrid_rag', 'search_hybrid', 'add_documents_hybrid',
    'EmbeddingProvider', 'FastEmbedProvider', 'TFIDFProvider', 'RandomProvider',
    'get_embedding_provider', 'embed_texts', 'embed_query',
    'SQLiteVectorStore', 'VectorSearchResult',
    'MinimalRAG', 'MinimalDocument', 'MinimalSearchResult',
    'get_minimal_rag', 'search_minimal', 'add_to_minimal_rag',
    # Оптимизированные компоненты
    'OptimizedRAGManager', 'OptimizedSearchResult', 'OptimizedSearchMetrics',

    # Legacy (deprecated)
    'RAGManager', 'SearchResult', 'SearchMetrics'
]
