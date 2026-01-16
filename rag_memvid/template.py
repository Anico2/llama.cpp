from pathlib import Path
from typing import List, Optional
import hashlib

import memvid_sdk as mv

class DocumentQA:
    """Document Q&A system with Memvid."""

    SUPPORTED_FORMATS = {'.pdf', '.docx', '.txt', '.md', '.html'}

    def __init__(self, memory_path: str = "documents.mv2", adapter="langchain"):
        
        common = {
            "enable_vec": True,
            "enable_lex": True,
        }
        if Path(memory_path).exists():
            self.mem = mv.use(adapter, memory_path, mode="auto", **common)
        else:
            self.mem = mv.create(memory_path, kind=adapter, **common)
        # mode can be "lex", "vec", "sem", "hybrid", "auto"
        #result = self.mem.verify(deep=True)
        
        self.stats = {"ingested": 0, "failed": 0}

    def ingest_file(self, filepath: str, metadata: Optional[dict] = None) -> bool:
        """Ingest a single file."""
        path = Path(filepath)

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            print(f"⚠️ Unsupported format: {path.suffix}")
            return False

        try:
            # Generate unique ID based on content hash
            content_hash = hashlib.md5(path.read_bytes()).hexdigest()[:8]

            self.mem.put(**{
                "title": path.name,
                "label": "document",
                "file": str(path.absolute()),
                "metadata": {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "hash": content_hash,
                    **(metadata or {})
                },
                "embedding_model":"bge-small",
                "enable_embedding": False,
                "auto_tag":True
            }
            
            )

            self.stats["ingested"] += 1
            print(f"✅ Ingested: {path.name}")
            return True

        except Exception as e:
            self.stats["failed"] += 1
            print(f"❌ Failed: {path.name} - {e}")
            return False

    def ingest_folder(self, folder_path: str, recursive: bool = True) -> dict:
        """Ingest all documents from a folder."""
        folder = Path(folder_path)

        pattern = "**/*" if recursive else "*"
        files = [f for f in folder.glob(pattern)
                 if f.is_file() and f.suffix.lower() in self.SUPPORTED_FORMATS]

        print(f"📂 Found {len(files)} documents to ingest...")

        for filepath in files:
            self.ingest_file(str(filepath))

        return self.stats

    def ask(self, question: str, k: int = 5) -> dict:
        """Ask a question about the documents."""
        result = self.mem.ask(question, k=k)
        print(result["answer"])
        assert 0
        return {
            "answer": result.text,
            "sources": [
                {
                    "title": s.title,
                    "snippet": s.snippet,
                    "score": s.score
                }
                for s in result.sources
            ],
            "confidence": result.confidence if hasattr(result, 'confidence') else None
        }

    def search(self, query: str, k: int = 10) -> List[dict]:
        """Search documents without generating an answer."""
        results = self.mem.find(query, k=k)

        return [
            {
                "title": hit.title,
                "snippet": hit.snippet,
                "score": hit.score,
                "metadata": hit.metadata
            }
            for hit in results.hits
        ]

    def get_stats(self) -> dict:
        """Get document store statistics."""
        stats = self.mem.stats()
        return {
            "total_documents": stats.get("frame_count", 0),
            "size_bytes": stats.get("size_bytes", 0),
            "size_mb": round(stats.get("size_bytes", 0) / 1024 / 1024, 2)
        }


qa = DocumentQA("llmsdocs2.mv2")
qa.ingest_folder("./llms/")
result = qa.ask("what is Mixture of Experts?")
print(result["answer"])