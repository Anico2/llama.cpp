import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("Chunking")

class ChunkSchema(BaseModel):
    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="Summary of this chunk to help retrieval")
    original_text: str = Field(description="The exact text from the document")

class ChunksResponse(BaseModel):
    chunks: List[ChunkSchema]

class LLMSemanticChunker:
    def __init__(self, llm):
        self.llm_chain = llm.with_structured_output(ChunksResponse)
        # Despite doing semantic chunking, we first pre-split to manageable sizes
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, 
            chunk_overlap=0
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        
        pre_split_docs = self.pre_splitter.split_documents(documents)
        logger.info(f"Pre-split into {len(pre_split_docs)} segments for semantic processing...")

        for i, doc in enumerate(pre_split_docs):
            try:
                
                source_name = doc.metadata.get('source', 'unknown')
                
                prompt = f"""
                Analyze the following document segment. 
                Split it into logical, self-contained chunks.
                For each chunk, generate a HEADLINE, a SUMMARY, and keep the ORIGINAL TEXT.
                
                Context (Source File): {source_name}
                
                Text to Split:
                {doc.page_content}
                """
                
                response = self.llm_chain.invoke(prompt)
                
                # If LLM fails or returns empty, we safely fallback to original
                if not response or not response.chunks:
                    final_chunks.append(doc)
                    continue

                for c in response.chunks:
                    augmented_content = (
                        f"FILE: {source_name}\n"
                        f"HEADLINE: {c.headline}\n"
                        f"SUMMARY: {c.summary}\n"
                        f"CONTENT: {c.original_text}"
                    )
                    
                    new_doc = Document(
                        page_content=augmented_content,
                        metadata={
                            **doc.metadata,
                            "headline": c.headline,
                            "summary": c.summary,
                            "original_text": c.original_text
                        }
                    )
                    final_chunks.append(new_doc)
            
            except Exception as e:
                logger.warning(f"Semantic chunking failed for batch {i}: {e}. Keeping original.")
                final_chunks.append(doc)
                
        return final_chunks