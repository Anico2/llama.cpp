from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document

# Define the schema for the LLM to follow
class ChunkSchema(BaseModel):
    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="Summary of this chunk to help retrieval")
    original_text: str = Field(description="The exact text from the document")

class ChunksResponse(BaseModel):
    chunks: List[ChunkSchema]

class LLMSemanticChunker:
    def __init__(self, llm):
        # We use .with_structured_output to ensure the LLM returns our Pydantic model
        self.llm_chain = llm.with_structured_output(ChunksResponse)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        final_chunks = []
        
        for doc in documents:
            prompt = f"""
            Split the following document into logical, overlapping chunks.
            For each chunk, provide a headline, a short summary, and the original text.
            Ensure NO part of the document is lost.
            
            Document Source: {doc.metadata.get('source', 'unknown')}
            Content:
            {doc.page_content}
            """
            
            # Get structured response from LLM
            response = self.llm_chain.invoke(prompt)
            
            for c in response.chunks:
                # Merge headline and summary into the page content for better RAG hits
                augmented_content = f"HEADLINE: {c.headline}\nSUMMARY: {c.summary}\nTEXT: {c.original_text}"
                
                new_doc = Document(
                    page_content=augmented_content,
                    metadata={
                        **doc.metadata,
                        "headline": c.headline,
                        "summary": c.summary
                    }
                )
                final_chunks.append(new_doc)
                
        return final_chunks