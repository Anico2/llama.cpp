import logging
import re
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser

from chb.utils.clients import DecoderClient

logger = logging.getLogger(__name__)


class ChunkSchema(BaseModel):
    headline: str = Field(description="Brief heading for this chunk")
    summary: str = Field(description="Summary of this chunk to help retrieval")
    original_text: str = Field(description="The exact text from the document")


class ChunksResponse(BaseModel):
    chunks: List[ChunkSchema]


class LLMSemanticChunker:
    def __init__(self, llm: DecoderClient):
        """_summary_

        Args:
            llm (DecoderClient): _description_
        """
        self.llm_chain = llm.with_structured_output(ChunksResponse)
        # Despite doing semantic chunking, we first pre-split to manageable sizes
        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=0
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:

        final_chunks = []

        pre_split_docs = self.pre_splitter.split_documents(documents)
        logger.info(
            f"Pre-split into {len(pre_split_docs)} segments for semantic processing..."
        )

        for i, doc in enumerate(pre_split_docs):
            try:
                source_name = doc.metadata.get("source", "unknown")

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
                            "original_text": c.original_text,
                        },
                    )
                    final_chunks.append(new_doc)

            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed for batch {i}: {e}. Keeping original."
                )
                final_chunks.append(doc)

        return final_chunks


class TocEntry(BaseModel):
    start: int = Field(description="The starting page number of the section")
    end: int = Field(description="The Ending page number of the section")


class TocResponse(BaseModel):
    toc: List[TocEntry] = Field(description="List of table of contents entries")


class TableOfContentsChunker:
    def __init__(
        self, llm: DecoderClient, delim: str | None = None, max_chunk_size: int = 4000
    ):
        """_summary_

        Args:
            llm (DecoderClient): _description_
            delim (str | None, optional): _description_. Defaults to None.
            max_chunk_size (int, optional): _description_. Defaults to 4000.
        """
        self.parser = JsonOutputParser(pydantic_object=TocResponse)
        self.llm = llm
        self.delim = delim
        self.max_chunk_size = max_chunk_size

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    def _sanitize_toc(
        self, raw_toc: List[Dict], total_pages: int
    ) -> List[Dict[str, int]]:
        """Converts Page Numbers (1-based) to valid Python List Indices (0-based).


        Args:
            raw_toc (List[Dict]): _description_
            total_pages (int): _description_

        Returns:
            List[Dict[str, int]]: _description_
        """
        valid_indices = []
        for entry in raw_toc:
            p_start = entry.get("start") if isinstance(entry, dict) else entry.start
            p_end = entry.get("end") if isinstance(entry, dict) else entry.end

            if p_start > p_end:
                continue

            # Attention: we have to shift since documents strats from 0
            start_idx = max(0, p_start - 1)

            end_idx = min(p_end, total_pages)

            if start_idx >= total_pages:
                continue

            valid_indices.append({"start": start_idx, "end": end_idx})

        # to be sure, we sort by start index
        valid_indices.sort(key=lambda x: x["start"])

        return valid_indices

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        1. Uses the first 5 pages to extract TOC via LLM.
        2. Sanitizes the TOC.
        3. Chunks the full document list based on TOC ranges.
        4. Recursively splits any chunks that exceed max_chunk_size.
        """

        # Step 1: sanitize documents
        if len(documents) == 1:
            # if pdf has been loaded as on string only
            # extract first 5
            assert self.delim is not None, (
                "delim must be provided for single document input."
            )
            doc_pages_list = documents[0].page_content.split(self.delim)
            total_pages = len(doc_pages_list)
            input_pages_for_prompt = doc_pages_list[:5]
            # TODO: test that in this case we have no error in sanitize toc
        else:
            total_pages = len(documents)
            input_pages_for_prompt = [d.page_content for d in documents[:5]]

        doc_pages_str = "\n\n---PAGE BREAK---\n\n".join(input_pages_for_prompt)

        # Step 2: ask llm to extract toc
        prompt = f"""
            Given the following document pages, identify the Table of Contents (TOC).
            Return the page ranges (start, end) for each section.
            
            {self.parser.get_format_instructions()} 
            
            IMPORTANT: Do not include any explanation or conversational text. 
            Only output the JSON object.

            Document Pages:
            {doc_pages_str}
            """

        print("Extracting TOC")
        response_msg = self.llm.invoke(prompt)
        response_str = response_msg.content

        try:
            parsed_data = self.parser.parse(response_str)
            raw_toc = parsed_data["toc"]
            print(f"Raw TOC extracted: {len(raw_toc)} entries.")
        except Exception as e:
            print(f"Parsing failed. Raw output: {response_str}")
            raise e

        # sanitize toc
        clean_indices = self._sanitize_toc(raw_toc, total_pages)
        final_chunks = []

        if len(documents) == 1:
            docs_text = documents[0].page_content
            assert self.delim is not None, (
                "delim must be provided for single document input."
            )
            doc_pages_list = docs_text.split(self.delim)
            documents = [
                Document(
                    page_content=page,
                    metadata={
                        "source": documents[0].metadata.get("source", "unknown"),
                        "page": i,
                    },
                )
                for i, page in enumerate(doc_pages_list)
            ]

        for entry in clean_indices:
            section_pages = documents[entry["start"] : entry["end"]]

            if not section_pages:
                continue

            merged_content = "\n\n".join([p.page_content for p in section_pages])

            base_metadata = {
                "source": section_pages[0].metadata.get("source", "unknown"),
                "toc_start_index": entry["start"],
                "toc_end_index": entry["end"],
                "section_length_chars": len(merged_content),
                "section_num_pages": entry["end"] - entry["start"],
            }

            if len(merged_content) > self.max_chunk_size:
                temp_doc = Document(page_content=merged_content, metadata=base_metadata)
                sub_chunks = self.recursive_splitter.split_documents([temp_doc])
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(
                    Document(page_content=merged_content, metadata=base_metadata)
                )

        print(
            f"Created {len(final_chunks)} chunks from {len(clean_indices)} TOC entries."
        )

        return final_chunks


class QAChunking:
    def __init__(self, llm: DecoderClient):
        """_summary_

        Args:
            llm (DecoderClient): _description_
        """
        self.llm_chain = llm.with_structured_output(ChunksResponse)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """_summary_

        Args:
            documents (List[Document]): _description_

        Returns:
            List[Document]: _description_
        """
        docs = " ".join([d.page_content for d in documents])

        text = re.sub(r"(?<!\n)\n(?!\n)", " ", docs)

        pt = r"(Q:|Question|\d+\.)"
        splits = re.split(pt, text)

        chunks, current_chunk = [], ""

        for fragment in splits:
            if not fragment.strip():
                continue

            if re.match(pt, fragment):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = fragment
            else:
                current_chunk += fragment

        if current_chunk:
            chunks.append(current_chunk)

        return [Document(page_content=c, metadata={"source": "..."}) for c in chunks]
