import argparse
import logging
import sys
from dotenv import load_dotenv
from pathlib import Path
from graph_builder import build_graph
from components.utils import ingest_documents


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(Path(__file__).name.removesuffix(".py"))

def main():
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(description="LangGraph-based RAG System")
    parser.add_argument("--ingest", help="Path to PDF folder to ingest", type=str)
    parser.add_argument("--query", help="Ask a question to the agent", type=str)
    parser.add_argument("--visualize", help="Optionally prints Mermaid graph",action="store_true", )
    
    args = parser.parse_args()

    # If asked, we ingest document. We conditionally do this to
    # avoid re-ingesting on every run. We exit after ignestion fot the moment,
    # to keep operations simple and atomic.
    if args.ingest:
        logger.info("Starting Ingestion Mode...")
        ingest_documents(args.ingest)
        return "Ingestion Complete."

    # We build the whole graph, definided inside the grap_builder module
    app = build_graph()

    # Allow user to print the graph: in this way, copying and pasting to
    # mermaid.live we can better visualize and understand the flow.
    if args.visualize:
        print("\nCopy to mermaid.live everthing contained between ###---###")
        print("###---###")
        print(app.get_graph().draw_mermaid())
        print("###---###\n")
        

    if not args.query:
        parser.print_help()
        sys.exit(0)
    
    # We start processing the query
    logger.info(f"Processing Query: {args.query}")
    inputs = {"question": args.query, "retry_count": 0}

    try:
        # Invoke the full graph application
        final_state = app.invoke(inputs)

        print("\n ---- FINAL ANSWER ---- ")
        print(final_state.get("generation"))
        print("\n ---------------------- ")
        
        if final_state.get("documents"):
            # If available and used, print the resources.
            print("Sources used:")
            for d in final_state["documents"]:
                src = d.metadata.get("source", "unknown")
                print(f"- {src}")

    except Exception as e:
        logger.error(f"Error executing graph: {e}")

if __name__ == "__main__":
    main()