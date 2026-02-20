from langchain_core.prompts import ChatPromptTemplate


def prompt_template(task: str) -> ChatPromptTemplate:
    """Function for applying templates. This should also work as a system message.

    Args:

        task (str): task of the given session

    Returns: ChatPromptTemplate

    """
    #assert task in ["rag", "..."]

    
    return ChatPromptTemplate.from_template(
        """Answer the question based on the context provided.
        If the context is irrelevant or empty, answer based on your own knowledge 
        but mention that the provided context was insufficient.

        Context: \n

        '''
        {context}
        '''

        \n\n\n
        Question: \n
        '''
        {question}
        '''
        """
    )
