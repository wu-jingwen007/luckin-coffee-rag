from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

from rag_app.llm.chat import get_chat_model
from rag_app.vectorstore.chromadb_store import VectorStoreManager
from rag_app.config.settings import SYSTEM_PROMPT


class RAGChain:
    def __init__(self, top_k: int = 4):
        self.top_k = top_k
        self.vector_manager = VectorStoreManager()
        self.vector_manager.load_existing()
        self.vector_store = self.vector_manager.get_vector_store()
        
        self.llm = get_chat_model()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ('system', SYSTEM_PROMPT),
            ('human', '{question}'),
        ])
        
        self.retriever = self.vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': self.top_k},
        )
        self.answer_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

    @staticmethod
    def _unsupported_numbers(answer, context, question):
        answer_without_list_numbers = re.sub(
            r'(?m)^\s*\d+[\.\)、]\s*',
            '',
            answer,
        )
        allowed_numbers = {
            number.replace(',', '')
            for number in re.findall(
                r'\d+(?:[.,]\d+)?(?:%|\+)?',
                f'{context}\n{question}',
            )
        }
        answer_numbers = re.findall(
            r'\d+(?:[.,]\d+)?(?:%|\+)?',
            answer_without_list_numbers,
        )
        unsupported = []
        for number in answer_numbers:
            normalized = number.replace(',', '')
            if normalized not in allowed_numbers and normalized not in unsupported:
                unsupported.append(normalized)
        return unsupported

    def _answer_from_docs(self, question, docs):
        context = self._format_docs(docs)
        answer = self.answer_chain.invoke({
            'context': context,
            'question': question,
        })
        unsupported = self._unsupported_numbers(answer, context, question)
        if unsupported:
            correction = (
                f'{question}\n\n'
                f'上一版答案包含参考资料和用户问题中都没有的数字：'
                f'{", ".join(unsupported)}。请重新回答并删除所有无依据数字、'
                f'单位换算和推导，只保留参考资料直接支持的内容。'
            )
            answer = self.answer_chain.invoke({
                'context': context,
                'question': correction,
            })
            if self._unsupported_numbers(answer, context, question):
                return '根据提供的参考资料，未找到能够支持完整回答的信息，因此无法回答。'
        return answer

    def invoke(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        return self._answer_from_docs(question, docs)
    
    def invoke_with_sources(self, question: str) -> dict:
        docs = self.retriever.invoke(question)
        sources = [
            {
                'content': doc.page_content[:300] + '...' if len(doc.page_content) > 300 else doc.page_content,
                'source': doc.metadata.get('source', 'unknown'),
            }
            for doc in docs
        ]
        
        answer = self._answer_from_docs(question, docs)
        return {'answer': answer, 'sources': sources}
