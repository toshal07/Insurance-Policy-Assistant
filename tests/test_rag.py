import os
import unittest
from unittest.mock import patch
from langchain_core.embeddings import Embeddings
from core.rag import _parse_rag_response, answer, get_llm_config, retrieve_evidence


class KeywordEmbeddings(Embeddings):
    """Deterministic unit-norm embeddings for fast FAISS unit testing."""

    _KEYWORDS = ('coverage', 'hospital', 'exclusion', 'pre-existing', 'penalty', 'prepayment', 'waiting')

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        vec = [1.0 if kw in lower else 0.0 for kw in self._KEYWORDS]
        vec.append(0.1)
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _sample_docs() -> list[dict]:
    return [
        {
            'id': 'doc-ins',
            'name': 'health-policy.pdf',
            'category': 'Insurance',
            'pages': 2,
            'size': 2048,
            'chunks': [
                {
                    'page': 1,
                    'clause': '2.1',
                    'clause_title': 'Hospital Coverage',
                    'chunk_index': 0,
                    'text': 'Inpatient hospital coverage is provided up to $250,000 per year.',
                },
                {
                    'page': 2,
                    'clause': '4.2',
                    'clause_title': 'Exclusions',
                    'chunk_index': 1,
                    'text': 'Pre-existing conditions have a 24-month waiting period.',
                },
            ],
        },
        {
            'id': 'doc-loan',
            'name': 'loan-agreement.pdf',
            'category': 'Loan',
            'pages': 2,
            'size': 2048,
            'chunks': [
                {
                    'page': 1,
                    'clause': '3.1',
                    'clause_title': 'Interest',
                    'chunk_index': 0,
                    'text': 'Annual interest rate is fixed at 8.5 percent.',
                },
                {
                    'page': 2,
                    'clause': '5.4',
                    'clause_title': 'Prepayment Penalty',
                    'chunk_index': 1,
                    'text': 'A prepayment penalty of 2% applies if repaid within 12 months.',
                },
            ],
        },
    ]


class RAGPipelineTests(unittest.TestCase):
    def test_llm_config_defaults_to_huggingface_local(self):
        clean_env = {
            'LLM_PROVIDER': 'auto',
            'HF_API_TOKEN': '',
            'HUGGINGFACEHUB_API_TOKEN': '',
            'GEMINI_API_KEY': '',
            'GEMINI_MODEL': '',
        }
        with patch.dict(os.environ, clean_env, clear=False):
            cfg = get_llm_config()
            self.assertEqual(cfg['provider'], 'huggingface_local')
            self.assertTrue(cfg['connected'])

    def test_llm_config_selects_huggingface_hub_and_gemini_and_rejects_invalid(self):
        with patch.dict(os.environ, {'LLM_PROVIDER': 'auto', 'HF_API_TOKEN': 'hf_test'}, clear=False):
            self.assertEqual(get_llm_config()['provider'], 'huggingface_hub')
        with patch.dict(
            os.environ,
            {'LLM_PROVIDER': 'auto', 'HF_API_TOKEN': '', 'HUGGINGFACEHUB_API_TOKEN': '', 'GEMINI_API_KEY': 'gk', 'GEMINI_MODEL': 'gemini-2.5-flash'},
            clear=False,
        ):
            self.assertEqual(get_llm_config()['provider'], 'gemini')
        with patch.dict(os.environ, {'LLM_PROVIDER': 'unknown_provider'}, clear=False):
            with self.assertRaises(ValueError):
                get_llm_config()

    def test_faiss_retrieval_global_ranking_and_compare_balance(self):
        docs = _sample_docs()
        with patch('core.rag.get_embeddings', return_value=KeywordEmbeddings()):
            hits = retrieve_evidence('What is the prepayment penalty?', docs, [], mode='ask')
            self.assertTrue(hits)
            self.assertEqual(hits[0]['document'], 'loan-agreement.pdf')
            self.assertEqual(hits[0]['clause'], '5.4')
            self.assertEqual(hits[0]['page'], 2)

            compare_hits = retrieve_evidence('Compare coverage and penalty', docs, [], mode='compare')
            hit_docs = {h['document_id'] for h in compare_hits}
            self.assertEqual(hit_docs, {'doc-ins', 'doc-loan'})

    def test_parse_rag_response_json_fenced_and_prose_citations(self):
        selected = [
            {'document': 'a.pdf', 'document_id': '1', 'page': 2, 'clause': '4.2', 'clause_title': 'Exclusions', 'text': 'Pre-existing conditions excluded.'},
            {'document': 'b.pdf', 'document_id': '2', 'page': 3, 'clause': '5.1', 'clause_title': 'Penalties', 'text': 'Prepayment penalty is 2%.'},
        ]

        parsed_json = _parse_rag_response('```json\n{"text": "Excluded [1].", "source_ids": [1, 99]}\n```', selected)
        self.assertEqual(parsed_json['text'], 'Excluded [1].')
        self.assertEqual([s['number'] for s in parsed_json['sources']], [1])

        parsed_prose = _parse_rag_response('Prepayment penalty is 2% according to [2].', selected)
        self.assertEqual([s['number'] for s in parsed_prose['sources']], [2])
        self.assertEqual(parsed_prose['sources'][0]['clause'], '5.1')

        parsed_unbracketed = _parse_rag_response('The prepayment penalty is 2%.', selected)
        self.assertTrue(parsed_unbracketed['sources'])
        self.assertIn('[2]', parsed_unbracketed['text'])

        parsed_missing = _parse_rag_response('Not specified in the selected documents.', selected)
        self.assertEqual(parsed_missing['sources'], [])

    def test_answer_end_to_end_with_mocked_llm(self):
        docs = _sample_docs()
        with patch('core.rag.get_embeddings', return_value=KeywordEmbeddings()), patch(
            'core.rag._invoke_llm',
            return_value='{"text": "A 2% prepayment penalty applies within 12 months [1].", "source_ids": [1]}',
        ):
            res = answer('Is there a prepayment penalty?', docs, [])
            self.assertIn('2% prepayment penalty', res['text'])
            self.assertEqual(len(res['sources']), 1)
            self.assertEqual(res['sources'][0]['page'], 2)
            self.assertEqual(res['sources'][0]['clause'], '5.4')

    def test_streamlit_app_smoke(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parent.parent / 'app.py'
        at = AppTest.from_file(str(app_path)).run(timeout=20)
        self.assertFalse(at.exception)


if __name__ == '__main__':
    unittest.main()
