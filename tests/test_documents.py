import base64
import io
import unittest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from core.documents import extract_document, to_langchain_documents


class PDFExtractionTests(unittest.TestCase):
    def payload(self, text=True, encrypted=False, pages_lines=None):
        pdf = PdfWriter()
        if pages_lines is None:
            pages_lines = [
                ['Section 1.1', 'Test fixture on page 1.'],
                ['Section 2.1', 'Test fixture on page 2.'],
            ]
        for lines in pages_lines:
            page = pdf.add_blank_page(612, 792)
            if text:
                font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
                page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): pdf._add_object(font)})})
                ops = ['BT /F1 12 Tf 72 720 Td']
                for idx, line in enumerate(lines):
                    safe = line.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
                    if idx == 0:
                        ops.append(f'({safe}) Tj')
                    else:
                        ops.append(f'0 -18 Td ({safe}) Tj')
                ops.append('ET')
                content = DecodedStreamObject()
                content.set_data(' '.join(ops).encode())
                page[NameObject('/Contents')] = pdf._add_object(content)
        if encrypted:
            pdf.encrypt('test-password')
        stream = io.BytesIO()
        pdf.write(stream)
        return {'name': 'test-fixture.pdf', 'data': base64.b64encode(stream.getvalue()).decode(), 'category': 'Insurance'}

    def test_page_and_clause_metadata(self):
        doc = extract_document(self.payload(), 20)
        self.assertEqual(doc['pages'], 2)
        self.assertEqual([c['page'] for c in doc['chunks']], [1, 2])
        self.assertEqual([c['clause'] for c in doc['chunks']], ['1.1', '2.1'])

    def test_clause_aware_segmentation_and_langchain_conversion(self):
        pages_lines = [
            [
                '2.1 Hospitalization Coverage',
                'Covers up to USD 500,000 per policy year for inpatient care.',
                '4.2 Exclusions',
                'Pre-existing conditions are excluded during the first 24 months.',
            ],
            [
                'Late payment penalty is 2 percent per month after the 15-day grace period.',
                'Clause 5.1 - Prepayment Penalty',
                'Early loan repayment carries a 1.5 percent fee on remaining principal.',
            ],
        ]
        doc = extract_document(self.payload(pages_lines=pages_lines), 20)
        clauses = [(c['page'], c['clause'], c['clause_title']) for c in doc['chunks']]
        self.assertEqual(
            clauses,
            [
                (1, '2.1', 'Hospitalization Coverage'),
                (1, '4.2', 'Exclusions'),
                (2, '4.2', 'Exclusions'),
                (2, '5.1', 'Prepayment Penalty'),
            ],
        )
        lc_docs = to_langchain_documents(doc)
        self.assertEqual(len(lc_docs), 4)
        self.assertEqual(lc_docs[1].metadata['clause'], '4.2')
        self.assertEqual(lc_docs[3].metadata['page'], 2)

    def test_reject_non_pdf(self):
        with self.assertRaises(ValueError):
            extract_document({'name': 'bad.pdf', 'data': base64.b64encode(b'not pdf').decode()}, 20)

    def test_reject_empty(self):
        with self.assertRaisesRegex(ValueError, 'OCR'):
            extract_document(self.payload(text=False), 20)

    def test_reject_encrypted(self):
        with self.assertRaisesRegex(ValueError, 'password'):
            extract_document(self.payload(encrypted=True), 20)

    def test_size_limit(self):
        with self.assertRaisesRegex(ValueError, 'smaller'):
            extract_document(self.payload(), 0)


if __name__ == '__main__':
    unittest.main()
