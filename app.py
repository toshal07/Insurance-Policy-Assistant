"""PolicyAI Streamlit host for the custom HTML workspace."""
import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from core.auth import authenticate_user, register_user
from core.documents import extract_document
from core.rag import answer, get_llm_config

load_dotenv()
st.set_page_config(page_title='PolicyAI · Your policies, understood.', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>[data-testid="stHeader"], [data-testid="stToolbar"]{display:none} .block-container{padding:0!important;max-width:100%!important} iframe{border:0} [data-testid="stAppViewContainer"]{background:#f7f8fa} .stMainBlockContainer{padding:0!important}</style>''', unsafe_allow_html=True)
workspace = components.declare_component('policyai_workspace', path=str(Path(__file__).parent / 'ui'))
for key, value in {'user': None, 'documents': [], 'messages': [], 'last_event': None, 'notice': None, 'revision': 0}.items():
    if key not in st.session_state:
        st.session_state[key] = value
state = st.session_state
public_docs = [{k: v for k, v in d.items() if k not in ('chunks', '_vectors', '_vectorstore')} for d in state.documents]
try:
    llm_info = get_llm_config()
except ValueError as exc:
    llm_info = {
        'provider': 'invalid',
        'provider_label': 'Configuration error',
        'model': str(exc),
        'connected': False,
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
    }

event = workspace(
    user=state.user,
    documents=public_docs,
    messages=state.messages,
    notice=state.notice,
    connected=llm_info['connected'],
    model=llm_info['model'],
    provider=llm_info['provider_label'],
    embedding_model=llm_info['embedding_model'],
    limit=int(os.getenv('MAX_FILE_SIZE_MB', '20')),
    revision=state.revision,
    default=None,
    key='workspace',
)
if event and event.get('id') != state.last_event:
    state.last_event = event['id']
    state.notice = None
    try:
        action = event.get('action')
        if action == 'signup':
            state.user = register_user(
                event.get('name', ''),
                event.get('email', ''),
                event.get('password', ''),
                event.get('confirm_password'),
            )
            state.documents = []
            state.messages = []
            state.notice = {'kind': 'success', 'text': f"Welcome to PolicyAI, {state.user['name']}. Upload a policy to get started."}
        elif action == 'login':
            state.user = authenticate_user(event.get('email', ''), event.get('password', ''))
            state.documents = []
            state.messages = []
            state.notice = {'kind': 'success', 'text': f"Welcome back, {state.user['name']}."}
        elif action == 'guest_login':
            state.user = {'id': 'guest', 'name': 'Guest Workspace', 'email': 'Session only', 'initials': 'GW', 'guest': True}
            state.documents = []
            state.messages = []
            state.notice = {'kind': 'success', 'text': 'Entered guest workspace. Documents stay in this session only.'}
        elif action == 'logout':
            state.user = None
            state.documents = []
            state.messages = []
            state.notice = {'kind': 'success', 'text': 'Signed out safely. Your session documents were cleared.'}
        elif not state.user:
            raise ValueError('Please log in to continue.')
        elif action == 'upload':
            for item in event.get('files', []):
                state.documents.append(extract_document(item, int(os.getenv('MAX_FILE_SIZE_MB', '20'))))
            state.notice = {'kind': 'success', 'text': 'Documents are indexed and ready. Ask your first question.'}
        elif action == 'delete':
            state.documents = [d for d in state.documents if d['id'] != event.get('document_id')]
            state.messages = []
            state.notice = {'kind': 'success', 'text': 'Document and conversation removed from this session.'}
        elif action == 'clear':
            state.messages = []
        elif action in ('ask', 'summary', 'compare'):
            ids = event.get('document_ids', [])
            docs = [d for d in state.documents if d['id'] in ids]
            question = str(event.get('question', '')).strip()[:4000]
            if not question:
                raise ValueError('Enter a question first.')
            response = answer(question, docs, state.messages, mode=action)
            state.messages.extend([{'role': 'user', 'text': question}, {'role': 'assistant', **response}])
        else:
            raise ValueError('Unknown action.')
    except ValueError as exc:
        state.notice = {'kind': 'error', 'text': str(exc)}
    except Exception as exc:
        state.notice = {'kind': 'error', 'text': f'Operation failed ({type(exc).__name__}). Check installed dependencies and model configuration, then retry.'}
    state.revision += 1
    st.rerun()
