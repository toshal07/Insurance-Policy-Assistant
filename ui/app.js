/* PolicyAI UI. Document answers originate from the Python LangChain/FAISS/RAG host. */
const paths = {
  grid: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
  upload: 'M12 16V3m-5 5 5-5 5 5M4 14v6h16v-6',
  chat: 'M21 11a8 8 0 0 1-8 8H7l-4 3V11a9 9 0 0 1 18 0Z M7 10h10M7 14h6',
  compare: 'M8 3v18m8-18v18M3 7h10m-3-3 3 3-3 3m1 7h10m-3-3 3 3-3 3',
  file: 'M5 3h9l5 5v13H5z M14 3v6h5M8 13h8M8 17h6',
  library: 'M4 4h4v16H4z M10 4h4v16h-4z M16 5l3-1 4 15-3 1z',
  settings: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8ZM12 2v3m0 14v3M2 12h3m14 0h3M5 5l2 2m10 10 2 2M5 19l2-2M17 7l2-2',
  shield: 'M12 2 3 6v6c0 5 9 10 9 10s9-5 9-10V6z M8 12l3 3 5-6',
  arrow: 'M5 12h14m-5-5 5 5-5 5',
  chevron: 'm9 5 7 7-7 7',
  down: 'm7 10 5 5 5-5',
  plus: 'M12 5v14M5 12h14',
  check: 'm5 12 4 4L19 6',
  close: 'm6 6 12 12M6 18 18 6',
  help: 'M9 8a3 3 0 0 1 6 0c0 2-3 2-3 5m0 3v1M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20',
  clock: 'M12 6v6l4 2M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20',
  lock: 'M6 10h12v11H6zM8 10V6a4 4 0 0 1 8 0v4',
  loan: 'M3 9h18L12 3zM5 10v8m5-8v8m4-8v8m5-8v8M2 21h20',
  spark: 'm12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z',
  search: 'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm5 12 6 6',
  trash: 'M3 6h18M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7',
  menu: 'M4 6h16M4 12h16M4 18h16',
  send: 'M12 19V5m-6 6 6-6 6 6',
  book: 'M12 5C8 2 3 3 3 3v16s5-1 9 2c4-3 9-2 9-2V3s-5-1-9 2Zm0 0v16',
  link: 'm10 14 4-4M8 16l-2 2a4 4 0 0 1-6-6l5-5a4 4 0 0 1 6 0m2 1 2-2a4 4 0 0 1 6 6l-5 5a4 4 0 0 1-6 0',
  user: 'M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 3a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z',
  mail: 'M4 4h16a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Zm0 3 8 5 8-5',
  logout: 'M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9',
  eye: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8Zm11-3a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z',
  eyeOff: 'M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24M1 1l22 22',
};

const icon = (name, size = 18) =>
  `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${paths[name] || paths.file}"/></svg>`;

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const navItems = [
  ['Dashboard', 'grid'],
  ['Upload Documents', 'upload'],
  ['Ask PolicyAI', 'chat'],
  ['Compare Policies', 'compare'],
  ['Document Summaries', 'file'],
  ['Document Library', 'library'],
];

let state = {
  user: null,
  documents: [],
  messages: [],
  connected: false,
  model: '',
  provider: 'Hugging Face (Local)',
  embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
  limit: 20,
  notice: null,
};

let page = 'Dashboard';
let host = false;
let busy = false;
let menu = false;
let selected = 'all';
let category = 'Insurance';
let draft = '';
let filter = '';
let search = '';
let motion = true;
let lastRevision = -1;
let pendingAction = '';
let focusBeforeModal = null;

let authMode = 'login';
let authDirection = '';
let showPass = false;
let authDraft = { name: '', email: '', password: '', confirm: '' };
let fieldErrors = {};
let shakeAuth = false;

let mountedView = 'none';
let renderedAuthMode = '';
let renderedAuthBusy = null;
let renderedPageSignature = '';
let lastMessageCount = 0;

const app = document.getElementById('app');
const post = (type, data = {}) => window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, '*');
post('streamlit:componentReady', { apiVersion: 1 });

function size() {
  post('streamlit:setFrameHeight', { height: Math.max(820, document.documentElement.scrollHeight) });
}

window.addEventListener('message', e => {
  if (e.source !== window.parent || e.data?.type !== 'streamlit:render') return;
  host = true;
  state = { ...state, ...e.data.args };
  if (state.revision !== lastRevision) {
    busy = false;
    lastRevision = state.revision;
    if (pendingAction === 'upload' && state.notice?.kind !== 'error') page = 'Document Library';
    if ((pendingAction === 'login' || pendingAction === 'signup' || pendingAction === 'guest_login') && state.user) {
      page = 'Dashboard';
      authDraft.password = '';
      authDraft.confirm = '';
      fieldErrors = {};
    }
    if ((pendingAction === 'login' || pendingAction === 'signup') && !state.user && state.notice?.kind === 'error') {
      shakeAuth = true;
    }
    if (pendingAction === 'logout' && !state.user) {
      page = 'Dashboard';
      selected = 'all';
      fieldErrors = {};
    }
    pendingAction = '';
    renderedPageSignature = '';
  }
  render();
});

function emit(action, data = {}) {
  if (!host) {
    toast('This is the design preview. Run the Streamlit app to process PDFs and run RAG queries.');
    return;
  }
  if (busy) return;
  busy = true;
  pendingAction = action;
  post('streamlit:setComponentValue', { value: { id: crypto.randomUUID(), action, ...data }, dataType: 'json' });
  render();
}

function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.classList.add('visible');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('visible'), 4200);
}

function go(next) {
  page = next;
  menu = false;
  render();
  window.scrollTo({ top: 0, behavior: motion ? 'smooth' : 'auto' });
}

function button(label, action, kind = 'primary', ico = 'plus') {
  return `<button class="button ${kind}" data-action="${action}">${icon(ico, 14)}${label}</button>`;
}

function heading(title, sub, action = '') {
  return `<div class="page-head"><div><div class="eyebrow">YOUR POLICY WORKSPACE</div><h1>${title}</h1><p class="subtitle">${sub}</p></div>${action}</div>`;
}

function noticeMarkup() {
  if (!state.notice) return '';
  return `<div class="notice ${esc(state.notice.kind)}" role="status"><span>${esc(state.notice.text)}</span><button type="button" data-action="dismiss" aria-label="Dismiss notification">${icon('close', 13)}</button></div>`;
}

function passwordStrength(pw) {
  let score = 0;
  if (pw.length >= 8) score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/\d/.test(pw)) score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  const labels = ['', 'Weak', 'Fair', 'Good', 'Strong'];
  return { score, pct: Math.max(score * 25, pw ? 15 : 0), label: pw ? labels[score] || 'Too short' : '' };
}

function authVisual() {
  return `
    <div class="auth-visual" aria-hidden="true">
      <div class="auth-pulse-ring"></div>
      <div class="auth-pulse-ring delayed"></div>
      <div class="auth-mini-doc">
        <div class="scan-beam"></div>
        <div class="mini-head">${icon('file', 13)}<span>POLICY.PDF</span></div>
        <div class="mini-line"></div>
        <div class="mini-line short"></div>
        <div class="mini-line accent"></div>
      </div>
      <div class="auth-mini-wire"></div>
      <div class="auth-mini-core">${icon('spark', 22)}</div>
      <div class="auth-mini-wire two"></div>
      <div class="auth-mini-cite">
        <span class="cite-dot"></span>
        <div>
          <strong>§ 4.2 · Page 3</strong>
          <small>Verified clause</small>
        </div>
      </div>
    </div>`;
}

function authFormPanelMarkup() {
  const isSignup = authMode === 'signup';
  const strength = passwordStrength(authDraft.password);
  const animClass = shakeAuth ? 'shake' : authDirection === 'right' ? 'from-right' : authDirection === 'left' ? 'from-left' : '';
  const matchState = isSignup && authDraft.confirm
    ? authDraft.password === authDraft.confirm
      ? '<span class="field-hint">Passwords match</span>'
      : '<span class="field-hint mismatch">Mismatch</span>'
    : '';

  return `
    <div class="auth-form-wrap ${animClass}">
      <h2>${isSignup ? 'Create your workspace account' : 'Welcome back'}</h2>
      <p class="auth-sub">${isSignup ? 'Sign up to analyze insurance policies and loan fine print with cited evidence.' : 'Log in to access your PolicyAI document workspace.'}</p>

      <form id="auth-form" class="auth-form" novalidate>
        ${isSignup ? `
          <div class="field ${fieldErrors.name ? 'has-error' : ''}">
            <label for="auth-name">Full name</label>
            <div class="input-wrap">
              ${icon('user', 15)}
              <input id="auth-name" type="text" autocomplete="name" placeholder="Alex Morgan" value="${esc(authDraft.name)}" required>
            </div>
            ${fieldErrors.name ? `<p class="field-error">${esc(fieldErrors.name)}</p>` : ''}
          </div>` : ''}

        <div class="field ${fieldErrors.email ? 'has-error' : ''}">
          <label for="auth-email">Email address</label>
          <div class="input-wrap">
            ${icon('mail', 15)}
            <input id="auth-email" type="email" autocomplete="email" placeholder="alex@company.com" value="${esc(authDraft.email)}" required>
          </div>
          ${fieldErrors.email ? `<p class="field-error">${esc(fieldErrors.email)}</p>` : ''}
        </div>

        <div class="field ${fieldErrors.password ? 'has-error' : ''}">
          <label for="auth-password">Password</label>
          <div class="input-wrap">
            ${icon('lock', 15)}
            <input id="auth-password" type="${showPass ? 'text' : 'password'}" autocomplete="${isSignup ? 'new-password' : 'current-password'}" placeholder="${isSignup ? 'At least 8 characters' : 'Enter your password'}" value="${esc(authDraft.password)}" required>
            <button type="button" class="eye-btn" data-action="toggle-pass" aria-label="${showPass ? 'Hide password' : 'Show password'}">${icon(showPass ? 'eyeOff' : 'eye', 15)}</button>
          </div>
          ${fieldErrors.password ? `<p class="field-error">${esc(fieldErrors.password)}</p>` : ''}
          ${isSignup ? `<div class="strength-row" ${authDraft.password ? '' : 'hidden'}><div class="strength-meter"><span class="s-${strength.score}" style="width:${strength.pct}%"></span></div><small>${strength.label}</small></div>` : ''}
        </div>

        ${isSignup ? `
          <div class="field ${fieldErrors.confirm ? 'has-error' : ''}">
            <label for="auth-confirm"><span>Confirm password</span><span id="confirm-match-hint">${matchState}</span></label>
            <div class="input-wrap">
              ${icon('lock', 15)}
              <input id="auth-confirm" type="${showPass ? 'text' : 'password'}" autocomplete="new-password" placeholder="Re-enter your password" value="${esc(authDraft.confirm)}" required>
            </div>
            ${fieldErrors.confirm ? `<p class="field-error">${esc(fieldErrors.confirm)}</p>` : ''}
          </div>` : ''}

        <button id="auth-submit-btn" type="submit" class="button primary auth-submit" ${busy ? 'disabled' : ''}>
          ${busy ? '<span class="spinner white"></span>Signing you in…' : `${isSignup ? 'Create account' : 'Log in to PolicyAI'} ${icon('arrow', 14)}`}
        </button>
      </form>

      <div class="auth-divider"><span>or</span></div>

      <button id="auth-guest-btn" type="button" class="button auth-guest" data-action="guest-login" ${busy ? 'disabled' : ''}>
        ${icon('shield', 14)} Continue in guest session
      </button>

      <p class="auth-switch">
        ${isSignup ? 'Already have an account?' : 'New to PolicyAI?'}
        <button type="button" class="text-button inline" data-auth-tab="${isSignup ? 'login' : 'signup'}">${isSignup ? 'Log in' : 'Create an account'}</button>
      </p>
    </div>`;
}

function authShellMarkup() {
  const isSignup = authMode === 'signup';
  return `
    <div class="auth-shell">
      <div class="auth-grid-bg" aria-hidden="true"></div>
      <div class="auth-orb one" aria-hidden="true"></div>
      <div class="auth-orb two" aria-hidden="true"></div>
      <div class="auth-orb three" aria-hidden="true"></div>
      <div class="auth-frame">
        <aside class="auth-showcase">
          <div class="auth-brand">
            <span class="brand-mark">${icon('shield', 32)}</span>
            <span>Policy<span class="ai">AI</span></span>
          </div>
          <div class="auth-showcase-body">
            <div class="auth-badge">${icon('spark', 12)} INTELLIGENT POLICY ASSISTANT</div>
            <h1>Understand every clause before you sign or claim.</h1>
            <p>Upload insurance policies or loan agreements and ask plain-English questions backed by exact page and clause citations.</p>
            ${authVisual()}
            <ul class="auth-features">
              <li style="--i:0"><span>${icon('check', 12)}</span>Clause-aware PDF chunking &amp; page tracking</li>
              <li style="--i:1"><span>${icon('check', 12)}</span>Sentence-Transformers + LangChain FAISS search</li>
              <li style="--i:2"><span>${icon('check', 12)}</span>Hugging Face &amp; open-source RAG answers</li>
            </ul>
          </div>
          <div class="auth-showcase-foot">${icon('lock', 12)} Local session isolation · PBKDF2-HMAC account security</div>
        </aside>

        <section class="auth-card" aria-label="Authentication">
          <div class="auth-card-top">
            <div class="auth-tabs ${isSignup ? 'is-signup' : ''}" role="tablist" aria-label="Authentication mode">
              <span class="auth-tab-pill" aria-hidden="true"></span>
              <button type="button" role="tab" aria-selected="${!isSignup}" class="auth-tab ${!isSignup ? 'active' : ''}" data-auth-tab="login">Log In</button>
              <button type="button" role="tab" aria-selected="${isSignup}" class="auth-tab ${isSignup ? 'active' : ''}" data-auth-tab="signup">Sign Up</button>
            </div>
            <button type="button" id="auth-motion-btn" class="icon-button" data-action="motion" title="${motion ? 'Disable animations' : 'Enable animations'}" aria-label="Toggle motion" aria-pressed="${motion}">${icon('spark', 16)}</button>
          </div>
          <div id="auth-notice-slot">${noticeMarkup()}</div>
          <div id="auth-panel">${authFormPanelMarkup()}</div>
        </section>
      </div>
    </div>`;
}

function flow() {
  return `
    <div class="flow-art" role="img" aria-label="Documents flow through PolicyAI into answers supported by sources">
      <div class="orbit"></div>
      <div class="orbit two"></div>
      <svg class="flow-wire" viewBox="0 0 420 230" preserveAspectRatio="none">
        <path id="flow-path-1" d="M80 110C130 110 120 115 190 115"/>
        <path id="flow-path-2" d="M230 115C280 115 280 100 340 100"/>
        ${motion ? '<circle r="3.5" class="flow-packet"><animateMotion dur="2.4s" repeatCount="indefinite"><mpath href="#flow-path-1"/></animateMotion></circle><circle r="3.5" class="flow-packet two"><animateMotion dur="2.4s" begin="1.1s" repeatCount="indefinite"><mpath href="#flow-path-2"/></animateMotion></circle>' : ''}
      </svg>
      <div class="flow-doc">
        <div class="scan-beam"></div>
        ${icon('file', 20)}
        <div class="tiny">YOUR POLICY</div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line"></div>
      </div>
      <div class="flow-core">${icon('spark', 35)}</div>
      <div class="flow-answer">
        <div class="answer-top">${icon('check', 11)}GROUNDED IN YOUR POLICY</div>
        <div class="line"></div>
        <div class="line"></div>
        <div class="line" style="width:65%"></div>
        <span class="source-tag">${icon('link', 8)} Linked to the source</span>
      </div>
      <div class="flow-caption">DOCUMENTS<span>→</span>UNDERSTANDING<span>→</span>CLARITY</div>
    </div>`;
}

function uploadZone(large = false) {
  return `<button class="upload-zone ${large ? 'large-upload' : ''}" data-action="browse" ${busy ? 'disabled' : ''}><span class="upload-icon">${icon('upload', large ? 25 : 20)}</span><strong>${busy && pendingAction === 'upload' ? 'Reading and indexing your documents…' : 'Drop your policy documents here, or <span>browse files</span>'}</strong><small>PDF files only · Up to ${state.limit} MB per file${large ? '<br>Insurance policies, loan agreements, and the fine print.' : ''}</small></button>`;
}

function docRows(docs, actions = false) {
  return docs
    .map(
      (d, idx) =>
        `<div class="doc-row" style="--i:${idx}"><div class="doc-icon">${icon('file', 19)}</div><div class="doc-meta"><strong title="${esc(d.name)}">${esc(d.name)}</strong><small>${esc(d.category)} <span>·</span> ${d.pages} pages <span>·</span> ${(d.size / 1024 / 1024).toFixed(2)} MB</small></div><span class="pill">${icon('check', 9)}Ready</span><button class="icon-button" data-doc="${esc(d.id)}" data-action="${actions ? 'remove' : 'open-doc'}" aria-label="${actions ? 'Remove' : 'Ask about'} ${esc(d.name)}">${icon(actions ? 'trash' : 'arrow', 15)}</button></div>`
    )
    .join('');
}

function dashboard() {
  const count = state.documents.length;
  const questions = state.messages.filter(m => m.role === 'user').length;
  const step1Class = count > 0 ? 'is-done' : 'is-current';
  const step2Class = questions > 0 ? 'is-done' : count > 0 ? 'is-current' : '';
  const step3Class = questions > 0 ? 'is-current' : '';

  return (
    heading(
      'A little clarity. A lot of confidence.',
      'Your policies, your questions. Every answer backed by the fine print.',
      button('Upload document', 'upload-page')
    ) +
    `<section class="hero"><div class="hero-copy"><div class="hero-tag">${icon('spark', 12)} LESS FINE PRINT. MORE PEACE OF MIND.</div><h2>Know what’s covered.<br><span>Understand what matters.</span></h2><p>Turn complex insurance policies and loan agreements into clear answers, with sources you can verify.</p><div class="hero-actions">${button('Ask your first question', 'chat-page', 'dark', 'chat')}<button class="text-button" data-action="how">How it works ${icon('arrow', 13)}</button></div></div>${flow()}</section>` +
    `<section class="stats" aria-label="Workspace overview">${[
      [count, 'Total documents', 'file', 'Your knowledge, in one place'],
      [state.documents.filter(d => d.category === 'Insurance').length, 'Insurance policies', 'shield', 'Understand your protection'],
      [state.documents.filter(d => d.category === 'Loan').length, 'Loan agreements', 'loan', 'Know every commitment'],
      [questions, 'Questions asked', 'chat', 'Clarity, one question at a time'],
    ]
      .map(
        ([n, label, ico, foot], i) =>
          `<div class="stat" style="--i:${i}"><div class="stat-top">${label}<span class="stat-icon">${icon(ico, 16)}</span></div><div class="stat-value" data-count="${n}">${String(n).padStart(2, '0')}</div><div class="stat-foot">${foot}</div></div>`
      )
      .join('')}</section>` +
    `<div class="grid"><section class="panel"><div class="panel-heading"><div><h2>Your documents</h2><p>A single home for all the important details.</p></div><button class="text-button" data-page="Document Library">View all ${icon('arrow', 12)}</button></div>${uploadZone()}<div class="document-list">${count ? docRows(state.documents.slice(-2).reverse()) : `<div class="empty-line">${icon('file', 23)}<div><strong>Your next clear answer starts here.</strong><br>Upload a policy to build your personal document library.</div></div>`}</div></section><section class="panel ask-panel"><div class="panel-heading"><div><h2>A good question changes everything.</h2><p>Start with something you’d like to understand.</p></div></div><div class="question-list">${[
      ['shield', 'What does my policy actually cover?'],
      ['clock', 'Are there any waiting periods?'],
      ['loan', 'Can I repay my loan early?'],
    ]
      .map(([ico, q]) => `<button class="question-card" data-question="${esc(q)}"><span class="q-icon">${icon(ico, 15)}</span>${q}<span class="arrow">${icon('arrow', 13)}</span></button>`)
      .join('')}</div><div class="panel-foot">${icon('link', 11)}Answers grounded in your documents. Always.</div></section></div>` +
    `<section class="bottom-strip"><div class="bottom-intro"><div class="icon-wrap">${icon('book', 18)}</div><div><strong>From fine print to a clearer picture.</strong><p>Three simple steps. No insurance dictionary needed.</p></div></div><div class="steps"><div class="step ${step1Class}"><b>1</b>Upload a document</div>${icon('chevron', 10)}<div class="step ${step2Class}"><b>2</b>Ask a question</div>${icon('chevron', 10)}<div class="step ${step3Class}"><b>3</b>Explore the sources</div></div></section>`
  );
}

function uploadPage() {
  return (
    heading('Bring your policies into focus.', 'Upload your documents. We’ll help you find your way through them.') +
    `<section class="panel"><div class="panel-heading"><h2>Add to your document library</h2><span class="pill">${icon('lock', 10)} Session storage</span></div>${uploadZone(true)}<div class="form-row"><label for="category">Document type</label><select id="category"><option ${category === 'Insurance' ? 'selected' : ''}>Insurance</option><option ${category === 'Loan' ? 'selected' : ''}>Loan</option></select></div></section>` +
    `<div class="helper-grid">${[
      ['file', 'The original page matters.', 'Text is extracted page by page and segmented by clause, so every source points back to its exact place in your PDF.'],
      ['search', 'Built for the fine print.', 'Ask about coverage, exclusions, waiting periods, repayments, and penalties.'],
      ['shield', 'Local FAISS & open-source RAG.', 'Chunks are embedded with Sentence-Transformers and indexed with LangChain FAISS in your session.'],
    ]
      .map(([i, t, p], idx) => `<div class="helper-card" style="--i:${idx}">${icon(i, 22)}<h3>${t}</h3><p>${p}</p></div>`)
      .join('')}</div><p class="subtitle" style="margin-top:20px">Use text-based, unlocked PDFs. Scanned documents need OCR before uploading.</p>`
  );
}

function docOptions(includeAll = true) {
  return `${includeAll ? '<option value="all">All documents</option>' : '<option value="">Choose a document</option>'}${state.documents.map(d => `<option value="${d.id}" ${selected === d.id ? 'selected' : ''}>${esc(d.name)}</option>`).join('')}`;
}

function sourceButtons(sources, messageIndex) {
  return `<div class="citations">${sources.map(s => `<button class="citation" data-source="${s.number}" data-message="${messageIndex}" data-document="${esc(s.document_id)}">[${s.number}] ${esc(s.document)} · p. ${s.page}${s.clause ? ' · § ' + esc(s.clause) : ''} ${icon('arrow', 10)}</button>`).join('')}</div>`;
}

function chatPage() {
  const lastIndex = state.messages.findLastIndex(m => m.role === 'assistant');
  const last = state.messages[lastIndex];
  return (
    heading('The answers are in your documents.', 'Let’s find them together. Ask a question in your own words.') +
    `<div class="chat-layout"><section class="panel chat-panel"><div class="chat-toolbar"><select id="document-select" aria-label="Select documents for your question">${docOptions()}</select><button class="text-button" data-action="clear">${icon('plus', 13)} New conversation</button></div><div class="chat-messages" id="messages">${
      state.messages.length
        ? state.messages
            .map(
              (m, i) =>
                `<div class="message ${m.role}">${m.role === 'assistant' ? `<div class="message-label">${icon('spark', 15)}PolicyAI</div>` : ''}${esc(m.text)}${m.sources ? sourceButtons(m.sources, i) : ''}</div>`
            )
            .join('')
        : `<div class="chat-empty"><div class="flow-core">${icon('spark', 33)}</div><h2>Make sense of the fine print.</h2><p>${state.documents.length ? 'Ask anything about your selected documents. I’ll find the relevant passages and link you to the evidence.' : 'Add your first document to start a conversation grounded in your policy.'}</p>${state.documents.length ? `<div class="chat-suggestions"><button data-question="What is covered by this policy?">What’s covered?</button><button data-question="What are the exclusions?">What’s excluded?</button></div>` : button('Upload a document', 'upload-page', 'primary', 'upload')}</div>`
    }${busy ? `<div class="busy-card"><div class="busy"><span class="spinner"></span><span>Retrieving relevant passages via FAISS and preparing a cited answer</span><span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span></div><div class="busy-shimmer" aria-hidden="true"><span></span><span></span><span></span></div></div>` : ''}</div><form class="composer" id="chat-form"><div class="composer-box"><textarea id="question" aria-label="Your policy question" placeholder="Ask about coverage, exclusions, or the fine print…" maxlength="4000">${esc(draft)}</textarea><button class="send" type="submit" aria-label="Send question" ${busy ? 'disabled' : ''}>${icon('send', 18)}</button></div><div class="composer-hint">Document-based information. Always check the cited wording.</div></form></section><aside class="panel evidence-panel"><div class="panel-heading"><h2>Supporting sources</h2>${icon('link', 15)}</div>${
      last?.sources?.length
        ? last.sources
            .map(
              (s, idx) =>
                `<button class="source-entry" style="--i:${idx}" data-source="${s.number}" data-message="${lastIndex}" data-document="${esc(s.document_id)}"><strong>[${s.number}] ${esc(s.document)}</strong><small>Page ${s.page}${s.clause ? ' · Clause ' + esc(s.clause) + (s.clause_title ? ' (' + esc(s.clause_title) + ')' : '') : ''}</small><p>${esc(s.text.slice(0, 160))}…</p></button>`
            )
            .join('')
        : `<div class="evidence-empty">${icon('book', 34)}<strong>Evidence, not guesswork.</strong><p>The passages supporting your answer will appear here. Open any citation to inspect the original extracted text.</p></div>`
    }</aside></div>`
  );
}

function libraryPage() {
  return (
    heading('Your document library.', 'All your policies and agreements, ready to be understood.', button('Upload document', 'upload-page')) +
    `<section class="panel"><div class="library-tools"><input type="search" id="search" placeholder="Search documents…" aria-label="Search documents" value="${esc(search)}"><select id="filter" aria-label="Filter document type"><option value="">All types</option><option ${filter === 'Insurance' ? 'selected' : ''}>Insurance</option><option ${filter === 'Loan' ? 'selected' : ''}>Loan</option></select></div><div id="library-results">${libraryResults()}</div></section>`
  );
}

function libraryResults() {
  const docs = state.documents.filter(d => (!filter || d.category === filter) && d.name.toLowerCase().includes(search.toLowerCase()));
  return docs.length
    ? `<div class="library-rows">${docRows(docs, true)}</div>`
    : `<div class="library-empty">${icon('library', 40)}<h2>${state.documents.length ? 'No matching documents.' : 'A little organization. A lot more clarity.'}</h2><p>${state.documents.length ? 'Try another name or document type.' : 'Your uploaded policies and agreements will live here.'}</p>${state.documents.length ? '' : button('Add your first document', 'upload-page', 'primary', 'upload')}</div>`;
}

function comparePage() {
  return (
    heading('See the differences that matter.', 'Compare two documents, with evidence from each side.') +
    `<section class="panel"><div class="panel-heading"><h2>Choose your documents</h2><span class="eyebrow">SIDE BY SIDE</span></div><div class="selection-grid"><div class="selection-card"><label for="compare-a">DOCUMENT 01</label>${icon('file', 28)}<select id="compare-a">${docOptions(false)}</select></div><div class="selection-vs" aria-hidden="true">VS</div><div class="selection-card"><label for="compare-b">DOCUMENT 02</label>${icon('file', 28)}<select id="compare-b">${docOptions(false)}</select></div></div><div class="comparison-action">${button('Compare documents', 'compare', 'primary', 'compare')}<p>Missing details are marked as not specified, never assumed.</p></div></section>` +
    `<div class="helper-grid">${[
      ['shield', 'Coverage & exclusions', 'Understand where protection begins and ends.'],
      ['loan', 'Costs & commitments', 'Explore premiums, interest rates, fees, and penalties.'],
      ['clock', 'Timelines & conditions', 'Review waiting periods, renewal terms, and repayment conditions.'],
    ]
      .map(([i, t, p], idx) => `<div class="helper-card" style="--i:${idx}">${icon(i, 22)}<h3>${t}</h3><p>${p}</p></div>`)
      .join('')}</div>`
  );
}

function summaryPage() {
  return (
    heading('The big picture. With the details.', 'Get a document-grounded overview of what matters.') +
    `<section class="panel"><div class="panel-heading"><h2>Choose a document to summarize</h2>${icon('file', 18)}</div><div class="form-row"><select id="summary-document" aria-label="Document to summarize" style="flex:1">${docOptions(false)}</select>${button('Create summary', 'summary', 'primary', 'spark')}</div><div class="panel-foot">${icon('link', 12)} Figures and conditions link back to supporting passages.</div></section>` +
    `<div class="helper-grid"><div class="helper-card" style="--i:0">${icon('shield', 23)}<h3>Insurance, made readable.</h3><p>Coverage, premiums, exclusions, waiting periods, claims, and renewal conditions.</p></div><div class="helper-card" style="--i:1">${icon('loan', 23)}<h3>Your obligations, in focus.</h3><p>Loan amount, rates, tenure, repayments, processing fees, and default conditions.</p></div><div class="helper-card" style="--i:2">${icon('book', 23)}<h3>Keep the source close.</h3><p>Summaries appear in your conversation, with clickable evidence cards for further reading.</p></div></div>`
  );
}

function settingsPage() {
  return (
    heading('Your workspace, your way.', 'A few details about how PolicyAI works.') +
    `<section class="panel"><div class="panel-heading"><h2>Workspace settings</h2></div><div class="settings-list">` +
    `<div class="setting"><div><strong>Signed-in account</strong><p>${esc(state.user?.name)} (${esc(state.user?.email)})</p></div><button class="button small" data-action="logout">${icon('logout', 13)}Log out</button></div>` +
    `<div class="setting"><div><strong>LLM backend (${esc(state.provider || 'Hugging Face (Local)')})</strong><p>${state.connected ? 'Configured via .env / local Hugging Face runtime. Switch LLM_PROVIDER between huggingface_local, huggingface_hub, ollama, or gemini.' : 'Configure your LLM provider in the server .env file.'}</p></div><code>${state.connected ? esc(state.model) : 'Not configured'}</code></div>` +
    `<div class="setting"><div><strong>Embeddings &amp; retrieval</strong><p>Clause-aware LangChain chunking with Sentence-Transformers dense embeddings and FAISS similarity search.</p></div><code>${esc(state.embedding_model || 'sentence-transformers/all-MiniLM-L6-v2')}</code></div>` +
    `<div class="setting"><div><strong>Document storage</strong><p>Documents and conversations are held in the current Streamlit session. Refreshing or signing out clears them.</p></div><span class="pill">Session only</span></div>` +
    `<div class="setting"><div><strong>Interface motion</strong><p>Smooth transitions and animated visualizers help show how documents become answers. Reduced-motion preferences are respected.</p></div><button class="switch-btn" data-action="motion" aria-pressed="${motion}"><span class="switch-track"><span class="switch-thumb"></span></span><span>${motion ? 'Motion on' : 'Motion off'}</span></button></div>` +
    `<div class="setting"><div><strong>Upload limit</strong><p>Change MAX_FILE_SIZE_MB in your server configuration.</p></div><code>${state.limit} MB / PDF</code></div>` +
    `</div></section>`
  );
}

function workspaceShellMarkup() {
  const initials = esc(state.user.initials || 'PW');
  const userName = esc(state.user.name || 'Personal workspace');
  const userEmail = esc(state.user.email || 'PolicyAI · Local session');

  return `<div class="shell ${menu ? 'menu-open' : ''}"><aside class="sidebar"><div class="brand"><span class="brand-mark">${icon('shield', 32)}</span><span>Policy<span class="ai">AI</span></span></div><div class="workspace-label"><b>${initials}</b><span class="ws-name">${userName}</span>${icon('lock', 12)}</div><p class="nav-label">WORKSPACE</p><nav class="nav" aria-label="Main navigation">${navItems
    .map(
      ([name, ico]) =>
        `<button class="nav-button ${page === name ? 'active' : ''}" data-page="${name}" ${page === name ? 'aria-current="page"' : ''}>${icon(ico, 17)}${name}${name === 'Document Library' && state.documents.length ? `<span class="badge">${state.documents.length}</span>` : ''}</button>`
    )
    .join('')}</nav><div class="sidebar-bottom"><div class="privacy-card">${icon('shield', 19)}<strong>Clarity you can trace.</strong><p>Answers linked to your documents,<br>with evidence one click away.</p></div><button class="nav-button ${page === 'Settings' ? 'active' : ''}" data-page="Settings">${icon('settings', 17)}Settings</button><button class="nav-button" data-action="help">${icon('help', 17)}Help &amp; getting started</button><div class="profile"><div class="avatar">${initials}</div><div class="profile-meta"><strong>${userName}</strong><span>${userEmail}</span></div><button class="icon-button" data-action="logout" title="Log out" aria-label="Log out">${icon('logout', 15)}</button></div></div></aside><main class="main"><header class="topbar"><div class="breadcrumb"><button class="icon-button mobile-menu" data-action="menu" aria-label="Toggle navigation">${icon('menu', 18)}</button><span>Workspace</span>${icon('chevron', 10)}<strong id="topbar-page">${page}</strong></div><div class="top-actions">${host ? `<div class="status"><i></i><span id="topbar-status">${state.connected ? esc(state.provider || 'RAG ready') : 'Workspace ready'}</span></div>` : '<div class="preview-note">Interactive design preview</div>'}<span class="divider"></span><button class="icon-button" data-action="help" aria-label="Help">${icon('help', 18)}</button><button class="text-button top-logout" data-action="logout">${icon('logout', 13)}Log out</button><div class="avatar" style="width:29px;height:29px;font-size:9px">${initials}</div></div></header><div class="content"><div id="notice-slot">${noticeMarkup()}</div><div id="page-slot"></div><footer class="footer"><span>${icon('shield', 10)}Built for understanding. Grounded in your documents.</span><span>PolicyAI &nbsp; / &nbsp; A clearer perspective.</span></footer></div></main></div><input type="file" id="file-input" accept=".pdf,application/pdf" multiple hidden>`;
}

function renderAuth() {
  if (mountedView !== 'auth') {
    app.innerHTML = authShellMarkup();
    mountedView = 'auth';
    renderedAuthMode = authMode;
    renderedAuthBusy = busy;
    bindAuthPanel();
    requestAnimationFrame(size);
    return;
  }

  const tabs = app.querySelector('.auth-tabs');
  if (tabs) {
    tabs.classList.toggle('is-signup', authMode === 'signup');
    tabs.querySelectorAll('.auth-tab').forEach(btn => {
      const active = btn.dataset.authTab === authMode;
      btn.classList.toggle('active', active);
      btn.setAttribute('aria-selected', String(active));
    });
  }

  const motionBtn = document.getElementById('auth-motion-btn');
  if (motionBtn) {
    motionBtn.title = motion ? 'Disable animations' : 'Enable animations';
    motionBtn.setAttribute('aria-pressed', String(motion));
  }

  const noticeSlot = document.getElementById('auth-notice-slot');
  if (noticeSlot) noticeSlot.innerHTML = noticeMarkup();

  if (renderedAuthMode !== authMode || renderedAuthBusy !== busy || shakeAuth) {
    const panel = document.getElementById('auth-panel');
    if (panel) {
      panel.innerHTML = authFormPanelMarkup();
      bindAuthPanel();
    }
    renderedAuthMode = authMode;
    renderedAuthBusy = busy;
    authDirection = '';
    shakeAuth = false;
  }
  requestAnimationFrame(size);
}

function animateStats() {
  if (!motion) return;
  document.querySelectorAll('.stat-value[data-count]').forEach(el => {
    const target = Number(el.dataset.count || 0);
    if (!target) return;
    const start = performance.now();
    const duration = 420;
    const step = now => {
      const t = Math.min(1, (now - start) / duration);
      const val = Math.round(target * (1 - Math.pow(1 - t, 3)));
      el.textContent = String(val).padStart(2, '0');
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  });
}

function renderWorkspace() {
  const views = {
    Dashboard: dashboard,
    'Upload Documents': uploadPage,
    'Ask PolicyAI': chatPage,
    'Compare Policies': comparePage,
    'Document Summaries': summaryPage,
    'Document Library': libraryPage,
    Settings: settingsPage,
  };

  if (mountedView !== 'workspace') {
    app.innerHTML = workspaceShellMarkup();
    mountedView = 'workspace';
    renderedPageSignature = '';
    document.getElementById('file-input')?.addEventListener('change', e => upload(e.target.files));
  }

  const shell = app.querySelector('.shell');
  if (shell) shell.classList.toggle('menu-open', menu);

  app.querySelectorAll('.nav-button[data-page]').forEach(btn => {
    const active = btn.dataset.page === page;
    btn.classList.toggle('active', active);
    if (active) btn.setAttribute('aria-current', 'page');
    else btn.removeAttribute('aria-current');
    if (btn.dataset.page === 'Document Library') {
      let badge = btn.querySelector('.badge');
      if (state.documents.length) {
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'badge';
          btn.appendChild(badge);
        }
        badge.textContent = String(state.documents.length);
      } else if (badge) {
        badge.remove();
      }
    }
  });

  const topbarPage = document.getElementById('topbar-page');
  if (topbarPage) topbarPage.textContent = page;

  const topbarStatus = document.getElementById('topbar-status');
  if (topbarStatus) topbarStatus.textContent = state.connected ? state.provider || 'RAG ready' : 'Workspace ready';

  const noticeSlot = document.getElementById('notice-slot');
  if (noticeSlot) noticeSlot.innerHTML = noticeMarkup();

  const sig = `${page}|${busy}|${pendingAction}|${state.revision}|${state.documents.length}|${state.messages.length}|${motion}|${state.connected}|${state.model}|${state.provider}`;
  if (sig !== renderedPageSignature) {
    const pageSlot = document.getElementById('page-slot');
    if (pageSlot) {
      pageSlot.innerHTML = `<div class="page">${views[page]()}</div>`;
      bindPage();
      if (page === 'Dashboard') animateStats();
      if (page === 'Ask PolicyAI') {
        const msgBox = document.getElementById('messages');
        if (msgBox && (state.messages.length !== lastMessageCount || busy)) {
          msgBox.scrollTop = msgBox.scrollHeight;
        }
        lastMessageCount = state.messages.length;
      }
    }
    renderedPageSignature = sig;
  }

  requestAnimationFrame(size);
}

function render() {
  document.body.classList.toggle('motion-off', !motion);
  if (!state.user) {
    renderAuth();
    return;
  }
  renderWorkspace();
}

function clearFieldError(field) {
  if (!fieldErrors[field]) return;
  delete fieldErrors[field];
  const el = document.getElementById(`auth-${field}`)?.closest('.field');
  if (el) {
    el.classList.remove('has-error');
    el.querySelector('.field-error')?.remove();
  }
}

function updatePasswordStrengthUI() {
  if (authMode !== 'signup') return;
  const field = document.getElementById('auth-password')?.closest('.field');
  const row = field?.querySelector('.strength-row');
  if (!row) return;
  if (!authDraft.password) {
    row.hidden = true;
    return;
  }
  const s = passwordStrength(authDraft.password);
  row.hidden = false;
  row.innerHTML = `<div class="strength-meter"><span class="s-${s.score}" style="width:${s.pct}%"></span></div><small>${s.label}</small>`;
}

function updateConfirmMatchUI() {
  const hint = document.getElementById('confirm-match-hint');
  if (!hint) return;
  if (!authDraft.confirm) {
    hint.innerHTML = '';
    return;
  }
  hint.innerHTML = authDraft.password === authDraft.confirm
    ? '<span class="field-hint">Passwords match</span>'
    : '<span class="field-hint mismatch">Mismatch</span>';
}

function bindAuthPanel() {
  document.getElementById('auth-name')?.addEventListener('input', e => {
    authDraft.name = e.target.value;
    clearFieldError('name');
  });
  document.getElementById('auth-email')?.addEventListener('input', e => {
    authDraft.email = e.target.value;
    clearFieldError('email');
  });
  document.getElementById('auth-password')?.addEventListener('input', e => {
    authDraft.password = e.target.value;
    clearFieldError('password');
    updatePasswordStrengthUI();
    updateConfirmMatchUI();
  });
  document.getElementById('auth-confirm')?.addEventListener('input', e => {
    authDraft.confirm = e.target.value;
    clearFieldError('confirm');
    updateConfirmMatchUI();
  });
  document.getElementById('auth-form')?.addEventListener('submit', e => {
    e.preventDefault();
    submitAuth();
  });
}

function submitAuth() {
  const email = authDraft.email.trim().toLowerCase();
  const password = authDraft.password;
  fieldErrors = {};

  if (authMode === 'signup') {
    const name = authDraft.name.trim();
    if (name.length < 2) fieldErrors.name = 'Enter your full name (at least 2 characters).';
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) fieldErrors.email = 'Enter a valid email address.';
    if (password.length < 8) fieldErrors.password = 'Password must be at least 8 characters long.';
    if (password !== authDraft.confirm) fieldErrors.confirm = 'Passwords do not match.';

    if (Object.keys(fieldErrors).length) {
      shakeAuth = true;
      renderedAuthBusy = null;
      renderAuth();
      toast(Object.values(fieldErrors)[0]);
      return;
    }

    if (!host) {
      const initials = name.split(/\s+/).slice(0, 2).map(p => p[0].toUpperCase()).join('') || 'PA';
      state.user = { id: 'preview', name, email, initials };
      state.notice = { kind: 'success', text: `Preview account created for ${name}.` };
      render();
      return;
    }
    emit('signup', { name, email, password, confirm_password: authDraft.confirm });
    return;
  }

  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) fieldErrors.email = 'Enter a valid email address.';
  if (!password) fieldErrors.password = 'Enter your password.';
  if (Object.keys(fieldErrors).length) {
    shakeAuth = true;
    renderedAuthBusy = null;
    renderAuth();
    toast(Object.values(fieldErrors)[0]);
    return;
  }

  if (!host) {
    const handle = email.split('@')[0] || 'User';
    const name = handle.charAt(0).toUpperCase() + handle.slice(1);
    state.user = { id: 'preview', name, email, initials: name.slice(0, 2).toUpperCase() };
    state.notice = { kind: 'success', text: `Welcome back, ${name} (design preview).` };
    render();
    return;
  }
  emit('login', { email, password });
}

function bindPage() {
  document.querySelectorAll('.upload-zone').forEach(el => {
    el.addEventListener('dragover', e => {
      e.preventDefault();
      el.classList.add('dragover');
    });
    el.addEventListener('dragleave', () => el.classList.remove('dragover'));
    el.addEventListener('drop', e => {
      e.preventDefault();
      el.classList.remove('dragover');
      upload(e.dataTransfer.files);
    });
  });
  document.getElementById('category')?.addEventListener('change', e => (category = e.target.value));
  document.getElementById('document-select')?.addEventListener('change', e => (selected = e.target.value));
  document.getElementById('question')?.addEventListener('input', e => (draft = e.target.value));
  document.getElementById('question')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });
  document.getElementById('chat-form')?.addEventListener('submit', e => {
    e.preventDefault();
    ask();
  });
  document.getElementById('search')?.addEventListener('input', e => {
    search = e.target.value;
    document.getElementById('library-results').innerHTML = libraryResults();
    size();
  });
  document.getElementById('filter')?.addEventListener('change', e => {
    filter = e.target.value;
    document.getElementById('library-results').innerHTML = libraryResults();
    size();
  });

  document.querySelectorAll('.citation[data-source]').forEach(chip => {
    const toggleHighlight = active => {
      const target = document.querySelector(`.source-entry[data-source="${CSS.escape(chip.dataset.source)}"][data-document="${CSS.escape(chip.dataset.document)}"]`);
      target?.classList.toggle('is-highlighted', active);
    };
    chip.addEventListener('mouseenter', () => toggleHighlight(true));
    chip.addEventListener('mouseleave', () => toggleHighlight(false));
  });
}

async function upload(files) {
  if (busy) return;
  if (!host) {
    toast('Run the Streamlit app to extract and index your PDFs.');
    return;
  }
  const items = Array.from(files);
  if (!items.length) return;
  for (const file of items) {
    if (!file.name.toLowerCase().endsWith('.pdf')) return toast('Please choose PDF files only.');
    if (file.size > state.limit * 1024 * 1024) return toast(`${file.name} exceeds the ${state.limit} MB limit.`);
  }
  if (items.reduce((n, f) => n + f.size, 0) > 60 * 1024 * 1024) return toast('Please upload no more than 60 MB in one batch.');
  try {
    const data = await Promise.all(
      items.map(
        file =>
          new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve({ name: file.name, data: reader.result.split(',')[1], category });
            reader.onerror = reject;
            reader.readAsDataURL(file);
          })
      )
    );
    emit('upload', { files: data });
  } catch {
    toast('The files could not be read. Please try again.');
  }
}

function ask() {
  const question = draft.trim();
  if (!question) return toast('Start with a question about your document.');
  if (!state.documents.length) return toast('Upload a document first to get a grounded answer.');
  const ids = selected === 'all' ? state.documents.map(d => d.id) : [selected];
  if (!host) return toast('Run Streamlit to ask PolicyAI.');
  draft = '';
  emit('ask', { question, document_ids: ids });
}

function modal(title, body, actions = '') {
  focusBeforeModal = document.activeElement;
  document.getElementById('modal-root').innerHTML = `<div class="modal-overlay"><section class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title"><div class="modal-head"><h2 id="modal-title">${title}</h2><button class="icon-button" data-action="close-modal" aria-label="Close dialog">${icon('close', 20)}</button></div>${body}${actions ? `<div class="modal-actions">${actions}</div>` : ''}</section></div>`;
  document.querySelector('[data-action="close-modal"]').focus();
}

function closeModal() {
  document.getElementById('modal-root').innerHTML = '';
  focusBeforeModal?.focus();
}

function showSource(number, documentId, messageIndex) {
  const s = state.messages[Number(messageIndex)]?.sources?.find(s => s.number === Number(number) && s.document_id === documentId);
  if (!s) return;
  modal(
    'Go straight to the source.',
    `<p><strong>${esc(s.document)}</strong><br>PDF page ${s.page}${s.clause ? ' · Clause ' + esc(s.clause) + (s.clause_title ? ' — ' + esc(s.clause_title) : '') : ''}</p><span class="pill">Original extracted passage</span><div class="source-quote">${esc(s.text)}</div><p style="margin-top:16px">This is the extracted text from the cited PDF page. Check the original document for its full layout and surrounding conditions.</p>`
  );
}

document.addEventListener('click', e => {
  if (e.target.classList?.contains('modal-overlay')) {
    closeModal();
    return;
  }
  const el = e.target.closest('button');
  if (!el) return;
  if (el.dataset.authTab) {
    const next = el.dataset.authTab;
    if (next !== authMode) {
      authDirection = next === 'signup' ? 'right' : 'left';
      authMode = next;
      fieldErrors = {};
      state.notice = null;
      render();
    }
    return;
  }
  if (el.dataset.page) return go(el.dataset.page);
  if (el.dataset.question) {
    draft = el.dataset.question;
    go('Ask PolicyAI');
    document.getElementById('question')?.focus();
    return;
  }
  if (el.dataset.source) return showSource(el.dataset.source, el.dataset.document, el.dataset.message);

  switch (el.dataset.action) {
    case 'toggle-pass': {
      showPass = !showPass;
      ['auth-password', 'auth-confirm'].forEach(id => {
        const input = document.getElementById(id);
        if (input) input.type = showPass ? 'text' : 'password';
      });
      el.innerHTML = icon(showPass ? 'eyeOff' : 'eye', 15);
      el.setAttribute('aria-label', showPass ? 'Hide password' : 'Show password');
      document.getElementById('auth-password')?.focus();
      break;
    }
    case 'guest-login':
      if (!host) {
        state.user = { id: 'guest', name: 'Guest Workspace', email: 'Design preview', initials: 'GW', guest: true };
        render();
        break;
      }
      emit('guest_login');
      break;
    case 'logout':
      if (!host) {
        state.user = null;
        state.documents = [];
        state.messages = [];
        state.notice = { kind: 'success', text: 'Signed out of preview workspace.' };
        render();
        break;
      }
      emit('logout');
      break;
    case 'menu':
      menu = !menu;
      render();
      break;
    case 'upload-page':
      go('Upload Documents');
      break;
    case 'chat-page':
      go('Ask PolicyAI');
      break;
    case 'browse':
      document.getElementById('file-input').click();
      break;
    case 'how':
    case 'help':
      modal(
        'A clearer answer, in three steps.',
        `<p><strong>01 / Bring your documents.</strong><br>Upload a text-based insurance policy or loan agreement. Choose its type to keep your library organized.</p><p><strong>02 / Ask in your own words.</strong><br>Select a document and ask what you need to understand. PolicyAI retrieves relevant passages via LangChain FAISS + Sentence-Transformers and asks the configured LLM to explain them.</p><p><strong>03 / Follow the evidence.</strong><br>Click a citation to see the supporting text, PDF page, and clause where available.</p><p>${host ? `Active LLM backend: ${esc(state.provider || 'Hugging Face (Local)')} (${esc(state.model || 'default')}).` : 'You’re viewing the frontend preview. Launch Streamlit to enable PDF processing and RAG answers.'}</p>`
      );
      break;
    case 'dismiss':
      state.notice = null;
      render();
      break;
    case 'close-modal':
      closeModal();
      break;
    case 'motion':
      motion = !motion;
      render();
      break;
    case 'open-doc':
      selected = el.dataset.doc;
      go('Ask PolicyAI');
      break;
    case 'remove':
      modal(
        'Remove this document?',
        `<p>This removes the document and clears the conversation in this session. Your original PDF is not affected.</p>`,
        `<button class="button" data-action="close-modal">Keep document</button><button class="button dark" data-action="confirm-remove" data-doc="${el.dataset.doc}">Remove document</button>`
      );
      break;
    case 'confirm-remove':
      closeModal();
      selected = 'all';
      emit('delete', { document_id: el.dataset.doc });
      break;
    case 'clear':
      if (state.messages.length) emit('clear');
      else {
        draft = '';
        renderedPageSignature = '';
        render();
      }
      break;
    case 'compare': {
      const a = document.getElementById('compare-a').value;
      const b = document.getElementById('compare-b').value;
      if (!a || !b || a === b) return toast('Choose two different documents to compare.');
      page = 'Ask PolicyAI';
      emit('compare', {
        document_ids: [a, b],
        question:
          'Compare these two documents. For each document, explain coverage, exclusions, premiums or interest, fees, penalties, waiting periods, repayment and renewal conditions where relevant. Cite evidence for each difference, and explicitly mark missing information.',
      });
      break;
    }
    case 'summary': {
      const id = document.getElementById('summary-document').value;
      if (!id) return toast('Choose a document to summarize.');
      selected = id;
      page = 'Ask PolicyAI';
      emit('summary', {
        document_ids: [id],
        question:
          'Summarize this document: key amounts, coverage or loan terms, exclusions, waiting periods, obligations, fees and penalties, claims or repayment, and renewal. Cite supporting passages and mark information that is not specified.',
      });
      break;
    }
  }
});

document.addEventListener('keydown', e => {
  const dialog = document.querySelector('.modal');
  if (!dialog) return;
  if (e.key === 'Escape') {
    closeModal();
    return;
  }
  if (e.key === 'Tab') {
    const elements = dialog.querySelectorAll('button,a,input,select,textarea,[tabindex="0"]');
    const first = elements[0];
    const last = elements[elements.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
});

window.addEventListener('resize', size);
render();
