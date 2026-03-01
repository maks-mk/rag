const API = window.location.origin;

marked.setOptions({ breaks: true, gfm: true });

const state = {
  pendingFiles: new Map(),
  indexedFiles: new Map(),
  isIndexing: false,
  isQuerying: false,
  sidebarOpen: window.innerWidth > 768, // Открыт по умолчанию только на ПК
};

const MAX_FILE_SIZE = 50 * 1024 * 1024;
const ALLOWED_EXTS = ['txt', 'md', 'pdf', 'docx'];

const DOM = {
  app: document.getElementById('app'),
  btnToggle: document.getElementById('btn-sidebar-toggle'),
  uploadZone: document.getElementById('upload-zone'),
  fileInput: document.getElementById('file-input'),
  fileList: document.getElementById('file-list'),
  btnIndex: document.getElementById('btn-index'),
  btnIndexText: document.getElementById('btn-index-text'),
  pendingCount: document.getElementById('pending-count'),
  chatInput: document.getElementById('chat-input'),
  chatMessages: document.getElementById('chat-messages'),
  btnSend: document.getElementById('btn-send'),
  apiStatus: document.getElementById('api-status'),
  chunksStatus: document.querySelector('#chunks-status .dot'),
  chunksCount: document.getElementById('chunks-count'),
  toastContainer: document.getElementById('toast-container')
};

// --- Инициализация состояния ---
if (!state.sidebarOpen) {
  DOM.app.classList.add('sidebar-collapsed');
}

// --- Улучшенная система уведомлений (Stackable Toasts) ---
function toast(msg, type = 'info', duration = 4000) {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  
  DOM.toastContainer.appendChild(el);
  
  // Trigger reflow для анимации
  el.offsetHeight; 
  el.classList.add('show');
  
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300); // Ожидание завершения CSS-transition
  }, duration);
}

function formatBytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  return (b / 1048576).toFixed(1) + ' MB';
}

function getExt(name) { return name.split('.').pop().toLowerCase(); }

// Правильный скролл через requestAnimationFrame
function scrollToBottom() {
  requestAnimationFrame(() => {
    DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
  });
}

// --- Обработчики событий ---
DOM.btnToggle.addEventListener('click', () => {
  state.sidebarOpen = !state.sidebarOpen;
  DOM.app.classList.toggle('sidebar-collapsed', !state.sidebarOpen);
  DOM.btnToggle.title = state.sidebarOpen ? 'Скрыть панель' : 'Показать панель';
});

// Плавный ресайз поля ввода
DOM.chatInput.addEventListener('input', function() {
  this.style.height = 'auto'; // Сброс высоты перед вычислением
  this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

DOM.chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { 
    e.preventDefault(); 
    sendMessage(); 
  }
});

DOM.btnSend.addEventListener('click', sendMessage);

// Делегирование событий для кнопок удаления (Оптимизация производительности)
DOM.fileList.addEventListener('click', (e) => {
  const deleteBtn = e.target.closest('.file-delete');
  if (deleteBtn) {
    deleteFile(deleteBtn.dataset.id);
  }
});

async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error('API Offline');
    const d = await r.json();
    DOM.apiStatus.innerHTML = '<div class="dot dot-green"></div><span>connected</span>';
    updateChunksCount(d.chunks_indexed);
  } catch {
    DOM.apiStatus.innerHTML = '<div class="dot dot-red"></div><span>offline</span>';
  }
}

function updateChunksCount(n) {
  DOM.chunksCount.textContent = `${n} chunks`;
  DOM.chunksStatus.style.opacity = n > 0 ? '1' : '0.4';
}

async function loadIndexedDocs() {
  try {
    const r = await fetch(`${API}/documents`);
    const d = await r.json();
    for (const doc of d.documents) {
      state.indexedFiles.set(doc.file_id, { ...doc, status: 'indexed' });
    }
    renderFileList();
  } catch (e) {
    console.warn("Failed to load documents", e);
  }
}

// --- Drag & Drop ---['dragover', 'drop'].forEach(evt => window.addEventListener(evt, e => e.preventDefault(), false));

DOM.uploadZone.addEventListener('dragover', () => DOM.uploadZone.classList.add('drag-over'));
DOM.uploadZone.addEventListener('dragleave', () => DOM.uploadZone.classList.remove('drag-over'));
DOM.uploadZone.addEventListener('drop', e => {
  DOM.uploadZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});
DOM.fileInput.addEventListener('change', () => {
  handleFiles(DOM.fileInput.files);
  DOM.fileInput.value = ''; 
});

async function handleFiles(files) {
  const uploadPromises =[];
  
  for (const file of files) {
    const ext = getExt(file.name);
    if (!ALLOWED_EXTS.includes(ext)) {
      toast(`Формат .${ext} не поддерживается`, 'err');
      continue;
    }
    if (file.size > MAX_FILE_SIZE) {
      toast(`Файл ${file.name} слишком большой (макс 50MB)`, 'err');
      continue;
    }
    uploadPromises.push(uploadFile(file));
  }
  await Promise.allSettled(uploadPromises);
}

async function uploadFile(file) {
  const tempId = 'tmp_' + Math.random().toString(36).slice(2);
  state.pendingFiles.set(tempId, {
    tempId, file_name: file.name, ext: getExt(file.name), size: formatBytes(file.size), status: 'uploading',
  });
  renderFileList();

  const fd = new FormData();
  fd.append('file', file);

  try {
    const r = await fetch(`${API}/upload`, { method: 'POST', body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Upload failed');

    state.pendingFiles.delete(tempId);
    state.pendingFiles.set(d.file_id, {
      file_id: d.file_id, file_name: d.file_name, ext: getExt(d.file_name), chars: d.chars, status: 'pending',
    });
    toast(`Загружено: ${d.file_name}`, 'ok');
  } catch (e) {
    const f = state.pendingFiles.get(tempId);
    if (f) f.status = 'error';
    toast(`Ошибка загрузки: ${e.message}`, 'err');
  }

  renderFileList();
  updateIndexButton();
}

async function deleteFile(fid) {
  if (state.pendingFiles.has(fid)) {
    state.pendingFiles.delete(fid);
  } else if (state.indexedFiles.has(fid)) {
    try {
      await fetch(`${API}/documents/${fid}`, { method: 'DELETE' });
      state.indexedFiles.delete(fid);
      toast('Документ удален', 'ok');
      checkHealth();
    } catch(e) {
      toast('Не удалось удалить документ', 'err');
    }
  }
  renderFileList();
  updateIndexButton();
}

function renderFileList() {
  const allFiles = [...state.pendingFiles.values(), ...state.indexedFiles.values()];

  if (allFiles.length === 0) {
    DOM.fileList.innerHTML = '<div class="empty-list">Нет загруженных файлов</div>';
    DOM.pendingCount.textContent = '';
    return;
  }

  const pendingCount = Array.from(state.pendingFiles.values()).filter(f => f.status === 'pending').length;
  DOM.pendingCount.textContent = pendingCount > 0 ? `(${pendingCount} ожидает)` : '';

  DOM.fileList.innerHTML = allFiles.map(f => {
    const statusMap = {
      uploading: ['загрузка…', 'status-indexing'],
      pending:   ['ожидает',    'status-pending'],
      indexing:['индексация…','status-indexing'],
      indexed:[`${f.chunks || '?'} чанков`, 'status-indexed'],
      error:     ['ошибка',      'status-error'],
    };
    const [statusText, statusClass] = statusMap[f.status] || ['неизвестно', ''];
    const isDeletable = ['indexed', 'pending', 'error'].includes(f.status);
    const fid = f.file_id || f.tempId;
    const ext = f.ext || (f.file_type ? f.file_type.replace('.', '') : null) || getExt(f.file_name) || '?';

    // XSS защита: экранирование имени файла в атрибутах
    const safeName = f.file_name.replace(/"/g, '&quot;');
    
    return `
      <div class="file-item">
        <div class="file-icon">${ext}</div>
        <div class="file-info">
          <div class="file-name" title="${safeName}">${safeName}</div>
          <div class="file-meta">${f.chars ? (f.chars/1000).toFixed(1)+'k симв.' : (f.size || '')}</div>
        </div>
        <div class="file-status ${statusClass}">${statusText}</div>
        ${isDeletable ? `<button class="file-delete" data-id="${fid}" title="Удалить">×</button>` : ''}
      </div>
    `;
  }).join('');
}

function updateIndexButton() {
  const pending = Array.from(state.pendingFiles.values()).filter(f => f.status === 'pending');
  DOM.btnIndex.disabled = pending.length === 0 || state.isIndexing;
  DOM.btnIndexText.textContent = pending.length > 0 ? `ИНДЕКСИРОВАТЬ (${pending.length})` : 'ИНДЕКСИРОВАТЬ';
}

DOM.btnIndex.addEventListener('click', async () => {
  const pending = Array.from(state.pendingFiles.values()).filter(f => f.status === 'pending');
  if (!pending.length) return;

  state.isIndexing = true;
  DOM.btnIndex.disabled = true;
  DOM.btnIndex.classList.add('indexing');
  DOM.btnIndexText.textContent = 'ОБРАБОТКА...';

  for (const f of pending) f.status = 'indexing';
  renderFileList();

  const ids = pending.map(f => f.file_id);

  try {
    const r = await fetch(`${API}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ file_ids: ids }),
    });
    const d = await r.json();

    d.indexed.forEach(res => {
      state.pendingFiles.delete(res.file_id);
      state.indexedFiles.set(res.file_id, { ...res, status: 'indexed' });
    });
    
    d.errors.forEach(err => {
      const f = state.pendingFiles.get(err.file_id);
      if (f) f.status = 'error';
      toast(`Ошибка: ${err.error}`, 'err', 5000);
    });
    
    if (d.indexed.length > 0) toast(`Успешно проиндексировано: ${d.indexed.length}`, 'ok');
  } catch (e) {
    toast(`Сбой сети: ${e.message}`, 'err');
    for (const f of pending) f.status = 'error';
  }

  state.isIndexing = false;
  DOM.btnIndex.classList.remove('indexing');
  renderFileList();
  updateIndexButton();
  checkHealth();
});

function formatMessage(text) {
  try {
    let html = marked.parse(text);
    // Оптимизированный фикс для кода внутри таблиц
    html = html.replace(/<code>([\s\S]*?)<\/code>/g, (match, content) => {
      if (content.match(/&lt;br\s*\/?&gt;/i)) {
        let unescaped = content.replace(/&lt;br\s*\/?&gt;/gi, '\n');
        return `<pre><code>${unescaped.trim()}</code></pre>`;
      }
      return match;
    });
    return html;
  } catch (e) {
    return `<p>${text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>')}</p>`;
  }
}

function appendMessage(role, content, meta = null) {
  const emptyState = document.getElementById('empty-state');
  if (emptyState) emptyState.remove();

  const el = document.createElement('div');
  el.className = `message ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `msg-avatar ${role === 'user' ? 'user-av' : 'bot-av'}`;
  avatar.textContent = role === 'user' ? 'ВЫ' : 'RAG';

  const contentDiv = document.createElement('div');
  contentDiv.className = 'msg-content';

  const bubble = document.createElement('div');
  bubble.className = 'msg-bubble';
  bubble.innerHTML = formatMessage(content);
  contentDiv.appendChild(bubble);

  if (meta && role === 'bot') {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-meta';

    if (meta.confidence !== undefined) {
      const conf = meta.confidence;
      const confClass = conf >= 0.7 ? 'conf-high' : conf >= 0.4 ? 'conf-mid' : 'conf-low';
      const confBadge = document.createElement('div');
      confBadge.className = `confidence-badge ${confClass}`;
      confBadge.innerHTML = `<span>⚡</span> Точность ~${Math.round(conf * 100)}%`;
      metaDiv.appendChild(confBadge);
    }

    if (meta.sources && meta.sources.length > 0) {
      const panel = document.createElement('div');
      panel.className = 'sources-panel';
      panel.innerHTML = meta.sources.map(s => `
        <div class="source-item">
          <div class="source-score">${Math.round(s.score * 100)}%</div>
          <div>
            <div class="source-file">${s.file_name} · фрагмент ${s.chunk_index + 1}</div>
            <div class="source-preview">${s.preview.replace(/</g, '&lt;')}</div>
          </div>
        </div>
      `).join('');

      const toggle = document.createElement('button');
      toggle.className = 'sources-toggle';
      
      const updateToggleText = () => {
        toggle.innerHTML = `${panel.classList.contains('open') ? 'Скрыть' : 'Показать'} источники (${meta.sources.length})`;
      };
      
      updateToggleText();
      toggle.onclick = () => {
        panel.classList.toggle('open');
        updateToggleText();
        if (panel.classList.contains('open')) scrollToBottom();
      };

      metaDiv.appendChild(toggle);
      contentDiv.appendChild(metaDiv);
      contentDiv.appendChild(panel);
    } else {
      contentDiv.appendChild(metaDiv);
    }
  }

  el.appendChild(avatar);
  el.appendChild(contentDiv);
  DOM.chatMessages.appendChild(el);
  scrollToBottom();

  return el;
}

function appendTyping() {
  const el = document.createElement('div');
  el.className = 'message bot';
  el.id = 'typing-indicator';
  el.innerHTML = `
    <div class="msg-avatar bot-av">RAG</div>
    <div class="msg-content">
      <div class="msg-bubble">
        <div class="typing-indicator">
          <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
        </div>
      </div>
    </div>
  `;
  DOM.chatMessages.appendChild(el);
  scrollToBottom();
  return el;
}

async function sendMessage() {
  const question = DOM.chatInput.value.trim();
  if (!question || state.isQuerying) return;

  if (state.indexedFiles.size === 0) {
    toast('Сначала загрузите и проиндексируйте хотя бы один документ', 'info');
    return;
  }

  DOM.chatInput.value = '';
  DOM.chatInput.style.height = 'auto'; // Сброс высоты
  DOM.btnSend.disabled = true;
  state.isQuerying = true;

  appendMessage('user', question);
  const typing = appendTyping();

  try {
    const r = await fetch(`${API}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: 7 }),
    });
    
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || 'Query failed');

    typing.remove();
    appendMessage('bot', d.answer, { confidence: d.confidence, sources: d.sources });
  } catch (e) {
    typing.remove();
    appendMessage('bot', `**Ошибка:** ${e.message}`);
    toast(`Не удалось получить ответ`, 'err');
  }

  state.isQuerying = false;
  DOM.btnSend.disabled = false;
  DOM.chatInput.focus();
}

// Запуск
checkHealth();
loadIndexedDocs();
setInterval(checkHealth, 30000);