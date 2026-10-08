const $ = (s) => document.querySelector(s);
const CURVE_KEYS = ['master', 'C', 'M', 'Y', 'K'];
const CHANNEL_COLORS = { master: '#171b18', C: '#0aa5c8', M: '#d6288f', Y: '#d8b40a', K: '#3b3f3c' };
const state = {
    batchId: null,
    batch: null,
    poll: null,
    compareModes: {},   // 每个文件的对比模式：'input' 或 'target'
    viewModes: {},      // 保留字段，审核页不再用 iframe 打开 CMYK PDF
    selectedId: null,   // 左侧列表当前选中的文件
    pageIndex: 0,       // 多页 PDF 当前页
    uploadedFiles: { inputs: [], targets: [] },
    curves: identityCurves(),
    curveChan: 'master',
    curveImageId: null,
    curvePageIndex: 0,
    curveDrag: null,
    curveSelected: null,
    curveDiscard: false,
    curvePreviewTimer: null,
    curvePreviewUrl: null,
    curvePoll: null
};

function identityCurves() {
    const curves = {};
    CURVE_KEYS.forEach(k => { curves[k] = [[0, 0], [255, 255]]; });
    return curves;
}

function toast(message) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), 2400);
}

async function api(url, options = {}) {
    let r;
    try {
        r = await fetch(url, options);
    } catch {
        throw new Error('无法连接服务器，请确认服务已启动');
    }
    let d;
    try {
        d = await r.json();
    } catch {
        d = null;
    }
    if (!r.ok) throw new Error(d?.error || `请求失败 ${r.status}`);
    return d;
}

function route() {
    const review = location.hash.startsWith('#review');
    $('#uploadView').classList.toggle('hidden', review);
    $('#reviewView').classList.toggle('hidden', !review);
    document.querySelectorAll('[data-nav]').forEach(x =>
        x.classList.toggle('active', x.dataset.nav === (review ? 'review' : 'upload'))
    );
    if (review) {
        const id = new URLSearchParams(location.hash.split('?')[1] || '').get('batch') || state.batchId;
        if (id) openReview(id);
        else loadBatches();
    }
}

// ---------- 文件选择与上传 ----------
const drop = $('#dropzone');
const input = $('#fileInput');

drop.onclick = () => input.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('drag'); };
drop.ondragleave = () => drop.classList.remove('drag');
drop.ondrop = e => {
    e.preventDefault();
    drop.classList.remove('drag');
    selectFiles(e.dataTransfer.files);
};
input.onchange = () => selectFiles(input.files);

function selectFiles(list) {
    const hasTarget = $('#hasTarget').checked;
    const images = [...list].filter(f => /\.(jpe?g|png|tiff?|pdf)$/i.test(f.name));
    if (hasTarget) {
        const nameMap = new Map();
        images.forEach(f => {
            const base = f.name.replace(/\.[^.]+$/, '');
            if (!nameMap.has(base)) nameMap.set(base, []);
            nameMap.get(base).push(f);
        });
        state.uploadedFiles.inputs = [];
        state.uploadedFiles.targets = [];
        for (const [base, files] of nameMap) {
            if (files.length === 1) {
                state.uploadedFiles.inputs.push(files[0]);
            } else {
                state.uploadedFiles.inputs.push(files[0]);
                state.uploadedFiles.targets.push(files[1] || files[0]);
            }
        }
    } else {
        state.uploadedFiles.inputs = images;
        state.uploadedFiles.targets = [];
    }
    updateSelectionDisplay();
}

function updateSelectionDisplay() {
    const total = state.uploadedFiles.inputs.length + state.uploadedFiles.targets.length;
    const sel = $('#selection');
    if (total === 0) {
        sel.classList.add('hidden');
    } else {
        sel.classList.remove('hidden');
        sel.textContent = `已选择 ${state.uploadedFiles.inputs.length} 张原图${state.uploadedFiles.targets.length ? `，${state.uploadedFiles.targets.length} 张目标图` : ''}`;
    }
    $('#startButton').disabled = state.uploadedFiles.inputs.length === 0;
}

// ---------- 创建批次并上传 ----------
$('#startButton').onclick = async () => {
    try {
        $('#startButton').disabled = true;
        const name = $('#batchName').value || '未命名批次';
        const batch = await api('/api/batches', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        state.batchId = batch.id;

        $('#emptyQueue').classList.add('hidden');
        $('#queue').classList.remove('hidden');
        $('#queue').innerHTML = '';

        for (const file of state.uploadedFiles.inputs) {
            await uploadOneFile(batch.id, file, 'input');
        }
        for (const file of state.uploadedFiles.targets) {
            await uploadOneFile(batch.id, file, 'target');
        }

        toast('上传完成，模型正在处理');
        startPolling();
    } catch (e) {
        toast(e.message);
        $('#startButton').disabled = false;
    }
};

async function uploadOneFile(batchId, file, type) {
    const response = await fetch(`/api/batches/${batchId}/images?filename=${encodeURIComponent(file.name)}`, {
        method: 'POST',
        headers: {
            'Content-Type': file.type || 'application/octet-stream',
        },
        body: file,
    });
    if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || '上传失败');
    }
    const row = document.createElement('div');
    row.className = 'queue-item';
    row.dataset.name = file.name;
    const isPdf = /\.pdf$/i.test(file.name);
    const thumb = isPdf
        ? '<div class="queue-thumb placeholder">PDF</div>'
        : `<img class="queue-thumb" src="${URL.createObjectURL(file)}">`;
    row.innerHTML = `
        ${thumb}
        <div>
            <div class="queue-name">${escapeHtml(file.name)}</div>
            <div class="queue-state">已上传 (${type})</div>
        </div>
        <div class="state-icon spinner">◌</div>
    `;
    $('#queue').append(row);
}

// ---------- 轮询 ----------
function startPolling() {
    clearInterval(state.poll);
    refreshQueue();
    state.poll = setInterval(refreshQueue, 2000);
}

async function refreshQueue() {
    if (!state.batchId) return;
    try {
        const b = await api(`/api/batches/${state.batchId}`);
        state.batch = b;
        $('#profileName').textContent = b.target_profile || '模型就绪';
        const items = b.images || [];
        $('#queue').innerHTML = items.map(i => {
            const statusMap = {
                'queued': '等待处理',
                'processing': '处理中',
                'completed': '已完成',
                'failed': '失败'
            };
            const stateIcon = i.process_status === 'completed' ? 'done' :
                              i.process_status === 'failed' ? 'fail' : 'spinner';
            const stateChar = i.process_status === 'completed' ? '✓' :
                              i.process_status === 'failed' ? '!' : '◌';
            const errorMsg = i.error ? escapeHtml(i.error) : '';
            const trailing = i.process_status === 'completed'
                ? `<a class="queue-download" href="/media/${state.batchId}/${i.id}/output" download="${escapeAttr(downloadName(i.filename))}">下载</a>`
                : `<div class="state-icon ${stateIcon}">${stateChar}</div>`;
            const thumb = i.process_status === 'completed'
                ? `<img class="queue-thumb" src="/media/${state.batchId}/${i.id}/preview?page=0">`
                : `<div class="queue-thumb placeholder">${i.is_pdf ? 'PDF' : '◌'}</div>`;
            const pages = (i.page_count || 1) > 1 ? ` · ${i.page_count} 页` : '';
            return `
                <div class="queue-item">
                    ${thumb}
                    <div>
                        <div class="queue-name">${escapeHtml(i.filename)}</div>
                        <div class="queue-state">${statusMap[i.process_status] || i.process_status}${pages} ${errorMsg}</div>
                    </div>
                    ${trailing}
                </div>
            `;
        }).join('');

        const counts = b.counts || { total: 0, completed: 0, processing: 0, failed: 0 };
        if (counts.processing === 0 && counts.total > 0) {
            clearInterval(state.poll);
            $('#reviewButton').classList.remove('hidden');
            $('#startButton').disabled = false;
            loadBatches();
        }
    } catch (e) {
        clearInterval(state.poll);
        toast(e.message);
    }
}

$('#reviewButton').onclick = () => location.hash = `review?batch=${state.batchId}`;
$('#backButton').onclick = () => location.hash = 'upload';

// ---------- 审核工作台 ----------
async function openReview(id) {
    state.batchId = id;
    try {
        const b = await api(`/api/batches/${id}`);
        state.batch = b;
        $('#reviewTitle').textContent = b.name;
        const curveNote = curvesAreIdentity(normalizeCurves(b.curves)) ? '' : ' · 已套用手工曲线';
        $('#reviewMeta').textContent =
            `${b.target_profile || 'CMYK'} 输出${curveNote} · 创建于 ${new Date(b.created_at).toLocaleString()}`;
        $('#profileName').textContent = b.target_profile || 'CMYK';
        $('#downloadButton').href = `/api/batches/${id}/download?status=approved`;
        renderStats(b.counts || {});
        renderReview();
    } catch (e) {
        toast(e.message);
    }
}

function renderStats(c) {
    $('#stats').innerHTML = [
        ['总计', c.total || 0],
        ['待审核', c.pending || 0],
        ['已通过', c.approved || 0],
        ['已驳回', c.rejected || 0],
        ['处理失败', c.failed || 0]
    ].map(([n, v]) => `<div class="stat"><strong>${v}</strong><span>${n}</span></div>`).join('');
}

function visibleItems() {
    const filter = $('#statusFilter').value;
    return (state.batch?.images || []).filter(i => filter === 'all' || i.review_status === filter);
}

function pageCount(item) {
    return Math.max(1, item.page_count || (item.pages || []).length || 1);
}

function mediaVersion() {
    return encodeURIComponent(state.batch?.updated_at || '');
}

// ---------- 左侧文件列表 ----------
function renderReview() {
    const items = visibleItems();
    $('#noReview').classList.toggle('hidden', items.length > 0);
    $('#fileCount').textContent = items.length ? `${items.length} 个` : '';

    if (!items.some(i => i.id === state.selectedId)) {
        const firstDone = items.find(i => i.process_status === 'completed');
        state.selectedId = (firstDone || items[0] || {}).id || null;
        state.pageIndex = 0;
    }

    const v = mediaVersion();
    $('#fileList').innerHTML = items.map(i => {
        const statusMap = { queued: '等待处理', processing: '处理中', completed: '已完成', failed: '失败' };
        const badge = i.review_status === 'approved' ? '<span class="tag ok">通过</span>'
            : i.review_status === 'rejected' ? '<span class="tag no">驳回</span>'
            : '<span class="tag wait">待审</span>';
        const pages = pageCount(i);
        const meta = i.process_status === 'completed'
            ? `${pages > 1 ? `${pages} 页 · ` : ''}${i.is_pdf ? 'PDF' : (i.filename.split('.').pop() || '').toUpperCase()}`
            : `${statusMap[i.process_status] || i.process_status}${i.error ? '：' + escapeHtml(i.error) : ''}`;
        const thumb = i.process_status === 'completed'
            ? `<img class="file-thumb" src="/media/${state.batchId}/${i.id}/preview?page=0&v=${v}">`
            : `<div class="file-thumb placeholder">${i.process_status === 'failed' ? '!' : '◌'}</div>`;
        return `
            <button type="button" class="file-item ${i.id === state.selectedId ? 'active' : ''}" data-id="${i.id}">
                ${thumb}
                <div class="file-info">
                    <div class="file-name" title="${escapeAttr(i.filename)}">${escapeHtml(i.filename)}</div>
                    <div class="file-meta">${meta}</div>
                </div>
                ${i.process_status === 'completed' ? badge : ''}
            </button>
        `;
    }).join('');

    document.querySelectorAll('.file-item').forEach(btn => {
        btn.onclick = () => selectFile(btn.dataset.id);
    });
    renderViewer();
}

function selectFile(id) {
    if (state.selectedId !== id) state.pageIndex = 0;
    state.selectedId = id;
    document.querySelectorAll('.file-item').forEach(b => b.classList.toggle('active', b.dataset.id === id));
    renderViewer();
}

// ---------- 中间大图对比 ----------
function renderViewer() {
    const item = (state.batch?.images || []).find(i => i.id === state.selectedId);
    const empty = $('#viewerEmpty');
    const viewer = $('#viewer');
    if (!item) {
        empty.classList.remove('hidden');
        viewer.classList.add('hidden');
        return;
    }
    empty.classList.add('hidden');
    viewer.classList.remove('hidden');

    if (item.process_status !== 'completed') {
        viewer.className = 'viewer';
        viewer.innerHTML = `<div class="viewer-status"><strong>${escapeHtml(item.filename)}</strong>
            <p>${item.error ? escapeHtml(item.error) : '正在处理，请稍候…'}</p></div>`;
        return;
    }

    const pages = pageCount(item);
    state.pageIndex = Math.max(0, Math.min(pages - 1, state.pageIndex || 0));
    const p = state.pageIndex;
    const v = mediaVersion();
    const hasTarget = !!item.target_file;
    const mode = state.compareModes[item.id] || (hasTarget ? 'target' : 'input');

    const inputUrl = `/media/${state.batchId}/${item.id}/input-preview?page=${p}&v=${v}`;
    const outputUrl = `/media/${state.batchId}/${item.id}/preview?page=${p}&v=${v}`;
    const targetUrl = hasTarget ? `/media/${state.batchId}/${item.id}/target-preview` : inputUrl;
    const before = mode === 'target' && hasTarget ? targetUrl : inputUrl;
    const beforeLabel = mode === 'target' && hasTarget ? '目标图' : '原文件';

    const de = item.metrics?.mean_delta_e;
    const dl = downloadName(item.filename);
    const stage = `
            <div class="compare large">
                <img class="before" src="${before}">
                <img class="after" src="${outputUrl}">
                <div class="divider"></div>
                <span class="compare-label before-label">${beforeLabel}</span>
                <span class="compare-label after-label">输出（交付文件软打样）</span>
                <input type="range" min="0" max="100" value="50" aria-label="拖动比较">
            </div>`;

    const pager = pages > 1 ? `
        <div class="pager">
            <button class="pager-btn" data-page="${p - 1}" ${p === 0 ? 'disabled' : ''}>‹</button>
            <span>第 ${p + 1} / ${pages} 页</span>
            <button class="pager-btn" data-page="${p + 1}" ${p === pages - 1 ? 'disabled' : ''}>›</button>
        </div>` : '';

    viewer.className = `viewer ${item.review_status}`;
    viewer.innerHTML = `
        <div class="viewer-head">
            <div>
                <h3 title="${escapeAttr(item.filename)}">${escapeHtml(item.filename)}</h3>
                <p class="muted">${item.width}×${item.height}${pages > 1 ? ` · ${pages} 页` : ''}${de ? ` · ΔE ${de.toFixed(2)}` : ''} · 预览为交付 PDF 栅格后的软打样</p>
            </div>
        </div>
        ${stage}
        ${pager}
        <div class="viewer-tools">
            <div class="mode-switch">
                <button class="compare-mode ${mode === 'input' ? 'active' : ''}" data-id="${item.id}" data-mode="input">原文件对比</button>
                <button class="compare-mode ${mode === 'target' ? 'active' : ''}" data-id="${item.id}" data-mode="target" ${hasTarget ? '' : 'disabled'}>目标图对比</button>
            </div>
            <button class="target-upload" data-id="${item.id}">+ 上传目标图</button>
            <input class="target-file-input" data-id="${item.id}" type="file" accept="image/*,.pdf" hidden>
            <a class="button secondary compact viewer-download" href="/media/${state.batchId}/${item.id}/output"
               download="${escapeAttr(dl)}">下载 ${escapeHtml(dl)}</a>
        </div>
        <textarea class="note" data-id="${item.id}" rows="2" placeholder="添加审核备注…">${escapeHtml(item.note || '')}</textarea>
        <div class="decision-row">
            <button data-id="${item.id}" data-status="approved" class="decision approve ${item.review_status === 'approved' ? 'active' : ''}">✓ 通过</button>
            <button data-id="${item.id}" data-status="rejected" class="decision reject ${item.review_status === 'rejected' ? 'active' : ''}">× 驳回</button>
        </div>
    `;
    bindViewerEvents();
}

function bindViewerEvents() {
    document.querySelectorAll('.compare input[type=range]').forEach(sl => {
        sl.oninput = () => sl.parentElement.style.setProperty('--split', `${sl.value}%`);
    });
    document.querySelectorAll('.decision').forEach(btn => {
        btn.onclick = () => reviewItem(btn.dataset.id, btn.dataset.status);
    });
    document.querySelectorAll('.note').forEach(n => {
        n.onchange = () => reviewItem(n.dataset.id,
            state.batch.images.find(x => x.id === n.dataset.id).review_status, n.value);
    });
    document.querySelectorAll('.compare-mode').forEach(btn => {
        btn.onclick = () => { state.compareModes[btn.dataset.id] = btn.dataset.mode; renderViewer(); };
    });
    document.querySelectorAll('.pager-btn').forEach(btn => {
        btn.onclick = () => { state.pageIndex = Number(btn.dataset.page); renderViewer(); };
    });
    document.querySelectorAll('.target-upload').forEach(btn => {
        btn.onclick = () => document.querySelector(`.target-file-input[data-id="${btn.dataset.id}"]`).click();
    });
    document.querySelectorAll('.target-file-input').forEach(input => {
        input.onchange = async () => {
            const file = input.files[0];
            if (!file) return;
            const id = input.dataset.id;
            try {
                toast('上传目标图并重新处理...');
                const response = await fetch(
                    `/api/batches/${state.batchId}/${id}/target?filename=${encodeURIComponent(file.name)}`,
                    { method: 'POST', headers: { 'Content-Type': file.type || 'application/octet-stream' }, body: file });
                if (!response.ok) throw new Error((await response.json()).error || '上传失败');
                toast('目标图已上传并重新处理完成');
                await openReview(state.batchId);
            } catch (e) {
                toast(e.message);
            }
            input.value = '';
        };
    });
}

async function reviewItem(id, status, note) {
    try {
        const item = state.batch.images.find(x => x.id === id);
        await api(`/api/batches/${state.batchId}/${id}/review`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status, note: note ?? item.note })
        });
        await openReview(state.batchId);
        toast(status === 'approved' ? '已通过' : status === 'rejected' ? '已驳回' : '备注已保存');
    } catch (e) {
        toast(e.message);
    }
}

// ---------- 删除批次 ----------
$('#deleteBatchButton').onclick = async () => {
    if (!state.batchId) return;
    try {
        if (!confirm(`确定删除批次「${state.batch.name}」？`)) return;
        await api(`/api/batches/${state.batchId}`, { method: 'DELETE' });
        state.batchId = null;
        state.batch = null;
        if (location.hash.startsWith('#review')) location.hash = 'upload';
        loadBatches();
        toast('批次已删除');
    } catch (e) {
        toast(e.message);
    }
};

// ---------- 最近批次 ----------
async function loadBatches() {
    try {
        const batches = await api('/api/batches');
        const list = $('#batchList');
        if (batches.length === 0) {
            list.innerHTML = '<p class="muted">还没有处理批次</p>';
            return;
        }
        list.innerHTML = batches.map(b => {
            const counts = b.counts || { total: 0, completed: 0, approved: 0 };
            return `
                <article class="batch-card" data-id="${b.id}">
                    <h3>${escapeHtml(b.name)}</h3>
                    <p class="muted">${b.target_profile || 'CMYK'} · ${counts.total} 张图片</p>
                    <div class="mini-progress"><span style="width:${counts.total ? (counts.completed / counts.total * 100) : 0}%"></span></div>
                    <div class="batch-card-footer">
                        <span>${new Date(b.created_at).toLocaleString()}</span>
                        <div class="batch-card-actions">
                            <span>${counts.approved || 0} 已通过</span>
                            <button type="button" class="batch-delete" data-id="${b.id}" data-name="${escapeAttr(b.name)}">删除</button>
                        </div>
                    </div>
                </article>
            `;
        }).join('');
        document.querySelectorAll('.batch-card').forEach(card => {
            card.onclick = () => location.hash = `review?batch=${card.dataset.id}`;
        });
        document.querySelectorAll('.batch-delete').forEach(btn => {
            btn.onclick = async e => {
                e.stopPropagation();
                try {
                    if (!confirm(`确定删除批次「${btn.dataset.name}」？`)) return;
                    await api(`/api/batches/${btn.dataset.id}`, { method: 'DELETE' });
                    loadBatches();
                    toast('批次已删除');
                } catch (err) { toast(err.message); }
            };
        });
    } catch (e) { toast(e.message); }
}
$('#refreshBatches').onclick = loadBatches;

// ---------- 筛选监听 ----------
$('#statusFilter').onchange = () => {
    if (state.batchId) renderReview();
};

// ---------- 辅助函数 ----------
function downloadName(filename) {
    return (filename || 'image').replace(/\.png$/i, '.tif');
}

// ---------- CMYK 曲线面板 ----------
function curvesAreIdentity(curves) {
    return CURVE_KEYS.every(k => {
        const pts = curves[k] || [];
        return pts.length === 2 && pts[0][0] === 0 && pts[0][1] === 0
            && pts[1][0] === 255 && pts[1][1] === 255;
    });
}

function normalizeCurves(raw) {
    const curves = identityCurves();
    if (!raw) return curves;
    CURVE_KEYS.forEach(k => {
        const pts = Array.isArray(raw[k]) ? raw[k] : null;
        if (!pts || pts.length < 2) return;
        curves[k] = pts.map(p => [
            Math.max(0, Math.min(255, Math.round(Number(p[0]) || 0))),
            Math.max(0, Math.min(255, Math.round(Number(p[1]) || 0)))
        ]).sort((a, b) => a[0] - b[0]);
    });
    return curves;
}

function curveGeometry() {
    const canvas = $('#curveCanvas');
    const pad = 18;
    return { canvas, pad, size: canvas.width - pad * 2 };
}

function valueToPixel(v, geo) {
    return { x: geo.pad + (v[0] / 255) * geo.size, y: geo.pad + (1 - v[1] / 255) * geo.size };
}

function pixelToValue(px, py, geo) {
    return [
        Math.max(0, Math.min(255, Math.round(((px - geo.pad) / geo.size) * 255))),
        Math.max(0, Math.min(255, Math.round((1 - (py - geo.pad) / geo.size) * 255)))
    ];
}

// 与后端一致的单调三次插值，用于画出与实际处理相同的曲线
function curveLut(points) {
    const pts = (points || []).slice().sort((a, b) => a[0] - b[0]);
    if (pts.length === 0) return Array.from({ length: 256 }, (_, i) => i);
    const xs = pts.map(p => p[0]);
    const ys = pts.map(p => p[1]);
    if (xs[0] !== 0) { xs.unshift(0); ys.unshift(ys[0]); }
    if (xs[xs.length - 1] !== 255) { xs.push(255); ys.push(ys[ys.length - 1]); }
    const n = xs.length;
    if (n < 3) {
        return Array.from({ length: 256 }, (_, i) => {
            const t = (i - xs[0]) / (xs[1] - xs[0] || 1);
            return Math.max(0, Math.min(255, Math.round(ys[0] + t * (ys[1] - ys[0]))));
        });
    }
    const h = [], delta = [];
    for (let i = 0; i < n - 1; i++) { h.push(xs[i + 1] - xs[i]); delta.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])); }
    const m = new Array(n);
    m[0] = delta[0];
    m[n - 1] = delta[n - 2];
    for (let i = 1; i < n - 1; i++) m[i] = delta[i - 1] * delta[i] <= 0 ? 0 : (delta[i - 1] + delta[i]) / 2;
    for (let i = 0; i < n - 1; i++) {
        if (delta[i] === 0) { m[i] = 0; m[i + 1] = 0; continue; }
        const a = m[i] / delta[i], b = m[i + 1] / delta[i], s = a * a + b * b;
        if (s > 9) { const t = 3 / Math.sqrt(s); m[i] = t * a * delta[i]; m[i + 1] = t * b * delta[i]; }
    }
    const out = new Array(256);
    let seg = 0;
    for (let v = 0; v < 256; v++) {
        while (seg < n - 2 && v >= xs[seg + 1]) seg++;
        const hh = xs[seg + 1] - xs[seg];
        const t = (v - xs[seg]) / hh, t2 = t * t, t3 = t2 * t;
        const val = (2 * t3 - 3 * t2 + 1) * ys[seg] + (t3 - 2 * t2 + t) * hh * m[seg]
            + (-2 * t3 + 3 * t2) * ys[seg + 1] + (t3 - t2) * hh * m[seg + 1];
        out[v] = Math.max(0, Math.min(255, Math.round(val)));
    }
    return out;
}

function drawCurve() {
    const geo = curveGeometry();
    const ctx = geo.canvas.getContext('2d');
    const { pad, size } = geo;
    const full = geo.canvas.width;
    ctx.clearRect(0, 0, full, full);

    ctx.fillStyle = '#fbfcf8';
    ctx.fillRect(pad, pad, size, size);
    ctx.strokeStyle = '#e2e5df';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const p = pad + (size / 4) * i;
        ctx.beginPath(); ctx.moveTo(p, pad); ctx.lineTo(p, pad + size); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, p); ctx.lineTo(pad + size, p); ctx.stroke();
    }
    ctx.strokeStyle = '#cdd2cb';
    ctx.strokeRect(pad, pad, size, size);
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, pad + size); ctx.lineTo(pad + size, pad); ctx.stroke();
    ctx.setLineDash([]);

    // 非当前通道用浅色作参考
    CURVE_KEYS.forEach(key => {
        if (key === state.curveChan) return;
        const pts = state.curves[key];
        if (curvesAreIdentity({ ...identityCurves(), [key]: pts })) return;
        strokeLut(ctx, geo, curveLut(pts), CHANNEL_COLORS[key], 1, 0.28);
    });

    const points = state.curves[state.curveChan];
    const visible = (state.curveDiscard && canDeletePoint(state.curveDrag))
        ? points.filter((_, i) => i !== state.curveDrag)
        : points;
    strokeLut(ctx, geo, curveLut(visible), CHANNEL_COLORS[state.curveChan], 2, 1);

    points.forEach((p, i) => {
        const { x, y } = valueToPixel(p, geo);
        const discarding = state.curveDiscard && i === state.curveDrag && canDeletePoint(i);
        const selected = i === state.curveSelected;
        ctx.beginPath();
        ctx.arc(x, y, selected || i === state.curveDrag ? 6 : 4.5, 0, Math.PI * 2);
        ctx.globalAlpha = discarding ? 0.35 : 1;
        ctx.fillStyle = selected && !discarding ? CHANNEL_COLORS[state.curveChan] : '#fff';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = CHANNEL_COLORS[state.curveChan];
        ctx.stroke();
        ctx.globalAlpha = 1;
        if (discarding) {
            ctx.beginPath();
            ctx.moveTo(x - 5, y - 5); ctx.lineTo(x + 5, y + 5);
            ctx.moveTo(x + 5, y - 5); ctx.lineTo(x - 5, y + 5);
            ctx.stroke();
        }
    });
}

function strokeLut(ctx, geo, lut, color, width, alpha) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    for (let v = 0; v < 256; v++) {
        const { x, y } = valueToPixel([v, lut[v]], geo);
        if (v === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
}

function curveEventValue(e) {
    const geo = curveGeometry();
    const rect = geo.canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (geo.canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (geo.canvas.height / rect.height);
    return { value: pixelToValue(px, py, geo), px, py, geo };
}

function findPointIndex(px, py, geo) {
    const points = state.curves[state.curveChan];
    for (let i = 0; i < points.length; i++) {
        const { x, y } = valueToPixel(points[i], geo);
        if (Math.hypot(x - px, y - py) <= 9) return i;
    }
    return -1;
}

function canDeletePoint(i) {
    const points = state.curves[state.curveChan] || [];
    return Number.isInteger(i) && i > 0 && i < points.length - 1;
}

function graphContains(px, py, geo, margin = 28) {
    return px >= geo.pad - margin && py >= geo.pad - margin
        && px <= geo.pad + geo.size + margin && py <= geo.pad + geo.size + margin;
}

function updateDeleteButton() {
    const btn = $('#curveDeletePoint');
    if (btn) btn.disabled = !canDeletePoint(state.curveSelected);
}

function deleteCurvePoint(i) {
    if (!canDeletePoint(i)) return false;
    state.curves[state.curveChan].splice(i, 1);
    state.curveDrag = null;
    state.curveSelected = null;
    state.curveDiscard = false;
    setReadout(null);
    updateDeleteButton();
    drawCurve();
    scheduleCurvePreview();
    return true;
}

function setReadout(value, discarding = false) {
    const pct = v => `${Math.round((v / 255) * 100)}%`;
    $('#curveReadout').textContent = discarding
        ? '松开以删除此控制点'
        : value
            ? `输入 ${pct(value[0])} → 输出 ${pct(value[1])}`
            : '输入 — → 输出 —';
}

function initCurveCanvas() {
    const canvas = $('#curveCanvas');

    const movePoint = (i, value) => {
        const points = state.curves[state.curveChan];
        let x = value[0];
        if (i === 0) x = 0;
        else if (i === points.length - 1) x = 255;
        else x = Math.max(points[i - 1][0] + 1, Math.min(points[i + 1][0] - 1, x));
        points[i] = [x, value[1]];
        return points[i];
    };

    canvas.onpointerdown = e => {
        if (e.button === 2) {
            const { px, py, geo } = curveEventValue(e);
            const idx = findPointIndex(px, py, geo);
            if (deleteCurvePoint(idx)) e.preventDefault();
            return;
        }
        if (e.button !== 0) return;
        const { value, px, py, geo } = curveEventValue(e);
        let idx = findPointIndex(px, py, geo);
        if (idx < 0) {
            const points = state.curves[state.curveChan];
            points.push(value);
            points.sort((a, b) => a[0] - b[0]);
            idx = points.findIndex(p => p[0] === value[0] && p[1] === value[1]);
        }
        state.curveDrag = idx;
        state.curveSelected = idx;
        state.curveDiscard = false;
        canvas.setPointerCapture(e.pointerId);
        canvas.style.cursor = 'grabbing';
        setReadout(state.curves[state.curveChan][idx]);
        updateDeleteButton();
        drawCurve();
    };

    canvas.onpointermove = e => {
        const { value, px, py, geo } = curveEventValue(e);
        if (state.curveDrag === null) {
            canvas.style.cursor = findPointIndex(px, py, geo) >= 0 ? 'grab' : 'crosshair';
            return;
        }
        const i = state.curveDrag;
        state.curveDiscard = canDeletePoint(i) && !graphContains(px, py, geo);
        if (!state.curveDiscard) movePoint(i, value);
        setReadout(state.curves[state.curveChan][i], state.curveDiscard);
        canvas.style.cursor = state.curveDiscard ? 'not-allowed' : 'grabbing';
        drawCurve();
        if (!state.curveDiscard) scheduleCurvePreview();
    };

    const endDrag = e => {
        if (state.curveDrag === null) return;
        const idx = state.curveDrag;
        const shouldDelete = state.curveDiscard && canDeletePoint(idx);
        state.curveDrag = null;
        state.curveDiscard = false;
        try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* already released */ }
        canvas.style.cursor = 'crosshair';
        if (shouldDelete) {
            deleteCurvePoint(idx);
            return;
        }
        updateDeleteButton();
        drawCurve();
        scheduleCurvePreview();
    };
    canvas.onpointerup = endDrag;
    canvas.onpointercancel = endDrag;
    canvas.oncontextmenu = e => e.preventDefault();

    canvas.ondblclick = e => {
        const { px, py, geo } = curveEventValue(e);
        deleteCurvePoint(findPointIndex(px, py, geo));
    };

    document.addEventListener('keydown', e => {
        if ($('#curveModal').classList.contains('hidden')) return;
        if (e.key !== 'Backspace' && e.key !== 'Delete') return;
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
        if (!canDeletePoint(state.curveSelected)) return;
        e.preventDefault();
        deleteCurvePoint(state.curveSelected);
    });

    document.querySelectorAll('.chan-tab').forEach(btn => {
        btn.onclick = () => {
            state.curveChan = btn.dataset.chan;
            document.querySelectorAll('.chan-tab').forEach(x => x.classList.toggle('active', x === btn));
            state.curveDrag = null;
            state.curveSelected = null;
            state.curveDiscard = false;
            setReadout(null);
            updateDeleteButton();
            drawCurve();
        };
    });

    $('#curveDeletePoint').onclick = () => deleteCurvePoint(state.curveSelected);
    $('#curveResetChan').onclick = () => {
        state.curves[state.curveChan] = [[0, 0], [255, 255]];
        state.curveDrag = null;
        state.curveSelected = null;
        state.curveDiscard = false;
        setReadout(null);
        updateDeleteButton();
        drawCurve();
        scheduleCurvePreview();
    };
    $('#curveResetAll').onclick = () => {
        state.curves = identityCurves();
        state.curveDrag = null;
        state.curveSelected = null;
        state.curveDiscard = false;
        setReadout(null);
        updateDeleteButton();
        drawCurve();
        scheduleCurvePreview();
    };
    $('#curveImage').onchange = () => {
        state.curveImageId = $('#curveImage').value;
        state.curvePageIndex = 0;
        renderCurvePageSelect();
        scheduleCurvePreview(0);
    };
    $('#curvePage').onchange = () => {
        state.curvePageIndex = Number($('#curvePage').value) || 0;
        scheduleCurvePreview(0);
    };
    $('#curveClose').onclick = closeCurveModal;
    $('#curveCancel').onclick = closeCurveModal;
    $('#curveApply').onclick = applyCurves;
    $('#curveModal').onclick = e => { if (e.target.id === 'curveModal') closeCurveModal(); };
}

function openCurveModal() {
    if (!state.batch) return;
    const done = (state.batch.images || []).filter(i => i.process_status === 'completed');
    if (done.length === 0) {
        toast('还没有已完成的输出，无法调整曲线');
        return;
    }
    state.curves = normalizeCurves(state.batch.curves);
    state.curveChan = 'master';
    state.curveDrag = null;
    state.curveSelected = null;
    state.curveDiscard = false;
    document.querySelectorAll('.chan-tab').forEach(x => x.classList.toggle('active', x.dataset.chan === 'master'));
    $('#curveImage').innerHTML = done
        .map(i => `<option value="${i.id}">${escapeHtml(i.filename)}</option>`).join('');
    if (!done.some(i => i.id === state.curveImageId)) {
        state.curveImageId = done[0].id;
        state.curvePageIndex = 0;
    }
    $('#curveImage').value = state.curveImageId;
    renderCurvePageSelect();
    $('#curveModal').classList.remove('hidden');
    setReadout(null);
    updateDeleteButton();
    drawCurve();
    scheduleCurvePreview(0);
}

function renderCurvePageSelect() {
    const item = (state.batch?.images || []).find(i => i.id === state.curveImageId);
    const pages = item ? pageCount(item) : 1;
    const select = $('#curvePage');
    select.classList.toggle('hidden', pages <= 1);
    if (pages <= 1) { state.curvePageIndex = 0; return; }
    select.innerHTML = Array.from({ length: pages },
        (_, n) => `<option value="${n}">第 ${n + 1} / ${pages} 页</option>`).join('');
    state.curvePageIndex = Math.min(state.curvePageIndex || 0, pages - 1);
    select.value = String(state.curvePageIndex);
}

function closeCurveModal() {
    $('#curveModal').classList.add('hidden');
    clearTimeout(state.curvePreviewTimer);
}

function scheduleCurvePreview(delay = 260) {
    clearTimeout(state.curvePreviewTimer);
    state.curvePreviewTimer = setTimeout(refreshCurvePreview, delay);
}

async function refreshCurvePreview() {
    if (!state.batchId || !state.curveImageId) return;
    $('#curveStatus').textContent = '正在生成 CMYK 软打样预览…';
    try {
        const r = await fetch(`/api/batches/${state.batchId}/curves/preview`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: state.curveImageId, curves: state.curves,
                                   size: 900, page: state.curvePageIndex || 0 })
        });
        if (!r.ok) throw new Error((await r.json().catch(() => null))?.error || '预览失败');
        const blob = await r.blob();
        if (state.curvePreviewUrl) URL.revokeObjectURL(state.curvePreviewUrl);
        state.curvePreviewUrl = URL.createObjectURL(blob);
        $('#curvePreview').src = state.curvePreviewUrl;
        $('#curveStatus').textContent = curvesAreIdentity(state.curves)
            ? '当前为交付文件软打样（PSOcoated_v3）'
            : '已套用曲线（交付文件软打样）';
    } catch (e) {
        $('#curveStatus').textContent = e.message;
    }
}

async function applyCurves() {
    if (!state.batchId) return;
    const button = $('#curveApply');
    button.disabled = true;
    try {
        await api(`/api/batches/${state.batchId}/curves`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ curves: state.curves })
        });
        toast('正在对整批重新出图…');
        closeCurveModal();
        pollCurveStatus();
    } catch (e) {
        toast(e.message);
    } finally {
        button.disabled = false;
    }
}

function pollCurveStatus() {
    clearInterval(state.curvePoll);
    state.curvePoll = setInterval(async () => {
        try {
            const b = await api(`/api/batches/${state.batchId}`);
            if (b.curves_status !== 'applying') {
                clearInterval(state.curvePoll);
                await openReview(state.batchId);
                toast('曲线已应用到整批');
            }
        } catch (e) {
            clearInterval(state.curvePoll);
            toast(e.message);
        }
    }, 1500);
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, '&quot;');
}

// ---------- 初始化 ----------
async function init() {
    try {
        const config = await api('/api/config');
        $('#uploadHint').textContent = `可一次选择多张，单张不超过 ${config.max_upload_mb} MB`;
    } catch (e) { /* 非关键 */ }
    initCurveCanvas();
    $('#curveButton').onclick = openCurveModal;
    loadBatches();
    route();
}

window.addEventListener('hashchange', route);
init();