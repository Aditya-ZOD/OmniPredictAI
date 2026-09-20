/**
 * OmniPredict AI - Main Client-side Interactions
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 OmniPredict AI Initialized Successfully.');

    const escapeHtml = value => String(value).replace(/[&<>'"]/g, character => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[character]));

    const root = document.documentElement;
    const body = document.body;
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle?.querySelector('.theme-icon');

    const applyTheme = (theme) => {
        root.setAttribute('data-theme', theme);
        body.setAttribute('data-theme', theme);
        root.style.colorScheme = theme;
        if (themeIcon) {
            themeIcon.className = theme === 'light' ? 'bi bi-moon-fill theme-icon' : 'bi bi-sun-fill theme-icon';
        }
        if (themeToggle) {
            themeToggle.setAttribute('aria-label', theme === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
        }
    };

    const savedTheme = localStorage.getItem('theme') || (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
    applyTheme(savedTheme);

    themeToggle?.addEventListener('click', () => {
        const nextTheme = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
        localStorage.setItem('theme', nextTheme);
        applyTheme(nextTheme);
    });

    // ── Homepage: Animate prediction confidence chart bars ──────────────────

    const chartBars = document.querySelectorAll('.mini-chart-graphic .bar');
    if (chartBars.length > 0) {
        setInterval(() => {
            chartBars.forEach((bar) => {
                if (Math.random() > 0.4) {
                    const current   = parseInt(bar.style.height) || 50;
                    const variation = Math.floor(Math.random() * 30) - 15;
                    const newH      = Math.max(10, Math.min(100, current + variation));
                    bar.style.height = `${newH}%`;
                }
            });
        }, 2000);
    }

    // ── Homepage: Feature card icon scale on hover ───────────────────────────
    document.querySelectorAll('.feature-card').forEach(card => {
        const icon = card.querySelector('.feature-icon');
        if (!icon) return;
        card.addEventListener('mouseenter', () => {
            icon.style.transform  = 'scale(1.15) rotate(5deg)';
            icon.style.transition = 'transform 0.3s ease';
        });
        card.addEventListener('mouseleave', () => {
            icon.style.transform = 'scale(1) rotate(0deg)';
        });
    });

    // ── Dashboard: Drag & Drop Upload ────────────────────────────────────────
    const dropZone          = document.getElementById('dropZone');
    const fileInput         = document.getElementById('fileInput');
    const browseBtn         = document.getElementById('browseBtn');
    const changeFileBtn     = document.getElementById('changeFileBtn');
    const dropZoneContent   = document.getElementById('dropZoneContent');
    const fileSelectedPrev  = document.getElementById('fileSelectedPreview');
    const selectedFileName  = document.getElementById('selectedFileName');
    const selectedFileSize  = document.getElementById('selectedFileSize');
    const uploadSubmitBtn   = document.getElementById('uploadSubmitBtn');
    const uploadForm        = document.getElementById('uploadForm');
    const uploadBtnText     = document.getElementById('uploadBtnText');
    const uploadSpinner     = document.getElementById('uploadSpinner');

    // ── Global: Auto-dismiss flash alerts after 5 seconds ────────────────────
    document.querySelectorAll('.alert').forEach(alert => {
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
            bsAlert.close();
        }, 5000);
    });

    // ── Global: Loading states for action buttons ─────────────────────────────
    document.querySelectorAll('[data-loading-trigger]').forEach(button => {
        button.addEventListener('click', () => {
            const loader = button.querySelector('.loading-spinner');
            if (loader) loader.classList.remove('d-none');
        });
    });

    // ── Dashboard: Drag & Drop Upload ─────────────────────────────────────────
    if (dropZone) {
        function formatBytes(bytes) {
            if (bytes < 1024)          return bytes + ' B';
            if (bytes < 1024 * 1024)   return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        function showFilePreview(file) {
            selectedFileName.textContent  = file.name;
            selectedFileSize.textContent  = formatBytes(file.size);
            dropZoneContent.classList.add('d-none');
            fileSelectedPrev.classList.remove('d-none');
            uploadSubmitBtn.disabled      = false;
        }

        function resetDropZone() {
            dropZoneContent.classList.remove('d-none');
            fileSelectedPrev.classList.add('d-none');
            uploadSubmitBtn.disabled = true;
            fileInput.value          = '';
        }

        // Click on drop zone triggers file browser
        dropZone.addEventListener('click', (e) => {
            if (e.target === changeFileBtn || changeFileBtn?.contains(e.target)) return;
            fileInput.click();
        });

        browseBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });

        changeFileBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            resetDropZone();
            fileInput.click();
        });

        fileInput?.addEventListener('change', () => {
            if (fileInput.files.length > 0) showFilePreview(fileInput.files[0]);
        });

        // Drag events
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        ['dragleave', 'dragend'].forEach(evt => {
            dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (!file) return;

            if (!file.name.endsWith('.csv')) {
                alert('Only CSV files are supported.');
                return;
            }

            // Inject file into the hidden input via DataTransfer
            const dt = new DataTransfer();
            dt.items.add(file);
            fileInput.files = dt.files;
            showFilePreview(file);
        });

        // On submit — show spinner, disable button
        uploadForm?.addEventListener('submit', () => {
            uploadSubmitBtn.disabled       = true;
            uploadBtnText.classList.add('d-none');
            uploadSpinner.classList.remove('d-none');
        });
    } // end dashboard block

    // ── Configure Target Page ─────────────────────────────────────────────────
    const targetColInput   = document.getElementById('targetColInput');
    const targetDisplay    = document.getElementById('targetDisplay');
    const saveConfigBtn    = document.getElementById('saveConfigBtn');
    const inputFeatCount   = document.getElementById('inputFeatCount');
    const inputFeatList    = document.getElementById('inputFeatList');
    const outputFeatName   = document.getElementById('outputFeatName');
    const outputFeatType   = document.getElementById('outputFeatType');
    const colSearch        = document.getElementById('colSearch');

    if (targetColInput && typeof COLUMNS !== 'undefined') {

        function getExcluded() {
            return [...document.querySelectorAll('.exclude-check:checked')].map(c => c.dataset.col);
        }

        function updateSummaryCards(target) {
            if (!target) return;
            const excluded  = getExcluded();
            const inputCols = COLUMNS.filter(c => c !== target && !excluded.includes(c));
            inputFeatCount.textContent = inputCols.length;
            inputFeatList.textContent  = inputCols.slice(0, 6).join(', ') + (inputCols.length > 6 ? ` +${inputCols.length - 6} more` : '');
            outputFeatName.textContent = target;
            const stats = COL_STATS[target] || {};
            outputFeatType.textContent = stats.dtype ? `${stats.dtype} · ${stats.unique} unique values` : '';
        }

        function setRowRole(col, role) {
            document.querySelectorAll('.feature-row').forEach(row => {
                const rCol = row.querySelector('[data-col]')?.dataset?.col;
                if (!rCol) return;
                const roleEl = row.querySelector('.role-indicator');
                if (rCol === col && role === 'target') {
                    row.className = row.className.replace(/row-\w+/g, '') + ' row-target';
                    if (roleEl) roleEl.innerHTML = `<span class="badge bg-primary rounded-pill px-2"><i class="bi bi-crosshair"></i></span>`;
                } else if (role === 'target' && rCol !== col) {
                    // demote old targets back to feature (unless excluded)
                    const excCheck = row.querySelector('.exclude-check');
                    if (excCheck && excCheck.checked) {
                        row.className = row.className.replace(/row-\w+/g, '') + ' row-excluded';
                        if (roleEl) roleEl.innerHTML = `<span class="badge bg-secondary rounded-pill px-2"><i class="bi bi-dash-circle"></i></span>`;
                    } else {
                        row.className = row.className.replace(/row-\w+/g, '') + ' row-feature';
                        if (roleEl) roleEl.innerHTML = `<span class="badge bg-success rounded-pill px-2"><i class="bi bi-arrow-right-circle"></i></span>`;
                    }
                }
            });
        }

        // "Set target" button clicks
        document.addEventListener('click', e => {
            const btn = e.target.closest('.btn-set-target');
            if (!btn) return;
            const col = btn.dataset.col;
            targetColInput.value = col;

            // Update display
            targetDisplay.innerHTML = `<span class="fw-bold text-light">${col}</span>`;
            if (saveConfigBtn) saveConfigBtn.disabled = false;

            // Uncheck exclude if this col was excluded
            const excCheck = document.querySelector(`.exclude-check[data-col="${col}"]`);
            if (excCheck) excCheck.checked = false;

            setRowRole(col, 'target');
            updateSummaryCards(col);
        });

        // Exclude checkbox toggles
        document.addEventListener('change', e => {
            if (!e.target.classList.contains('exclude-check')) return;
            const col    = e.target.dataset.col;
            const target = targetColInput.value;
            const row    = e.target.closest('.feature-row');
            const roleEl = row?.querySelector('.role-indicator');

            // Prevent excluding the target column
            if (col === target && e.target.checked) {
                e.target.checked = false;
                return;
            }

            if (e.target.checked) {
                row?.classList.replace('row-feature', 'row-excluded');
                if (roleEl) roleEl.innerHTML = `<span class="badge bg-secondary rounded-pill px-2"><i class="bi bi-dash-circle"></i></span>`;
            } else {
                row?.classList.replace('row-excluded', 'row-feature');
                if (roleEl) roleEl.innerHTML = `<span class="badge bg-success rounded-pill px-2"><i class="bi bi-arrow-right-circle"></i></span>`;
            }
            updateSummaryCards(target);
        });

        // Column search filter
        colSearch?.addEventListener('input', () => {
            const q = colSearch.value.toLowerCase();
            document.querySelectorAll('.feature-row').forEach(row => {
                row.style.display = row.dataset.col.includes(q) ? '' : 'none';
            });
        });

        // Init summary cards with existing saved target
        const savedTarget = targetColInput.value;
        if (savedTarget) updateSummaryCards(savedTarget);

    } // end configure-target block

    // ── Full dataset row details ────────────────────────────────────────────
    const rowDetailsForm = document.querySelector('[data-row-details-form]');
    if (rowDetailsForm) {
        const rowInput = rowDetailsForm.querySelector('.row-number-input');
        const modalElement = document.getElementById('rowDetailsModal');
        const modalTitle = document.getElementById('rowDetailsTitle');
        const modalBody = document.getElementById('rowDetailsBody');
        const inlineDetails = document.querySelector('[data-row-details-inline]');
        const datasetId = rowDetailsForm.dataset.datasetId;
        const modal = window.bootstrap?.Modal.getOrCreateInstance(modalElement);

        const renderMatches = matches => matches.map(match => `
            <h6 class="text-light border-bottom border-secondary-subtle pb-2 mb-2">Row ${match.row_number}</h6>
            ${Object.entries(match.values).map(([column, value]) => `
                <div class="row border-bottom border-secondary-subtle py-2">
                    <div class="col-sm-4 text-secondary-custom small fw-semibold">${escapeHtml(column)}</div>
                    <div class="col-sm-8 text-light small text-break">${escapeHtml(value == null || value === '' ? '(empty)' : String(value))}</div>
                </div>
            `).join('')}
        `).join('<hr class="border-secondary-subtle my-3">');

        rowDetailsForm.addEventListener('submit', async event => {
            event.preventDefault();
            const query = rowInput.value.trim();
            if (!datasetId || !query) {
                modalBody.innerHTML = '<div class="text-danger">Enter a row number or search text.</div>';
                modal?.show();
                return;
            }
            modalTitle.textContent = 'Row details';
            modalBody.innerHTML = '<div class="text-secondary-custom">Finding complete row...</div>';
            if (modal) modal.show();
            try {
                const response = await fetch(`/dataset/${datasetId}/row-search?q=${encodeURIComponent(query)}`);
                const contentType = response.headers.get('content-type') || '';
                if (!contentType.includes('application/json')) {
                    throw new Error('Your login session expired. Refresh the page and try again.');
                }
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || 'Could not load row.');
                const rendered = renderMatches(result.matches);
                modalBody.innerHTML = rendered;
                if (inlineDetails) {
                    inlineDetails.innerHTML = rendered;
                    inlineDetails.classList.remove('d-none');
                }
            } catch (error) {
                const message = `<div class="text-danger">${escapeHtml(error.message)}</div>`;
                modalBody.innerHTML = message;
                if (inlineDetails) {
                    inlineDetails.innerHTML = message;
                    inlineDetails.classList.remove('d-none');
                }
            }
        });
    }

    // ── Dataset Copilot ─────────────────────────────────────────────────────
    document.querySelectorAll('[data-assistant-dataset]').forEach(panel => {
        const datasetId = panel.dataset.assistantDataset;
        const form = panel.querySelector('[data-assistant-form]');
        const input = panel.querySelector('[data-assistant-input]');
        const messages = panel.querySelector('[data-assistant-messages]');
        const status = panel.querySelector('[data-assistant-status]');
        const send = panel.querySelector('[data-assistant-send]');
        const mic = panel.querySelector('[data-assistant-mic]');
        const speech = panel.querySelector('[data-assistant-speech]');
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        let readAnswersAloud = false;
        let recognition = null;

        const addMessage = (text, bot = true) => {
            const item = document.createElement('div');
            item.className = `assistant-message ${bot ? 'assistant-message-bot' : 'assistant-message-user'}`;
            item.textContent = text;
            messages.appendChild(item);
            messages.scrollTop = messages.scrollHeight;
        };

        const setBusy = (busy) => {
            send.disabled = busy;
            input.disabled = busy;
            mic.disabled = busy;
            panel.querySelectorAll('[data-assistant-prompt]').forEach(prompt => {
                prompt.disabled = busy;
            });
            status.textContent = busy ? 'Analyzing the complete dataset...' : '';
        };

        const speak = text => {
            if (!readAnswersAloud || !('speechSynthesis' in window) || !text) return;
            window.speechSynthesis.cancel();
            window.speechSynthesis.resume();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = document.documentElement.lang || 'en-US';
            utterance.onstart = () => { status.textContent = 'Reading answer aloud...'; };
            utterance.onend = () => { status.textContent = ''; };
            utterance.onerror = () => { status.textContent = 'The browser could not read this answer aloud.'; };
            window.speechSynthesis.speak(utterance);
        };

        speech.addEventListener('click', () => {
            readAnswersAloud = !readAnswersAloud;
            speech.setAttribute('aria-pressed', String(readAnswersAloud));
            speech.classList.toggle('assistant-speech-active', readAnswersAloud);
            if (!readAnswersAloud && 'speechSynthesis' in window) {
                window.speechSynthesis.cancel();
                status.textContent = '';
                return;
            }
            const latestAnswer = panel.querySelector('.assistant-message-bot:last-of-type');
            speak(latestAnswer?.textContent.trim());
        });

        if (!('speechSynthesis' in window)) {
            speech.disabled = true;
            speech.title = 'Spoken answers are not supported in this browser';
        }

        if (!SpeechRecognition) {
            mic.disabled = true;
            mic.title = 'Voice input is not supported in this browser';
        } else {
            recognition = new SpeechRecognition();
            recognition.lang = document.documentElement.lang || 'en-US';
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.addEventListener('start', () => {
                mic.classList.add('assistant-listening');
                mic.setAttribute('aria-label', 'Stop voice question');
                status.textContent = 'Listening...';
            });
            recognition.addEventListener('result', event => {
                input.value = event.results[0][0].transcript;
                input.focus();
                status.textContent = 'Voice question captured. Press send to ask it.';
            });
            recognition.addEventListener('error', event => {
                status.textContent = event.error === 'not-allowed'
                    ? 'Microphone permission was denied.'
                    : 'Voice input could not be captured.';
            });
            recognition.addEventListener('end', () => {
                mic.classList.remove('assistant-listening');
                mic.setAttribute('aria-label', 'Start voice question');
            });
            mic.addEventListener('click', () => {
                try { recognition.start(); } catch (error) { recognition.stop(); }
            });
        }

        panel.querySelectorAll('[data-assistant-prompt]').forEach(prompt => {
            prompt.addEventListener('click', () => {
                input.value = prompt.dataset.assistantPrompt;
                if (typeof form.requestSubmit === 'function') {
                    form.requestSubmit();
                } else {
                    form.dispatchEvent(new Event('submit', {cancelable: true}));
                }
            });
        });

        form.addEventListener('submit', async event => {
            event.preventDefault();
            const question = input.value.trim();
            if (!question) return;
            addMessage(question, false);
            input.value = '';
            setBusy(true);
            try {
                const response = await fetch(`/dataset/${datasetId}/assistant`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question})
                });
                const result = await response.json();
                if (!response.ok) throw new Error(result.error || 'Assistant request failed.');
                const answer = result.message || 'I could not form an answer.';
                addMessage(answer);
                speak(answer);
                if (result.requires_confirmation && result.changes?.length) {
                    const preview = result.changes.map(change =>
                        `Row ${change.row_number}, ${change.column}: ${change.old_value ?? '(empty)'} -> ${change.new_value}`
                    ).join('\n');
                    addMessage(`Update preview:\n${preview}`);
                    const confirm = document.createElement('button');
                    confirm.className = 'btn btn-sm btn-warning-soft rounded-pill mt-2';
                    confirm.textContent = 'Confirm update';
                    confirm.addEventListener('click', async () => {
                        confirm.disabled = true;
                        try {
                            const saved = await fetch(`/dataset/${datasetId}/assistant/confirm`, {method: 'POST'});
                            const savedResult = await saved.json();
                            addMessage(savedResult.message || savedResult.error || 'Update failed.');
                            if (saved.ok) confirm.remove();
                        } catch (error) {
                            addMessage(error.message || 'Update failed.');
                            confirm.disabled = false;
                        }
                    });
                    messages.appendChild(confirm);
                }
            } catch (error) {
                addMessage(error.message || 'Assistant request failed.');
            } finally {
                setBusy(false);
                input.focus();
            }
        });
    });
});
