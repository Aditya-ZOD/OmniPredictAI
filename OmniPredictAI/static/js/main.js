/**
 * OmniPredict AI - Main Client-side Interactions
 */
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 OmniPredict AI Initialized Successfully.');

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
});
