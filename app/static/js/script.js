/* ==========================================================================
   LoanPulse AI — Frontend Interaction Controller
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('loan-form');
    const btnSubmit = document.getElementById('btn-submit');
    const btnThemeToggle = document.getElementById('btn-theme-toggle');
    const btnMetricsModal = document.getElementById('btn-metrics-modal');
    const modalBenchmarks = document.getElementById('modal-benchmarks');
    const btnCloseModal = document.getElementById('btn-close-modal');
    const presetButtons = document.querySelectorAll('.pill-btn');

    // Result Elements
    const gaugeFill = document.getElementById('gauge-fill');
    const probValue = document.getElementById('prob-value');
    const statusChip = document.getElementById('status-chip');
    const statusText = document.getElementById('status-text');
    const riskChip = document.getElementById('risk-chip');
    const riskText = document.getElementById('risk-text');
    const modelTag = document.getElementById('model-tag');
    const factorsList = document.getElementById('factors-list');

    // Summary Tiles
    const summaryIncome = document.getElementById('summary-income');
    const summaryEmi = document.getElementById('summary-emi');
    const summaryDti = document.getElementById('summary-dti');
    const summaryLoan = document.getElementById('summary-loan');
    const toast = document.getElementById('toast');

    // SVG Gauge Circumference for r=50
    const CIRCUMFERENCE = 2 * Math.PI * 50; // ~314.159

    // Sample Presets Database
    const PRESETS = {
        prime: {
            applicant_income: 6500,
            coapplicant_income: 2500,
            loan_amount: 140,
            loan_term: 360,
            credit_history: 1,
            property_area: 'Semiurban',
            education: 'Graduate',
            married: 'Yes',
            dependents: '1',
            self_employed: 'No'
        },
        starter: {
            applicant_income: 4200,
            coapplicant_income: 0,
            loan_amount: 110,
            loan_term: 360,
            credit_history: 1,
            property_area: 'Urban',
            education: 'Graduate',
            married: 'No',
            dependents: '0',
            self_employed: 'No'
        },
        overleveraged: {
            applicant_income: 2200,
            coapplicant_income: 0,
            loan_amount: 250,
            loan_term: 180,
            credit_history: 1,
            property_area: 'Rural',
            education: 'Not Graduate',
            married: 'Yes',
            dependents: '3',
            self_employed: 'Yes'
        },
        adverse_credit: {
            applicant_income: 5800,
            coapplicant_income: 1200,
            loan_amount: 130,
            loan_term: 360,
            credit_history: 0,
            property_area: 'Urban',
            education: 'Graduate',
            married: 'No',
            dependents: '0',
            self_employed: 'No'
        }
    };

    // 1. Theme Toggle
    const savedTheme = localStorage.getItem('loanpulse_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    btnThemeToggle.addEventListener('click', () => {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('loanpulse_theme', newTheme);
    });

    // 2. Preset Click Handlers
    presetButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            presetButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.getAttribute('data-preset');
            const data = PRESETS[presetKey];
            if (data) {
                applyFormData(data);
                showToast(`Applied preset: ${btn.textContent.trim()}`);
                submitEvaluation();
            }
        });
    });

    function applyFormData(data) {
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) {
                el.value = value;
            }
        }
    }

    // 3. Form Submission
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        submitEvaluation();
    });

    async function submitEvaluation() {
        const formData = {
            applicant_income: parseFloat(document.getElementById('applicant_income').value) || 0,
            coapplicant_income: parseFloat(document.getElementById('coapplicant_income').value) || 0,
            loan_amount: parseFloat(document.getElementById('loan_amount').value) || 0,
            loan_term: parseFloat(document.getElementById('loan_term').value) || 360,
            credit_history: parseFloat(document.getElementById('credit_history').value),
            property_area: document.getElementById('property_area').value,
            education: document.getElementById('education').value,
            married: document.getElementById('married').value,
            dependents: document.getElementById('dependents').value,
            self_employed: document.getElementById('self_employed').value,
            engine: document.querySelector('input[name="engine"]:checked').value
        };

        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<span>Evaluating Risk...</span>`;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            const result = await response.json();
            if (result.status === 'success') {
                renderResults(result.data);
            } else {
                showToast(`Error: ${result.message || 'Evaluation failed'}`);
            }
        } catch (error) {
            console.error(error);
            showToast('Network error while predicting loan status.');
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `
                <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
                <span>Evaluate Application</span>
            `;
        }
    }

    // 4. Render Results in Live Panel
    function renderResults(data) {
        const isApproved = data.is_approved;
        const prob = data.approval_probability;

        // Gauge update
        const offset = CIRCUMFERENCE - (prob / 100) * CIRCUMFERENCE;
        gaugeFill.style.strokeDashoffset = offset;
        probValue.textContent = `${Math.round(prob)}%`;

        if (prob >= 70) {
            gaugeFill.style.stroke = 'var(--success)';
        } else if (prob >= 45) {
            gaugeFill.style.stroke = 'var(--warning)';
        } else {
            gaugeFill.style.stroke = 'var(--danger)';
        }

        // Status Chip
        if (isApproved) {
            statusChip.className = 'status-chip chip-approved';
            statusText.textContent = 'LOAN APPROVED';
        } else {
            statusChip.className = 'status-chip chip-rejected';
            statusText.textContent = 'LOAN REJECTED';
        }

        // Risk Chip
        riskText.textContent = data.risk_level;
        riskText.style.color = data.risk_color;

        // Model Tag
        modelTag.textContent = `Engine: ${data.model_used}`;

        // Financial Summary Tiles
        const fin = data.financial_summary;
        summaryIncome.textContent = `$${fin.total_income.toLocaleString()} / mo`;
        summaryEmi.textContent = `$${fin.monthly_emi.toLocaleString()} / mo`;
        summaryDti.textContent = `${fin.dti_ratio}%`;
        summaryLoan.textContent = `$${fin.loan_amount.toLocaleString()}`;

        // Factors List
        factorsList.innerHTML = '';
        if (data.decision_factors && data.decision_factors.length > 0) {
            data.decision_factors.forEach(f => {
                const item = document.createElement('div');
                const statusClass = f.status === 'Positive' ? 'positive' : (f.status === 'Negative' ? 'negative' : 'neutral');
                const icon = f.status === 'Positive' ? '✔' : (f.status === 'Negative' ? '✖' : 'ℹ');

                item.className = `factor-item ${statusClass}`;
                item.innerHTML = `
                    <div class="factor-icon">${icon}</div>
                    <div><strong>${f.factor}:</strong> ${f.detail}</div>
                `;
                factorsList.appendChild(item);
            });
        }
    }

    // 5. Benchmark Modal Handling
    btnMetricsModal.addEventListener('click', async () => {
        modalBenchmarks.classList.add('open');
        try {
            const res = await fetch('/api/metrics');
            const json = await res.json();
            if (json.status === 'success' && json.data.models_benchmarked) {
                renderBenchmarksTable(json.data.models_benchmarked, json.data.best_model_name);
            }
        } catch (e) {
            console.error('Failed to load metrics:', e);
        }
    });

    btnCloseModal.addEventListener('click', () => {
        modalBenchmarks.classList.remove('open');
    });

    modalBenchmarks.addEventListener('click', (e) => {
        if (e.target === modalBenchmarks) {
            modalBenchmarks.classList.remove('open');
        }
    });

    function renderBenchmarksTable(models, bestName) {
        const tbody = document.getElementById('benchmark-tbody');
        tbody.innerHTML = '';

        // Sort by accuracy descending
        const sorted = Object.entries(models).sort((a, b) => b[1].accuracy - a[1].accuracy);

        sorted.forEach(([name, m]) => {
            const tr = document.createElement('tr');
            const isBest = name === bestName;
            tr.innerHTML = `
                <td><strong>${name}</strong> ${isBest ? '⭐ (Active)' : ''}</td>
                <td>${m.type || 'Classical ML'}</td>
                <td>${(m.accuracy * 100).toFixed(2)}%</td>
                <td>${m.f1_score.toFixed(4)}</td>
                <td>${m.roc_auc.toFixed(4)}</td>
                <td>${(m.cv_accuracy_mean * 100).toFixed(2)}%</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Helper Toast
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add('show');
        setTimeout(() => toast.classList.remove('show'), 3000);
    }

    // Initial evaluation on page load
    submitEvaluation();
});
