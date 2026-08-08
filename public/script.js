let lastPatientData = null;
let explanationInFlight = false;

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = e.target.querySelector('button');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span>⏳</span> Analyzing...';
    btn.disabled = true;

    // Gather form data
    const formData = new FormData(e.target);
    const data = {
        age: parseInt(formData.get('age')),
        sex: formData.get('sex'),
        cp: parseInt(formData.get('cp')),
        trestbps: parseInt(formData.get('trestbps')),
        chol: parseInt(formData.get('chol')),
        fbs: parseInt(formData.get('fbs')),
        restecg: parseInt(formData.get('restecg')),
        thalach: parseInt(formData.get('thalach')),
        exang: parseInt(formData.get('exang')),
        oldpeak: parseFloat(formData.get('oldpeak')),
        slope: parseInt(formData.get('slope')),
        ca: parseInt(formData.get('ca')),
        thal: parseInt(formData.get('thal'))
    };

    // Reset any previous AI explanation when a new prediction is made.
    const explanationCard = document.getElementById('explanation');
    explanationCard.classList.add('hidden');
    document.getElementById('explanationContent').innerHTML = '';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();

        const resultCard = document.getElementById('result');
        const resultContent = resultCard.querySelector('.result-content');
        const explainBtn = document.getElementById('explainBtn');

        resultCard.classList.remove('hidden', 'success', 'danger');

        const probPercent = (result.probability * 100).toFixed(2);

        if (result.prediction === 1) {
            resultCard.classList.add('danger');
            resultContent.innerHTML = `<h3>⚠️ High Risk of Heart Disease</h3><p>Probability: ${probPercent}%</p>`;
        } else {
            resultCard.classList.add('success');
            resultContent.innerHTML = `<h3>✅ Low Risk of Heart Disease</h3><p>Probability: ${probPercent}%</p>`;
        }

        // Remember the validated patient data so the explanation request
        // reuses exactly what was just predicted on.
        lastPatientData = data;
        explainBtn.classList.remove('hidden');
        explainBtn.disabled = false;
        explainBtn.innerHTML = '<span>✨</span> Generate AI Risk Explanation';

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});

document.getElementById('explainBtn').addEventListener('click', async () => {
    if (explanationInFlight || !lastPatientData) {
        return;
    }
    explanationInFlight = true;

    const explainBtn = document.getElementById('explainBtn');
    const explanationCard = document.getElementById('explanation');
    const explanationContent = document.getElementById('explanationContent');

    explainBtn.disabled = true;
    explainBtn.innerHTML = '<span>⏳</span> Generating explanation...';
    explanationCard.classList.remove('hidden');
    explanationContent.innerHTML = '<p class="explanation-loading">Generating explanation…</p>';

    try {
        const response = await fetch('/api/explain-risk', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(lastPatientData)
        });

        if (!response.ok) {
            throw new Error('Explanation request failed');
        }

        const result = await response.json();
        renderExplanation(result);

    } catch (error) {
        console.error('Error:', error);
        explanationContent.innerHTML = renderUnavailableMessage();
    } finally {
        explainBtn.disabled = false;
        explainBtn.innerHTML = '<span>✨</span> Regenerate AI Risk Explanation';
        explanationInFlight = false;
    }
});

function renderUnavailableMessage() {
    return `<p class="explanation-unavailable">AI explanation is temporarily unavailable. Your risk prediction is still available above.</p>`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function renderList(items) {
    if (!items || items.length === 0) {
        return '';
    }
    return `<ul>${items.map(item => `<li>${escapeHtml(String(item))}</li>`).join('')}</ul>`;
}

function renderExplanation(result) {
    const explanationContent = document.getElementById('explanationContent');

    if (!result.explanation_available) {
        explanationContent.innerHTML = renderUnavailableMessage();
        return;
    }

    let html = '';

    if (result.risk_category) {
        html += `<p class="explanation-summary"><strong>Risk category:</strong> ${escapeHtml(result.risk_category)}</p>`;
    }
    if (typeof result.probability === 'number') {
        html += `<p class="explanation-summary"><strong>Probability:</strong> ${(result.probability * 100).toFixed(2)}%</p>`;
    }
    if (result.summary) {
        html += `<h4>Summary</h4><p>${escapeHtml(result.summary)}</p>`;
    }
    if (result.input_factors && result.input_factors.length > 0) {
        html += `<h4>Relevant Factors</h4>${renderList(result.input_factors)}`;
    }
    if (result.educational_information && result.educational_information.length > 0) {
        html += `<h4>Educational Information</h4>${renderList(result.educational_information)}`;
    }
    if (result.questions_for_professional && result.questions_for_professional.length > 0) {
        html += `<h4>Questions You Can Ask a Healthcare Professional</h4>${renderList(result.questions_for_professional)}`;
    }
    if (result.citations && result.citations.length > 0) {
        const citationItems = result.citations.map(c => {
            const title = escapeHtml(c.title || 'Source');
            return c.uri
                ? `<li><a href="${escapeHtml(c.uri)}" target="_blank" rel="noopener noreferrer">${title}</a></li>`
                : `<li>${title}</li>`;
        }).join('');
        html += `<h4>Sources</h4><ul>${citationItems}</ul>`;
    }
    if (result.disclaimer) {
        html += `<p class="explanation-disclaimer">${escapeHtml(result.disclaimer)}</p>`;
    }

    explanationContent.innerHTML = html;
}
