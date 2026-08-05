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
        
        resultCard.classList.remove('hidden', 'success', 'danger');
        
        const probPercent = (result.probability * 100).toFixed(2);

        if (result.prediction === 1) {
            resultCard.classList.add('danger');
            resultContent.innerHTML = `<h3>⚠️ High Risk of Heart Disease</h3><p>Probability: ${probPercent}%</p>`;
        } else {
            resultCard.classList.add('success');
            resultContent.innerHTML = `<h3>✅ Low Risk of Heart Disease</h3><p>Probability: ${probPercent}%</p>`;
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while making the prediction. Please try again.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
});
