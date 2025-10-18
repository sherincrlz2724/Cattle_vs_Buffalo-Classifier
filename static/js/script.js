document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData();
    const fileInput = document.getElementById('image');
    const file = fileInput.files[0];
    formData.append('file', file);

    const preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.error) {
            document.getElementById('result').textContent = `Error: ${data.error}`;
        } else {
            const resultText = `Prediction: ${data.prediction.toUpperCase()}, Confidence: ${(data.confidence*100).toFixed(2)}%`;
            document.getElementById('result').textContent = resultText;
        }
    } catch (err) {
        document.getElementById('result').textContent = `Error: ${err}`;
    }
});
