document.addEventListener('DOMContentLoaded', () => {
    const detectionForm = document.getElementById('detection-form');
    const generationForm = document.getElementById('generation-form');
    const detectionResult = document.getElementById('detection-result');
    const generationResult = document.getElementById('generation-result');
    const generationPreview = document.getElementById('generation-preview');
    const downloadBtn = document.getElementById('download-btn');
    const detectionFileDisplay = document.getElementById('detection-file-display');
    const generationFileDisplay = document.getElementById('generation-file-display');
    
    let currentVideoUrl = null;

    // Handle file selection display for detection
    document.getElementById('detect-file').addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'No file selected';
        detectionFileDisplay.textContent = fileName;
    });

    // Handle file selection display for generation
    document.getElementById('generate-image').addEventListener('change', (e) => {
        const fileName = e.target.files[0]?.name || 'No file selected';
        generationFileDisplay.textContent = fileName;
    });

    detectionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(detectionForm);
        const file = formData.get('file');
        
        if (!file || file.size === 0) {
            showError(detectionResult, 'Please select a file to analyze');
            return;
        }

        try {
            showLoading(detectionResult);
            
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                const resultHTML = `
                    <div class="success">
                        <p><strong>Result:</strong> ${data.is_fake ? 'Fake' : 'Real'}</p>
                        <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Message:</strong> ${data.message}</p>
                    </div>
                `;
                detectionResult.innerHTML = resultHTML;
            } else {
                showError(detectionResult, data.error || 'An error occurred during detection');
            }
        } catch (error) {
            showError(detectionResult, 'Failed to connect to the server');
        }
    });

    generationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(generationForm);
        const image = formData.get('image');
        const text = formData.get('text');

        if (!image || image.size === 0) {
            showError(generationResult, 'Please select an image');
            return;
        }

        if (!text.trim()) {
            showError(generationResult, 'Please enter some text');
            return;
        }

        try {
            showLoading(generationResult);
            showPreviewLoading();
            
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (response.ok) {
                if (data.video_url) {
                    currentVideoUrl = data.video_url;
                    showPreviewVideo(data.video_url);
                    downloadBtn.style.display = 'block';
                }
                
                const resultHTML = `
                    <div class="success">
                        <p><strong>Status:</strong> ${data.status}</p>
                        <p><strong>Message:</strong> ${data.message}</p>
                    </div>
                `;
                generationResult.innerHTML = resultHTML;
            } else {
                showError(generationResult, data.error || 'An error occurred during generation');
                showPreviewError();
            }
        } catch (error) {
            showError(generationResult, 'Failed to connect to the server');
            showPreviewError();
        }
    });

    downloadBtn.addEventListener('click', () => {
        if (currentVideoUrl) {
            const link = document.createElement('a');
            link.href = currentVideoUrl;
            link.download = 'generated_deepfake.mp4';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    });

    function showError(element, message) {
        element.innerHTML = `
            <div class="error">
                <p>${message}</p>
            </div>
        `;
    }

    function showLoading(element) {
        element.innerHTML = `
            <div class="loading">
                <p>Processing... Please wait...</p>
            </div>
        `;
    }

    function showPreviewLoading() {
        generationPreview.innerHTML = `
            <div class="loading">
                <p>Generating video... Please wait...</p>
            </div>
        `;
        downloadBtn.style.display = 'none';
    }

    function showPreviewError() {
        generationPreview.innerHTML = `
            <div class="error">
                <p>Failed to generate preview</p>
            </div>
        `;
        downloadBtn.style.display = 'none';
    }

    function showPreviewVideo(videoUrl) {
        generationPreview.innerHTML = `
            <video controls>
                <source src="${videoUrl}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        `;
    }
}); 