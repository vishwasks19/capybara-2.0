let selectedFile = null;

document.addEventListener('DOMContentLoaded', () => {
    // Drop zone interactions
    const dropZone = document.getElementById('drop-zone');
    const imageUpload = document.getElementById('image-upload');
    
    if (dropZone && imageUpload) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--accent-blue)';
            dropZone.style.backgroundColor = 'rgba(0, 168, 255, 0.05)';
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.backgroundColor = 'transparent';
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.borderColor = 'var(--border-color)';
            dropZone.style.backgroundColor = 'transparent';
            
            if (e.dataTransfer.files.length > 0) {
                imageUpload.files = e.dataTransfer.files;
                handleFileSelection(e.dataTransfer.files[0]);
            }
        });
        
        imageUpload.addEventListener('change', (e) => {
            if(e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });
    }

    // Map controls on analysis page
    const mapBtns = document.querySelectorAll('.map-btn');
    const mapImage = document.getElementById('analysis-image');
    if (mapBtns.length > 0 && mapImage) {
        
        const storedOriginal = sessionStorage.getItem('originalImage');
        const storedMask = sessionStorage.getItem('maskImage');
        const storedOverlay = sessionStorage.getItem('overlayImage');
        
        // Default to overlay if available
        if (storedOverlay) {
            mapImage.src = storedOverlay;
        }

        mapBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                mapBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                const mode = btn.innerText.trim();
                if (mode === 'Original' && storedOriginal) {
                    mapImage.src = storedOriginal;
                } else if (mode === 'Detection Mask' && storedMask) {
                    mapImage.src = storedMask;
                } else if (mode === 'Spill Overlay' && storedOverlay) {
                    mapImage.src = storedOverlay;
                }
            });
        });
    }

    // Vessel list interaction on attribution page
    const vesselCards = document.querySelectorAll('.vessel-card');
    if (vesselCards.length > 0) {
        vesselCards.forEach(card => {
            card.addEventListener('click', () => {
                vesselCards.forEach(c => c.classList.remove('active'));
                card.classList.add('active');
            });
        });
    }
});

function handleFileSelection(file) {
    selectedFile = file;
    const title = document.querySelector('#drop-zone h3');
    if (title) {
        title.innerText = file.name;
        title.style.color = 'var(--accent-blue)';
    }
}

async function analyzeImage() {
    if (!selectedFile) {
        alert("Please select an image file first.");
        return;
    }
    
    const btn = document.getElementById('analyze-btn');
    const originalText = btn.innerText;
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        // Call the Firebase Function backend
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const data = await response.json();
        
        // Save results to sessionStorage
        sessionStorage.setItem('maskImage', "data:image/png;base64," + data.mask);
        sessionStorage.setItem('overlayImage', "data:image/png;base64," + data.overlay);
        
        // Read the original file as data URL to show in 'Original' mode
        const reader = new FileReader();
        reader.onload = (e) => {
            sessionStorage.setItem('originalImage', e.target.result);
            // Redirect to analysis page
            window.location.href = 'analysis.html';
        };
        reader.readAsDataURL(selectedFile);
        
    } catch (error) {
        console.error("Analysis failed:", error);
        alert("Analysis failed. The cloud backend could not process the image.");
        btn.innerText = originalText;
        btn.disabled = false;
    }
}
