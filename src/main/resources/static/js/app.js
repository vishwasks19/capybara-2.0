document.addEventListener('DOMContentLoaded', () => {
    // Drop zone interactions
    const dropZone = document.getElementById('drop-zone');
    
    if (dropZone) {
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
                alert(`File selected: ${e.dataTransfer.files[0].name}. Ready for analysis!`);
            }
        });
        
        dropZone.addEventListener('click', () => {
            // Simulate file dialog opening
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/png, image/jpeg, image/tiff';
            input.onchange = e => {
                if(e.target.files.length > 0) {
                    alert(`File selected: ${e.target.files[0].name}. Ready for analysis!`);
                }
            };
            input.click();
        });
    }

    // Map controls on analysis page
    const mapBtns = document.querySelectorAll('.map-btn');
    if (mapBtns.length > 0) {
        mapBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                mapBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
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
