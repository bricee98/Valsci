let pendingFormSubmission = null;
let pendingFormData = null;

function showPasswordModal(formData, submitFunction) {
    const modal = document.getElementById('passwordModal');
    const passwordInput = document.getElementById('passwordInput');
    
    pendingFormSubmission = submitFunction;
    pendingFormData = formData;
    
    modal.style.display = 'block';
    passwordInput.value = '';
    passwordInput.focus();
}

document.getElementById('submitPassword').addEventListener('click', () => {
    const password = document.getElementById('passwordInput').value;
    const modal = document.getElementById('passwordModal');
    
    if (pendingFormSubmission && pendingFormData) {
        pendingFormData.append('password', password);
        pendingFormSubmission(pendingFormData);
    }
    
    modal.style.display = 'none';
});

document.getElementById('cancelPassword').addEventListener('click', () => {
    const modal = document.getElementById('passwordModal');
    modal.style.display = 'none';
    pendingFormSubmission = null;
    pendingFormData = null;
});

document.addEventListener('DOMContentLoaded', function() {
    const claimForm = document.getElementById('claimForm');
    const fileForm = document.getElementById('fileForm');
    const referenceForm = document.getElementById('referenceForm');

    claimForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const claimText = document.getElementById('claimText').value;
        
        const formData = new FormData();
        formData.append('text', claimText);
        
        if (requirePassword) {
            showPasswordModal(formData, async (data) => {
                const response = await fetch('/api/v1/claims', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(Object.fromEntries(data)),
                });
                window.location.href = `/progress?claim_id=${response.claim_id}`;
            });
        } else {
            const response = await fetch('/api/v1/claims', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: claimText }),
            });
            window.location.href = `/progress?claim_id=${response.claim_id}`;
        }
    });

    fileForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('claimFile');
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        
        if (requirePassword) {
            showPasswordModal(formData, async (data) => {
                const response = await fetch('/api/v1/batch', {
                    method: 'POST',
                    body: data,
                });
                window.location.href = `/progress?batch_id=${response.batch_id}`;
            });
        } else {
            const response = await fetch('/api/v1/batch', {
                method: 'POST',
                body: formData,
            });
            window.location.href = `/progress?batch_id=${response.batch_id}`;
        }
    });

    referenceForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const referenceID = document.getElementById('claimReferenceID').value;
        checkStatus(referenceID);
    });

    function checkStatus(referenceID) {
        window.location.href = `/progress?${referenceID.length === 8 ? 'claim_id' : 'batch_id'}=${referenceID}`;
    }
});
