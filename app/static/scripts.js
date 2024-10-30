document.addEventListener('DOMContentLoaded', function() {
    const claimForm = document.getElementById('claimForm');
    const fileForm = document.getElementById('fileForm');
    const referenceForm = document.getElementById('referenceForm');

    claimForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const claimText = document.getElementById('claimText').value;
        const password = document.getElementById('claimPassword') ? document.getElementById('claimPassword').value : null;
        submitClaim(claimText, password);
    });

    fileForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const fileInput = document.getElementById('claimFile');
        const file = fileInput.files[0];
        if (file) {
            submitFile(file);
        } else {
            alert('Please select a file to upload.');
        }
    });

    referenceForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const referenceID = document.getElementById('claimReferenceID').value;
        checkStatus(referenceID);
    });

    function submitClaim(claimText, password) {
        const body = { text: claimText };
        if (requirePassword && password) {
            body.password = password;
        }

        fetch('/api/v1/claims', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        })
        .then(response => response.json())
        .then(data => {
            if (data.claim_id) {
                window.location.href = `/progress?claim_id=${data.claim_id}`;
            } else {
                console.error('Claim ID is undefined:', data);
                alert('There was an error processing your claim. Please try again.');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('There was an error processing your claim. Please try again.');
        });
    }

    function submitFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        if (requirePassword) {
            const password = document.getElementById('batchPassword') ? document.getElementById('batchPassword').value : null;
            if (password) {
                formData.append('password', password);
            }
        }

        fetch('/api/v1/batch', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            window.location.href = `/progress?batch_id=${data.batch_id}`;
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    }

    function checkStatus(referenceID) {
        window.location.href = `/progress?${referenceID.length === 8 ? 'claim_id' : 'batch_id'}=${referenceID}`;
    }
});
