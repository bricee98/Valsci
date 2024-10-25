document.getElementById('claimForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const claimText = document.getElementById('claimText').value;
    fetch('/api/v1/claims', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: claimText })
    })
    .then(response => response.json())
    .then(data => {
        if (data.claim_id) {
            window.location.href = `/progress?claim_id=${data.claim_id}`;
        }
    });
});

document.getElementById('fileForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const fileInput = document.getElementById('claimFile');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    fetch('/api/v1/batch', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.job_id) {
            window.location.href = `/progress?job_id=${data.job_id}`;
        }
    });
});

document.getElementById('referenceForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const claimReferenceID = document.getElementById('claimReferenceID').value;
    window.location.href = `/results?claim_id=${claimReferenceID}`;
});
