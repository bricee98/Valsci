document.addEventListener('DOMContentLoaded', function() {
    const claimForm = document.getElementById('claimForm');
    const fileForm = document.getElementById('fileForm');
    const referenceForm = document.getElementById('referenceForm');

    const enhanceClaimBtn = document.getElementById('enhanceClaimBtn');
    const enhanceBatchBtn = document.getElementById('enhanceBatchBtn');

    function getSearchConfig() {
        return {
            numQueries: parseInt(document.getElementById('numQueries').value) || 10,
            resultsPerQuery: parseInt(document.getElementById('resultsPerQuery').value) || 1,
            abstractsOnly: document.getElementById('abstractsOnly').checked
        };
    }

    claimForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const claimText = document.getElementById('claimText').value;
        const password = requirePassword ? document.getElementById('claimPassword').value : '';
        const config = getSearchConfig();
        
        try {
            const response = await fetch('/api/v1/claims', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: claimText,
                    password: password,
                    searchConfig: config
                })
            });
            const data = await response.json();
            if (data.claim_id) {
                window.location.href = `/progress?claim_id=${data.claim_id}`;
            } else {
                console.error('Claim ID is undefined:', data);
                alert('There was an error processing your claim. Please try again.');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('There was an error processing your claim. Please try again.');
        }
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

    enhanceClaimBtn.addEventListener('click', function() {
        const claimText = document.getElementById('claimText').value;
        if (!claimText) {
            alert('Please enter a claim to enhance.');
            return;
        }
        
        const password = document.getElementById('claimPassword') ? document.getElementById('claimPassword').value : null;
        enhanceClaim(claimText, password);
    });

    enhanceBatchBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('claimFile');
        const passwordInput = document.getElementById('batchPassword');
        
        if (!fileInput.files.length) {
            alert('Please select a file first');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        if (passwordInput) {
            formData.append('password', passwordInput.value);
        }
        
        try {
            const response = await fetch('/api/v1/enhance-batch', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                window.location.href = `/enhance-progress?batch_id=${data.batch_id}`;
            } else {
                alert(data.error || 'Error enhancing claims');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error enhancing claims');
        }
    });

    function submitFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        if (requirePassword) {
            const password = document.getElementById('batchPassword') ? document.getElementById('batchPassword').value : null;
            if (password) {
                formData.append('password', password);
            }
        }

        const config = getSearchConfig();
        formData.append('numQueries', config.numQueries);
        formData.append('resultsPerQuery', config.resultsPerQuery);
        formData.append('abstractsOnly', config.abstractsOnly);

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

    function enhanceClaim(claimText, password) {
        const enhanceBtn = document.getElementById('enhanceClaimBtn');
        const spinner = document.getElementById('enhanceSpinner');
        const claimInput = document.getElementById('claimText');
        
        // Show loading state
        enhanceBtn.classList.add('loading');
        spinner.style.display = 'block';
        claimInput.classList.add('loading');

        const body = { text: claimText };
        if (requirePassword && password) {
            body.password = password;
        }

        fetch('/api/v1/enhance-claim', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(body),
        })
        .then(response => response.json())
        .then(data => {
            if (data.suggested) {
                document.getElementById('claimText').value = data.suggested;
            } else {
                alert('Error enhancing claim: ' + (data.error || 'Unknown error'));
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('There was an error enhancing your claim. Please try again.');
        })
        .finally(() => {
            // Hide loading state
            enhanceBtn.classList.remove('loading');
            spinner.style.display = 'none';
            claimInput.classList.remove('loading');
        });
    }

    function enhanceBatch(file, password) {
        const formData = new FormData();
        formData.append('file', file);
        if (requirePassword && password) {
            formData.append('password', password);
        }

        fetch('/api/v1/enhance-batch', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => {
            if (data.batch_id) {
                window.location.href = `/enhance-results?batch_id=${data.batch_id}`;
            } else {
                alert('Error enhancing claims: ' + (data.error || 'Unknown error'));
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('There was an error enhancing your claims. Please try again.');
        });
    }
});
