document.addEventListener('DOMContentLoaded', function() {
    // State management for staged claims
    let stagedClaims = [];
    
    // DOM Elements
    const stagedClaimsContainer = document.getElementById('stagedClaims');
    const newClaimText = document.getElementById('newClaimText');
    const addClaimBtn = document.getElementById('addClaimBtn');
    const enhanceClaimBtn = document.getElementById('enhanceClaimBtn');
    const enhanceAllBtn = document.getElementById('enhanceAllBtn');
    const processAllBtn = document.getElementById('processAllBtn');
    const fileInput = document.getElementById('claimFile');
    const fileName = document.getElementById('fileName');
    const configToggle = document.getElementById('configToggle');
    const configPanel = document.getElementById('configPanel');

    // Show or hide configuration settings based on review type selection
    const reviewTypeRadios = document.querySelectorAll('input[name="reviewType"]');

    function updateConfigPanelVisibility() {
        const selectedReviewType = document.querySelector('input[name="reviewType"]:checked').value;
        if (selectedReviewType === 'full' || selectedReviewType === 'abstracts') {
            configPanel.style.display = 'block';
        } else {
            configPanel.style.display = 'none';
        }
    }

    // Initialize config panel visibility
    updateConfigPanelVisibility();

    // Add event listeners to review type radio buttons
    reviewTypeRadios.forEach(radio => {
        radio.addEventListener('change', updateConfigPanelVisibility);
    });

    // Helper function to get search configuration
    function getSearchConfig() {
        return {
            numQueries: parseInt(document.getElementById('numQueries').value) || 5,
            resultsPerQuery: parseInt(document.getElementById('resultsPerQuery').value) || 5,
            reviewType: document.querySelector('input[name="reviewType"]:checked').value
        };
    }

    // Helper function to create a claim element
    function createClaimElement(claim, index) {
        const claimDiv = document.createElement('div');
        claimDiv.className = 'staged-claim';
        claimDiv.innerHTML = `
            <div class="claim-text">${claim}</div>
            <div class="claim-actions">
                <button class="action-button edit-button" data-index="${index}">Edit</button>
                <button class="action-button enhance-button" data-index="${index}">
                    <span>Enhance</span>
                    <div class="spinner enhance-spinner" style="display: none;"></div>
                </button>
                <button class="action-button delete-button" data-index="${index}">Delete</button>
            </div>
        `;
        return claimDiv;
    }

    // Update the staging area display
    function updateStagingArea() {
        stagedClaimsContainer.innerHTML = '';
        stagedClaims.forEach((claim, index) => {
            stagedClaimsContainer.appendChild(createClaimElement(claim, index));
        });
        
        // Update button states
        enhanceAllBtn.disabled = stagedClaims.length === 0;
        processAllBtn.disabled = stagedClaims.length === 0;
    }

    // Add claim to staging area
    function addClaim(claim) {
        if (claim.trim()) {
            stagedClaims.push(claim.trim());
            updateStagingArea();
            newClaimText.value = '';
        }
    }

    // Event listener for adding a claim
    addClaimBtn.addEventListener('click', () => {
        addClaim(newClaimText.value);
    });

    // Event listener for enhance and add
    enhanceClaimBtn.addEventListener('click', async () => {
        const claim = newClaimText.value.trim();
        if (!claim) return;

        enhanceClaimBtn.classList.add('loading');
        const spinner = document.getElementById('enhanceSpinner');
        spinner.style.display = 'block';

        try {
            const response = await fetch('/api/v1/enhance-claim', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: claim }),
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.suggested) {
                // Add both original and enhanced claims to staging area
                addClaim(data.suggested);
                
                // Show enhancement feedback
                const feedback = document.createElement('div');
                feedback.className = 'enhancement-feedback';
                feedback.innerHTML = `
                    <div class="feedback-content">
                        <p><strong>Enhanced claim:</strong> ${data.suggested}</p>
                        ${data.explanation ? `<p><small>${data.explanation}</small></p>` : ''}
                    </div>
                `;
                
                // Insert feedback after the claim input
                const claimInputContainer = document.querySelector('.claim-input-container');
                claimInputContainer.appendChild(feedback);
                
                // Remove feedback after 5 seconds
                setTimeout(() => {
                    feedback.remove();
                }, 5000);
                
                // Clear the input
                newClaimText.value = '';
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error enhancing claim: ' + error.message);
        } finally {
            enhanceClaimBtn.classList.remove('loading');
            spinner.style.display = 'none';
        }
    });

    // Event listener for file upload
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            fileName.textContent = file.name;
            try {
                const text = await file.text();
                // Split by newlines and filter out empty lines
                const claims = text.split('\n')
                    .map(claim => claim.trim())
                    .filter(claim => claim);
                stagedClaims.push(...claims);
                updateStagingArea();
            } catch (error) {
                console.error('Error reading file:', error);
                alert('Error reading file');
            }
        }
    });

    // Event delegation for claim actions
    stagedClaimsContainer.addEventListener('click', async (e) => {
        const button = e.target.closest('button');
        if (!button) return;

        const index = parseInt(button.dataset.index);
        
        if (button.classList.contains('delete-button')) {
            stagedClaims.splice(index, 1);
            updateStagingArea();
        } else if (button.classList.contains('edit-button')) {
            const newText = prompt('Edit claim:', stagedClaims[index]);
            if (newText) {
                stagedClaims[index] = newText.trim();
                updateStagingArea();
            }
        } else if (button.classList.contains('enhance-button')) {
            const spinner = button.querySelector('.enhance-spinner');
            const buttonText = button.querySelector('span');
            
            button.disabled = true;
            spinner.style.display = 'inline-block';
            buttonText.style.opacity = '0.5';
            
            try {
                const response = await fetch('/api/v1/enhance-claim', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: stagedClaims[index] }),
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                if (data.suggested) {
                    stagedClaims[index] = data.suggested;
                    updateStagingArea();
                    
                    // Show enhancement feedback
                    const claimElement = button.closest('.staged-claim');
                    const feedback = document.createElement('div');
                    feedback.className = 'enhancement-feedback';
                    feedback.innerHTML = `
                        <div class="feedback-content">
                            <p><small>${data.explanation || 'Claim enhanced successfully'}</small></p>
                        </div>
                    `;
                    claimElement.appendChild(feedback);
                    
                    // Remove feedback after 5 seconds
                    setTimeout(() => {
                        feedback.remove();
                    }, 5000);
                } else {
                    throw new Error(data.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error enhancing claim: ' + error.message);
            } finally {
                button.disabled = false;
                spinner.style.display = 'none';
                buttonText.style.opacity = '1';
            }
        }
    });

    // Process all claims
    processAllBtn.addEventListener('click', async () => {
        const emailInput = document.getElementById('notificationEmail');
        const emailCheckbox = document.getElementById('emailNotification');
        const passwordInput = document.getElementById('accessPassword');
        
        // Only get email details if the notification section exists
        const email = emailInput ? emailInput.value : '';
        const notify = emailCheckbox ? emailCheckbox.checked : false;
        
        // Validate email if notifications are enabled
        if (notify) {
            if (!email) {
                alert('Please enter an email address for notifications');
                return;
            }
            
            // Basic email validation
            const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
            if (!emailRegex.test(email)) {
                alert('Please enter a valid email address');
                return;
            }
        }
        
        // Get password if required
        const password = requirePassword ? passwordInput.value : '';
        
        // Validate password if required
        if (requirePassword && !password) {
            alert('Please enter the access password');
            return;
        }
        
        const config = getSearchConfig();
        
        processAllBtn.disabled = true;  // Disable button while processing
        
        try {
            const formData = new FormData();
            
            // Create a text file with one claim per line instead of JSON stringifying the array
            const claimsText = stagedClaims.join('\n');
            const claimsBlob = new Blob([claimsText], { type: 'text/plain' });
            formData.append('file', claimsBlob, 'claims.txt');
            
            // Only append email if notification is checked
            if (notify && email) {
                formData.append('email', email);
            }

            // Add password if required
            if (requirePassword) {
                formData.append('password', password);
            }

            // Add configuration to formData
            formData.append('numQueries', config.numQueries);
            formData.append('resultsPerQuery', config.resultsPerQuery);
            formData.append('reviewType', config.reviewType);

            const response = await fetch('/api/v1/batch', {
                method: 'POST',
                body: formData,
            });
            
            const data = await response.json();
            if (data.error === "Invalid password") {
                alert('Invalid password. Please try again.');
                return;
            }
            
            if (data.batch_id) {
                // Redirect to progress page
                window.location.href = `/progress?batch_id=${data.batch_id}`;
            } else {
                throw new Error('No batch_id received');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error processing claims');
        } finally {
            processAllBtn.disabled = false;  // Re-enable button
        }
    });

    // Enhance all claims
    enhanceAllBtn.addEventListener('click', async () => {
        enhanceAllBtn.disabled = true;
        
        // Add spinner to button
        const originalButtonText = enhanceAllBtn.innerHTML;
        enhanceAllBtn.innerHTML = `
            <span>Enhancing Claims...</span>
            <div class="spinner enhance-spinner" style="display: inline-block;"></div>
        `;
        
        try {
            const enhancedClaims = [];
            for (let i = 0; i < stagedClaims.length; i++) {
                // Update button text to show progress
                const progressSpan = enhanceAllBtn.querySelector('span');
                progressSpan.textContent = `Enhancing Claim ${i + 1}/${stagedClaims.length}...`;
                
                const response = await fetch('/api/v1/enhance-claim', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: stagedClaims[i] }),
                });
                
                const data = await response.json();
                enhancedClaims.push(data.suggested || stagedClaims[i]);
            }
            
            stagedClaims = enhancedClaims;
            updateStagingArea();
            
            // Show success feedback
            const feedback = document.createElement('div');
            feedback.className = 'enhancement-feedback';
            feedback.innerHTML = `
                <div class="feedback-content">
                    <p><strong>All claims enhanced successfully!</strong></p>
                </div>
            `;
            
            // Insert feedback after the enhance all button
            enhanceAllBtn.parentElement.appendChild(feedback);
            
            // Remove feedback after 5 seconds
            setTimeout(() => {
                feedback.remove();
            }, 5000);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error enhancing claims: ' + error.message);
        } finally {
            // Restore button to original state
            enhanceAllBtn.innerHTML = originalButtonText;
            enhanceAllBtn.disabled = false;
        }
    });

    // Add email validation feedback
    const emailInput = document.getElementById('notificationEmail');
    const emailCheckbox = document.getElementById('emailNotification');
    
    if (emailInput && emailCheckbox) {
        emailCheckbox.addEventListener('change', function() {
            if (this.checked) {
                emailInput.required = true;
            } else {
                emailInput.required = false;
                emailInput.classList.remove('invalid');
            }
        });
        
        emailInput.addEventListener('input', function() {
            if (emailCheckbox.checked) {
                const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
                if (!emailRegex.test(this.value)) {
                    this.classList.add('invalid');
                } else {
                    this.classList.remove('invalid');
                }
            }
        });
    }

    // Add this at the beginning of your DOMContentLoaded event listener
    const toggleInstructions = document.getElementById('toggleInstructions');
    const instructions = document.querySelector('.instructions');

    toggleInstructions.addEventListener('click', () => {
        const isHidden = instructions.style.display === 'none';
        instructions.style.display = isHidden ? 'block' : 'none';
        toggleInstructions.classList.toggle('active');
        toggleInstructions.querySelector('.toggle-text').textContent = 
            isHidden ? 'Hide Instructions' : 'Show Instructions';
    });
});
