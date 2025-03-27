document.addEventListener('DOMContentLoaded', function() {
    // State management for staged claims
    let stagedClaims = [];
    
    // DOM Elements
    const stagedClaimsContainer = document.getElementById('stagedClaims');
    const newClaimText = document.getElementById('newClaimText');
    const addClaimBtn = document.getElementById('addClaimBtn');
    const processAllBtn = document.getElementById('processAllBtn');
    const fileInput = document.getElementById('claimFile');
    const fileName = document.getElementById('fileName');
    const configToggle = document.getElementById('configToggle');
    const configPanel = document.getElementById('configPanel');

    // Show or hide configuration settings based on review type selection
    const reviewTypeRadios = document.querySelectorAll('input[name="reviewType"]');

    // Helper function to get search configuration
    function getSearchConfig() {
        return {
            numQueries: parseInt(document.getElementById('numQueries').value) || 5,
            resultsPerQuery: parseInt(document.getElementById('resultsPerQuery').value) || 5
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

            // Add bibliometric configuration
            const useBibliometrics = document.getElementById('useBibliometrics');
            formData.append('useBibliometrics', useBibliometrics ? useBibliometrics.checked : true);
            
            if (useBibliometrics && useBibliometrics.checked) {
                const authorImpactWeight = document.getElementById('authorImpactWeight');
                const citationImpactWeight = document.getElementById('citationImpactWeight');
                const venueImpactWeight = document.getElementById('venueImpactWeight');
                
                formData.append('authorImpactWeight', authorImpactWeight ? authorImpactWeight.value : 0.4);
                formData.append('citationImpactWeight', citationImpactWeight ? citationImpactWeight.value : 0.4);
                formData.append('venueImpactWeight', venueImpactWeight ? venueImpactWeight.value : 0.2);
            }

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
                // Store batch_id in localStorage for progress tracking
                localStorage.setItem('currentBatchId', data.batch_id);
                // Redirect to progress page with batch_id
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

    // Add this function to check claim status with batch_id
    async function checkClaimStatus(claimId, batchId) {
        try {
            const response = await fetch(`/api/v1/claims/${batchId}/${claimId}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error checking claim status:', error);
            return null;
        }
    }

    // Initialize bibliometric configuration toggle
    const toggleBibliometrics = document.getElementById('toggleBibliometrics');
    const bibliometricOptions = document.getElementById('bibliometricOptions');
    const toggleIcon = toggleBibliometrics.querySelector('.toggle-icon');
    
    if (toggleBibliometrics) {
        toggleBibliometrics.addEventListener('click', function() {
            bibliometricOptions.classList.toggle('hidden');
            toggleIcon.classList.toggle('rotated');
        });
    }
    
    // Initialize slider value displays
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        // Get the base name without "Weight" to match the ID in the HTML
        const baseId = slider.id.replace('Weight', '');
        const valueDisplay = document.getElementById(baseId + 'Value');
        if (valueDisplay) {
            // Initialize with current value
            valueDisplay.textContent = slider.value;
            
            // Update on input change
            slider.addEventListener('input', function() {
                valueDisplay.textContent = this.value;
            });
        }
    });
    
    // Toggle bibliometric weights visibility based on checkbox
    const useBibliometrics = document.getElementById('useBibliometrics');
    const bibliometricWeights = document.getElementById('bibliometricWeights');
    
    if (useBibliometrics && bibliometricWeights) {
        // Set initial state
        bibliometricWeights.style.display = useBibliometrics.checked ? 'block' : 'none';
        
        // Update on change
        useBibliometrics.addEventListener('change', function() {
            bibliometricWeights.style.display = this.checked ? 'block' : 'none';
        });
    }
    
    // Ensure bibliometric config is included in form submission
    if (processAllBtn) {
        const originalClick = processAllBtn.onclick;
        
        processAllBtn.onclick = function(event) {
            if (originalClick) {
                // Call the original handler first
                originalClick.call(this, event);
            }
            
            // Add bibliometric configuration to form data
            const formData = new FormData(document.querySelector('form'));
            
            // Add bibliometric configuration
            if (useBibliometrics) {
                formData.append('useBibliometrics', useBibliometrics.checked);
                
                if (useBibliometrics.checked) {
                    const authorImpactWeight = document.getElementById('authorImpactWeight');
                    const citationImpactWeight = document.getElementById('citationImpactWeight');
                    const venueImpactWeight = document.getElementById('venueImpactWeight');
                    
                    if (authorImpactWeight) formData.append('authorImpactWeight', authorImpactWeight.value);
                    if (citationImpactWeight) formData.append('citationImpactWeight', citationImpactWeight.value);
                    if (venueImpactWeight) formData.append('venueImpactWeight', venueImpactWeight.value);
                }
            }
        };
    }
});
