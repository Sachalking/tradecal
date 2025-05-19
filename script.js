document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM fully loaded");

    const tradeForm = document.getElementById('tradeForm');
    const entryPriceEl = document.getElementById('entryPrice');
    const stopLossPriceEl = document.getElementById('stopLossPrice');
    const takeProfitPriceEl = document.getElementById('takeProfitPrice');
    const positionSizeEl = document.getElementById('positionSize');
    const accountSizeEl = document.getElementById('accountSize');
    const riskToleranceEl = document.getElementById('riskTolerance');
    // const assetTypeEl = document.getElementById('assetType'); // Optional, not used in core calculations yet
    const tradeTypeEl = document.getElementById('tradeType');
    const calculateBtn = document.getElementById('calculateBtn');

    // Results Elements
    const resultsCards = document.querySelectorAll('.card');
    const riskPerUnitResultEl = document.getElementById('riskPerUnit');
    const totalRiskResultEl = document.getElementById('totalRisk');
    const profitPerUnitResultEl = document.getElementById('profitPerUnit');
    const totalProfitResultEl = document.getElementById('totalProfit');
    const rrrResultEl = document.getElementById('rrr');
    const percentAccountRiskedResultEl = document.getElementById('percentAccountRisked');

    // ML Results Elements
    const mlPredictedOutcomeEl = document.getElementById('mlPredictedOutcome');
    const mlConfidenceScoreEl = document.getElementById('mlConfidenceScore');
    const mlNotesEl = document.getElementById('mlNotes');

    console.log("Form elements initialized");

    // Prevent form from submitting normally
    tradeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        console.log("Form submission prevented");
        return false;
    });

    calculateBtn.addEventListener('click', async (e) => { // Make the function async
        e.preventDefault(); // Prevent default button behavior
        console.log("Calculate button clicked");
        const entryPrice = parseFloat(entryPriceEl.value);
        const stopLossPrice = parseFloat(stopLossPriceEl.value);
        const takeProfitPrice = parseFloat(takeProfitPriceEl.value);
        const positionSize = parseFloat(positionSizeEl.value);
        const accountSize = parseFloat(accountSizeEl.value);
        const riskTolerance = parseFloat(riskToleranceEl.value); // Not directly used in these calculations but good to have
        const tradeType = tradeTypeEl.value;

        if (isNaN(entryPrice) || isNaN(stopLossPrice) || isNaN(takeProfitPrice) || isNaN(positionSize) || isNaN(accountSize)) {
            alert('Please fill in all required numerical fields.');
            return;
        }

        let riskPerUnit;
        let profitPerUnit;

        if (tradeType === 'long') {
            if (entryPrice <= stopLossPrice) {
                alert('For a long trade, entry price must be greater than stop-loss price.');
                return;
            }
            if (takeProfitPrice <= entryPrice) {
                alert('For a long trade, take-profit price must be greater than entry price.');
                return;
            }
            riskPerUnit = entryPrice - stopLossPrice;
            profitPerUnit = takeProfitPrice - entryPrice;
        } else { // Short trade
            if (entryPrice >= stopLossPrice) {
                alert('For a short trade, entry price must be less than stop-loss price.');
                return;
            }
             if (takeProfitPrice >= entryPrice) {
                alert('For a short trade, take-profit price must be less than entry price.');
                return;
            }
            riskPerUnit = stopLossPrice - entryPrice;
            profitPerUnit = entryPrice - takeProfitPrice;
        }

        const totalRisk = riskPerUnit * positionSize;
        const totalProfit = profitPerUnit * positionSize;
        const rrr = totalRisk > 0 ? totalProfit / totalRisk : 0;
        const percentAccountRisked = accountSize > 0 ? (totalRisk / accountSize) * 100 : 0;

        // Function to update result with subtle animation
        const updateResult = (element, value) => {
            element.classList.add('animate__animated', 'animate__fadeIn', 'animate__faster');
            element.textContent = value;
            setTimeout(() => {
                element.classList.remove('animate__animated', 'animate__fadeIn', 'animate__faster');
            }, 500);
        };

        // Update results with subtle animations
        updateResult(riskPerUnitResultEl, `$${riskPerUnit.toFixed(2)}`);
        updateResult(totalRiskResultEl, `$${totalRisk.toFixed(2)}`);
        updateResult(profitPerUnitResultEl, `$${profitPerUnit.toFixed(2)}`);
        updateResult(totalProfitResultEl, `$${totalProfit.toFixed(2)}`);
        updateResult(rrrResultEl, rrr.toFixed(2));
        updateResult(percentAccountRiskedResultEl, `${percentAccountRisked.toFixed(2)}%`);

        // Add subtle animation to results cards
        resultsCards.forEach(card => {
            card.classList.add('animate__animated', 'animate__fadeIn', 'animate__faster');
            setTimeout(() => {
                card.classList.remove('animate__animated', 'animate__fadeIn', 'animate__faster');
            }, 500);
        });

        // Call ML Backend
        try {
            console.log("Preparing to call ML backend...");

            // Prepare request data
            const requestData = {
                entry_price: entryPrice,
                stop_loss: stopLossPrice,
                take_profit: takeProfitPrice,
                position_size: positionSize,
                account_size: accountSize,
                trade_type: tradeType
            };

            console.log("Request data:", requestData);

            // Show loading state
            updateResult(mlPredictedOutcomeEl, 'Loading...');
            updateResult(mlConfidenceScoreEl, '...');
            updateResult(mlNotesEl, 'Fetching prediction from ML model...');

            // Make API request - try different URLs to handle various hosting scenarios
            const apiUrls = [
                'https://tradecal.onrender.com/predict',  // Production URL
                'http://127.0.0.1:5001/predict',          // Local development
                'http://localhost:5001/predict',          // Alternative local
                '/predict'                                // Same-origin (when served by Flask)
            ];

            let response = null;
            let lastError = null;

            // Try each URL until one works
            for (const url of apiUrls) {
                try {
                    console.log(`Trying API endpoint: ${url}`);
                    response = await fetch(url, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        },
                        body: JSON.stringify(requestData),
                    });

                    console.log(`Response status from ${url}:`, response.status);

                    if (response.ok) {
                        // If successful, break out of the loop
                        break;
                    } else {
                        let errorMessage = `HTTP error! status: ${response.status}`;
                        try {
                            const errorData = await response.json();
                            errorMessage = errorData.error || errorMessage;
                        } catch (e) {
                            console.error("Failed to parse error response:", e);
                        }
                        lastError = new Error(errorMessage);
                        // Continue to try the next URL
                    }
                } catch (e) {
                    console.error(`Error with endpoint ${url}:`, e);
                    lastError = e;
                    // Continue to try the next URL
                }
            }

            // If we've tried all URLs and none worked
            if (!response || !response.ok) {
                throw lastError || new Error("Failed to connect to any API endpoint");
            }

            // Parse response data
            const responseText = await response.text();
            console.log("Raw response:", responseText);

            let mlData;
            try {
                mlData = JSON.parse(responseText);
                console.log("Parsed ML data:", mlData);

                // Log specific fields for debugging
                console.log("Prediction outcome:", mlData.predicted_outcome);
                console.log("Confidence score:", mlData.confidence_score);
                console.log("Notes:", mlData.notes);
                console.log("Model features:", mlData.model_features_used);

                // Check for any unexpected data structure
                if (!mlData.predicted_outcome) {
                    console.warn("Warning: Missing predicted_outcome in response");
                }
                if (mlData.confidence_score === undefined) {
                    console.warn("Warning: Missing confidence_score in response");
                }
                if (!mlData.model_features_used) {
                    console.warn("Warning: Missing model_features_used in response");
                }
            } catch (e) {
                console.error("Failed to parse JSON response:", e);
                throw new Error("Invalid response from server: " + responseText.substring(0, 100) + "...");
            }

            // Update UI with prediction results immediately
            console.log("Updating UI with prediction results");

            // Function to update ML prediction display in an analytical way
            const updateMLPrediction = () => {
                // Update prediction outcome
                if (mlPredictedOutcomeEl) {
                    console.log(`Updating prediction outcome: ${mlData.predicted_outcome}`);

                    // Format the prediction text to be more readable
                    let predictionText = mlData.predicted_outcome || '-';
                    predictionText = predictionText.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    mlPredictedOutcomeEl.textContent = predictionText;

                    // Set color based on prediction
                    if (predictionText.includes('Increase')) {
                        mlPredictedOutcomeEl.style.color = '#4caf50'; // Green for positive
                    } else if (predictionText.includes('Decrease')) {
                        mlPredictedOutcomeEl.style.color = '#f44336'; // Red for negative
                    } else {
                        mlPredictedOutcomeEl.style.color = '#ff9800'; // Orange for neutral
                    }
                } else {
                    console.error('Prediction outcome element not found');
                }

                // Update confidence score and bar
                if (mlConfidenceScoreEl) {
                    const confidenceValue = mlData.confidence_score !== undefined ? mlData.confidence_score : 0;
                    const confidencePercent = (confidenceValue * 100).toFixed(0);
                    console.log(`Updating confidence: ${confidencePercent}%`);

                    mlConfidenceScoreEl.textContent = `${confidencePercent}%`;

                    // Update confidence bar
                    const confidenceBar = document.getElementById('confidenceBar');
                    if (confidenceBar) {
                        confidenceBar.style.width = `${confidencePercent}%`;

                        // Change color based on confidence level
                        if (confidenceValue >= 0.7) {
                            confidenceBar.style.background = 'linear-gradient(90deg, #4caf50, #8bc34a)'; // Green
                        } else if (confidenceValue >= 0.4) {
                            confidenceBar.style.background = 'linear-gradient(90deg, #ff9800, #ffc107)'; // Orange
                        } else {
                            confidenceBar.style.background = 'linear-gradient(90deg, #f44336, #ff5722)'; // Red
                        }
                    }
                } else {
                    console.error('Confidence score element not found');
                }

                // Update features grid
                const featuresGrid = document.getElementById('mlFeatures');
                if (featuresGrid && mlData.model_features_used) {
                    // Clear previous features
                    featuresGrid.innerHTML = '';

                    // Add each feature as a grid item
                    Object.entries(mlData.model_features_used).forEach(([key, value]) => {
                        const featureItem = document.createElement('div');
                        featureItem.className = 'feature-item';

                        const featureName = document.createElement('div');
                        featureName.className = 'feature-name';
                        featureName.textContent = key.replace(/_/g, ' ').toUpperCase();

                        const featureValue = document.createElement('div');
                        featureValue.className = 'feature-value';
                        featureValue.textContent = typeof value === 'number' ? value.toFixed(2) : value;

                        featureItem.appendChild(featureName);
                        featureItem.appendChild(featureValue);
                        featuresGrid.appendChild(featureItem);
                    });
                }

                // Update notes
                if (mlNotesEl) {
                    // Extract just the notes part without the features
                    const notesText = mlData.notes || '-';
                    console.log(`Updating notes: ${notesText}`);
                    mlNotesEl.textContent = notesText;
                } else {
                    console.error('Notes element not found');
                }
            };

            // Update the ML prediction display
            updateMLPrediction();

            // Force a DOM update
            document.body.offsetHeight;

            // Add a subtle highlight effect to the ML card
            const mlCard = document.querySelector('.results-ml');
            mlCard.style.backgroundColor = 'rgba(58, 123, 213, 0.1)';
            setTimeout(() => {
                mlCard.style.backgroundColor = '';
            }, 1000);

            console.log("ML prediction displayed successfully");

        } catch (error) {
            console.error('Error fetching ML prediction:', error);

            // Update error messages in analytical format
            if (mlPredictedOutcomeEl) {
                mlPredictedOutcomeEl.textContent = 'Connection Error';
                mlPredictedOutcomeEl.style.color = '#f44336'; // Red for error
            }

            if (mlConfidenceScoreEl) {
                mlConfidenceScoreEl.textContent = '-';
            }

            // Update confidence bar to show error
            const confidenceBar = document.getElementById('confidenceBar');
            if (confidenceBar) {
                confidenceBar.style.width = '100%';
                confidenceBar.style.background = 'linear-gradient(90deg, #f44336, #ff5722)'; // Red for error
            }

            // Clear features grid and show error
            const featuresGrid = document.getElementById('mlFeatures');
            if (featuresGrid) {
                featuresGrid.innerHTML = '<div class="feature-item" style="grid-column: span 2;"><div class="feature-name">ERROR</div><div class="feature-value" style="color: #f44336;">API Connection Failed</div></div>';
            }

            // Show error message in notes
            if (mlNotesEl) {
                mlNotesEl.textContent = `Error: ${error.message}. Please try again or check server connection.`;
            }

            // Force a DOM update
            document.body.offsetHeight;

            // Add subtle error highlight to ML card
            const mlCard = document.querySelector('.results-ml');
            mlCard.style.backgroundColor = 'rgba(231, 76, 60, 0.1)';
            setTimeout(() => {
                mlCard.style.backgroundColor = '';
            }, 1000);
        }
    });
});