// Load ONNX runtime module
const ort = require('onnxruntime-web');

// Function to update predictions when the header is changed
async function updatePrediction() {
    // Get the selected header value
    const selectedHeader = document.getElementById('headerSelect').value;

    // Make sure the selected header is valid
    if (!selectedHeader) {
        alert('Please select a valid header.');
        return;
    }

    // Get the index of the selected header
    const headerIndex = headersList.indexOf(selectedHeader);

    // Extract the corresponding column from the test data
    const testDataColumn = X_test_tr[:, headerIndex];

    // Prepare the input data for ONNX model
    const inputData = new Float32Array(testDataColumn.data);

    // Create an ONNX Tensor from the input data
    const inputTensor = new ort.Tensor(ort.WebGLFloat32, new Float32Array(inputData), [inputData.length]);

    // Run the ONNX model to get predictions
    const outputTensor = await session.run([labelName], { [inputName]: inputTensor });

    // Get the prediction result
    const predictionResult = outputTensor.getValues();

    // Display the prediction result
    document.getElementById('predictionResult').textContent = predictionResult[0].toFixed(2);
}

// Replace with your actual headers
const headersList = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI'];

// Populate the header dropdown options
const headerSelect = document.getElementById('headerSelect');
headersList.forEach(header => {
    const option = document.createElement('option');
    option.value = header;
    option.text = header;
    headerSelect.appendChild(option);
});

// Initial prediction on page load
updatePrediction();
