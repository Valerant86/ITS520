// Load ONNX runtime module
const ort = require('onnxruntime-web');

// Function to create input boxes for each header
function createInputBoxes() {
    const inputBoxesDiv = document.getElementById('inputBoxes');

    headersList.forEach(header => {
        const label = document.createElement('label');
        label.textContent = `${header}:`;

        const input = document.createElement('input');
        input.type = 'number';
        input.id = header;

        inputBoxesDiv.appendChild(label);
        inputBoxesDiv.appendChild(input);
    });
}

// Function to update predictions when the button is clicked
async function updatePrediction() {
    const inputValues = {};

    // Collect values from input boxes
    headersList.forEach(header => {
        const inputValue = document.getElementById(header).value;
        inputValues[header] = inputValue ? parseFloat(inputValue) : 0;
    });

    // Prepare the input data for ONNX model
    const inputData = new Float32Array(headersList.map(header => inputValues[header]));

    // Create an ONNX Tensor from the input data
    const inputTensor = new ort.Tensor(ort.WebGLFloat32, new Float32Array(inputData), [1, headersList.length]);

    // Run the ONNX model to get predictions
    const outputTensor = await onnxModel.run([labelName], { [inputName]: inputTensor });

    // Get the prediction result
    const predictionResult = outputTensor.getValues();

    // Display the prediction result
    document.getElementById('predictionResult').textContent = predictionResult[0].toFixed(2);
}

// Replace with your actual headers
const headersList = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'];

// Populate the header input boxes
createInputBoxes();

// ONNX model file name
const onnxModelFileName = 'xgboost_AirQuality_ort.onnx';

// Load the ONNX model
const onnxModel = await ort.InferenceSession.create({ backendHint: 'webgl' });
await onnxModel.loadModel(`./${onnxModelFileName}`);

// Extract input and output names
const inputName = onnxModel.inputNames[0];
const labelName = onnxModel.outputNames[0];

// Update the model status on the HTML page
document.getElementById('modelLoaded').textContent = 'Yes';
