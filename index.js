// Async function to initialize the model and run example
async function initializeAndRunExample() {
    // ONNX model file name
    const onnxModelFileName = 'xgboost_AirQuality_ort.onnx';

    try {
        // Load the ONNX model using fetch and ArrayBuffer
        const response = await fetch(onnxModelFileName);
        const buffer = await response.arrayBuffer();
        const onnxModel = await ort.InferenceSession.create(buffer);

        // Extract input and output names
        const inputName = onnxModel.inputNames[0];
        const outputName = onnxModel.outputNames[0];

        // Function to collect input values from text boxes
        function collectInputValues() {
            const inputValues = {};

            const inputIds = [
                'box1', 'box2', 'box3', 'box4', 'box5', 'box6',
                'box7', 'box8', 'box9', 'box10', 'box11', 'box12'
            ];

            inputIds.forEach(id => {
                const inputValue = document.getElementById(id).value;
                inputValues[id] = inputValue ? parseFloat(inputValue) : 0;
            });

            return inputValues;
        }

        // Function to update predictions when the button is clicked
        window.runExample = async function () {
            // Collect input values from text boxes
            const inputValues = collectInputValues();

            // Prepare the input data for ONNX model
            const inputData = new Float32Array(Object.values(inputValues));

            // Create an ONNX Tensor from the input data
            const inputTensor = new ort.Tensor(ort.WebGLFloat32, new Float32Array(inputData), [1, Object.keys(inputValues).length]);

            // Run the ONNX model to get predictions
            const outputTensor = await onnxModel.run([outputName], { [inputName]: inputTensor });

            // Get the prediction result
            const predictionResult = outputTensor.getValues();

            // Display the prediction result
            const predictionsDiv = document.getElementById('predictions');
            predictionsDiv.textContent = `Predicted AQI: ${predictionResult[0].toFixed(2)}`;
        };

        // Update the model status on the HTML page
        const modelLoadedDiv = document.getElementById('modelLoaded');
        modelLoadedDiv.textContent = 'Yes';
    } catch (error) {
        console.error('Error loading the ONNX model:', error);
        const modelLoadedDiv = document.getElementById('modelLoaded');
        modelLoadedDiv.textContent = 'No';
    }
}

// Call the initialization function
initializeAndRunExample();
