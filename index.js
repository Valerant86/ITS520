// Async function to initialize the model and run example
async function initializeAndRunExample() {
    // ONNX model file name
    const onnxModelFileName = 'xgboost_AirQuality_ort.onnx';

    try {
        // Load the ONNX model using onnxjs
        const response = await fetch(`./${onnxModelFileName}`);
        const arrayBuffer = await response.arrayBuffer();
        const onnxModel = new onnx.ModelProto(arrayBuffer);

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
            const inputTensor = new onnx.Tensor(inputData, 'float32', [1, Object.keys(inputValues).leng
