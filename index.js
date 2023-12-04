// Async function to initialize the model and run example
async function initializeAndRunExample() {
    // ONNX model file name
    const onnxModelFileName = 'xgboost_AirQuality_ort.onnx';

    try {
        // Load the ONNX model using a URL with the ONNX Runtime Web Bundle
        const onnxModel = await ort.InferenceSession.create();
        await onnxModel.loadModel(`./${onnxModelFileName}`);

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
       
