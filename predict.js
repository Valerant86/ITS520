// predict.js
let model, session;

async function loadModel() {
    model = await onnx.load('xgboost_Classification_AirQuality_ort.onnx');
    session = new onnx.InferenceSession({ backendHint: 'webgl' });
    await session.loadModel(model);
    console.log('Model loaded successfully');
}

async function predict() {
    const inputValue = parseFloat(document.getElementById("inputValue").value);

    if (!model || !session) {
        await loadModel();
    }

    // Prepare input tensor
    const tensorArray = new Float32Array([inputValue]);
    const tensor = new onnx.Tensor(tensorArray, 'float32', [1, 1]);

    // Run the model
    const outputMap = await session.run([tensor]);

    // Get the predicted result
    const prediction = outputMap.values().next().value.data[0];
    console.log('Prediction:', prediction);

    // Display the prediction
    const resultElement = document.getElementById("predictionResult");
    resultElement.textContent = `Prediction: ${prediction}`;

    // Change the background color based on the output value
    const predictions = document.getElementById("predictions");
    const boxColor = getBoxColor(prediction);
    predictions.style.backgroundColor = boxColor;
}

function getBoxColor(aqi) {
    // Customize this logic based on your desired color assignment
    if (aqi <= 50) {
        return "green"; // Good
    } else if (aqi <= 100) {
        return "yellow"; // Moderate
    } else if (aqi <= 150) {
        return "orange"; // Unhealthy for sensitive groups
    } else if (aqi <= 200) {
        return "red"; // Unhealthy
    } else if (aqi <= 300) {
        return "purple"; // Very Unhealthy
    } else {
        return "maroon"; // Hazardous
    }
}

// Load the model on page load
loadModel();
