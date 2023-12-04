async function predict() {
    // ... (existing code)

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
