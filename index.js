async function runExample() {
  // Fetch values from input boxes
  var x = new Float32Array(12);
  for (let i = 1; i <= 12; i++) {
    x[i - 1] = parseFloat(document.getElementById(`box${i}`).value);
  }

  // Create the input tensor
  let tensorX = new ort.Tensor('float32', x, [1, 12]);
  let feeds = { float_input: tensorX };

  // Load the ONNX model
  let session = await ort.InferenceSession.create('xgboost_AirQuality_ort.onnx');

  // Run the model and get the result
  let result = await session.run(feeds);
  let outputData = result.values().next().value.data; // Update this line to get the correct output

  // Format the output value
  outputData = parseFloat(outputData[0]).toFixed(2);

  // Display the output value in the predictions box
  let predictions = document.getElementById('predictions');
  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
    <table>
      <tr>
        <td>  Rating of Air Quality  </td>
        <td id="td0">  ${outputData}  </td>
      </tr>
    </table>`;

  // Change the background color based on the output value for predictions box
  var boxColor = getBoxColor(outputData);
  predictions.style.backgroundColor = boxColor;

  // Display the predicted AQI in a separate box with background color
  let predictedAQI = document.getElementById('predictedAQI');
  predictedAQI.innerHTML = ` <hr> Predicted AQI: ${outputData} </hr>`;
  predictedAQI.style.backgroundColor = boxColor;
}
