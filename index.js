// index.js

async function runExample() {
  // Use a Float32Array for the input tensor
  var x = new Float32Array(12);

  // Populate the input values
  x[0] = parseFloat(document.getElementById('box1').value);
  x[1] = parseFloat(document.getElementById('box2').value);
  x[2] = parseFloat(document.getElementById('box3').value);
  x[3] = parseFloat(document.getElementById('box4').value);
  x[4] = parseFloat(document.getElementById('box5').value);
  x[5] = parseFloat(document.getElementById('box6').value);
  x[6] = parseFloat(document.getElementById('box7').value);
  x[7] = parseFloat(document.getElementById('box8').value);
  x[8] = parseFloat(document.getElementById('box9').value);
  x[9] = parseFloat(document.getElementById('box10').value);
  x[10] = parseFloat(document.getElementById('box11').value);
  x[11] = parseFloat(document.getElementById('box12').value);

  // Create the input tensor
  let tensorX = new ort.Tensor('float32', x, [1, 12]);
  let feeds = { float_input: tensorX };

  // Load the ONNX model
  let session = await ort.InferenceSession.create('xgboost_AirQuality_ort.onnx');

  // Run the model and get the result
  let result = await session.run(feeds);
  let outputData = result.variable.data;

  // Format the output value
  outputData = parseFloat(outputData).toFixed(2);

  // Display the output value
  let predictions = document.getElementById('predictions');
  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Rating of Air Quality  </td>
       <td id="td0">  ${outputData}  </td>
     </tr>
  </table>`;

  // Change the background color based on the output value
  var boxColor = getBoxColor(outputData);
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
    return "Brown"; // Hazardous
  }
}
