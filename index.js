
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
      let session = await ort.InferenceSession.create('xgboost_AirQuality_ort');

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
        return "maroon"; // Hazardous
      }
    }
  </script>
</body>
</html>
