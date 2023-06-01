function submitForm() {
    return new Promise((resolve, reject) => {
      const jsonInput = document.getElementById("json-input");
      const file = jsonInput.files[0];
  
      const reader = new FileReader();
      reader.onload = function (event) {
        const jsonContents = event.target.result;
        try {
          const inputData = JSON.parse(jsonContents);
          console.log(inputData);
          resolve(inputData);
        } catch (error) {
          console.error('Error parsing JSON', error);
          throw error;
        }
      };
      reader.readAsText(file);
    });
  }

function sendSmokingData(inputData) {
    fetch('/predict-smoking', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(inputData)
    })
    .then(response => response.json())
    .then(data => {
        const prediction = data.prediction;
        document.getElementById("result").innerText = 'Predicted class: ' + prediction;
    })
    .catch(error => {
        console.error('Error', error);
        document.getElementById("result").innerText = 'Error: ' + error.message;
    });
}

function sendDrinkingData(inputData) {
  fetch('/predict-drinking', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(inputData)
  })
  .then(response => response.json())
  .then(data => {
      const prediction = data.prediction;
      document.getElementById("result").innerText = 'Predicted class: ' + prediction;
  })
  .catch(error => {
      console.error('Error', error);
      document.getElementById("result").innerText = 'Error: ' + error.message;
  });
}
