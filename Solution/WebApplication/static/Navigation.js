document.getElementById("left-arrow").addEventListener("click", function () {
    navigate("left");
});

document.getElementById("right-arrow").addEventListener("click", function () {
    navigate("right");
});

function navigate(direction) {
    const currentPage = window.location.pathname;

    if (direction === "left" && currentPage !== "/") {
        window.location.href = "/";
    } else if (direction === "right" && currentPage !== "/DrinkingPrediction") {
        window.location.href = "/DrinkingPrediction";
    }
}

var predictButton = document.getElementById("predict-button");
if(predictButton) {
  predictButton.addEventListener("click", function () {
    submitForm()
      .then(inputData => sendSmokingData(inputData))
      .catch(error => console.error('Error', error));
  });
}

var predictDrinkingButton = document.getElementById("predict-drinking-button");
if(predictDrinkingButton) {
  predictDrinkingButton.addEventListener("click", function () {
    submitForm()
      .then(inputData => sendDrinkingData(inputData))
      .catch(error => console.error('Error', error));
  });
}
