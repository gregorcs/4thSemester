const leftArrow = document.getElementById("left-arrow");
const rightArrow = document.getElementById("right-arrow");

leftArrow.addEventListener("mousedown", function () {
    leftArrow.classList.add("arrow-clicked");
});

leftArrow.addEventListener("mouseup", function () {
    leftArrow.classList.remove("arrow-clicked");
});

rightArrow.addEventListener("mousedown", function () {
    rightArrow.classList.add("arrow-clicked");
});

rightArrow.addEventListener("mouseup", function () {
    rightArrow.classList.remove("arrow-clicked");
});
