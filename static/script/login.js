const container = document.getElementById('container');
const registerBtn = document.getElementById('registerBtn');
const loginBtn = document.getElementById('loginBtn');

registerBtn.addEventListener('click', () => {
    container.classList.add("active");
});

loginBtn.addEventListener('click', () => {
    container.classList.remove("active");
});

// Add these lines to handle toggle buttons inside hidden panels
document.getElementById('login').addEventListener('click', () => {
    container.classList.remove("active");
});

document.getElementById('register').addEventListener('click', () => {
    container.classList.add("active");
});
