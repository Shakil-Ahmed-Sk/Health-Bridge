// static/script.js

let socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('new_comment', function (data) {
    displayComment(data);
});

function submitComment() {
    let username = document.getElementById('username').value;
    let commentText = document.getElementById('comment').value;

    if (username && commentText) {
        let data = { username: username, text: commentText };
        socket.emit('new_comment', data);
        document.getElementById('comment-form').reset();
    }
}

function displayComment(data) {
    let commentsContainer = document.getElementById('comments-container');
    let commentDiv = document.createElement('div');
    commentDiv.innerHTML = '<strong>' + data.username + ':</strong> ' + data.text;
    commentsContainer.appendChild(commentDiv);
}
