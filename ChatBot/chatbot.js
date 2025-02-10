function sendMessage() {
    var userMessage = document.getElementById("user-input").value;
    var chatBox = document.getElementById("chat-box");

    // Display user message
    var userDiv = document.createElement("div");
    userDiv.classList.add("user-message");
    userDiv.textContent = userMessage;
    chatBox.appendChild(userDiv);

    // Send user input to the backend (Flask API) and get response
    fetch('/get_bot_response', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
    })
    .then(response => response.json())
    .then(data => {
        var botDiv = document.createElement("div");
        botDiv.classList.add("bot-message");
        botDiv.textContent = data.response;
        chatBox.appendChild(botDiv);
    });

    document.getElementById("user-input").value = '';
}
