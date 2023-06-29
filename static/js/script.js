
// Agregar mensagens do chat
function getCompletion() {
    let userText = $("#userMessage").val();
    let userHtml = '<div class="row justify-content-start my-4"><div class="col-10 d-flex justify-content-start"><img src="static\\img\\002-man.png" class="me-3" alt="" width="50" height="58"><div class="alert alert-secondary" role="alert">' + userText + '</div></div></div>';
    $("#userMessage").val("");
    $("#messagebox").append(userHtml);

    // Aplicar "scroll to bottom" depois do input de mensagem
    var messageBox = document.getElementById("messagebox");
    messageBox.scrollTop = messageBox.scrollHeight;

    $.get("/get", { msg: userText }).done(function (data) {
        var assistantHTML = '<div class="row justify-content-end my-4"><div class="col-10 d-flex justify-content-end"><div class="alert alert-primary" role="alert">' + data + '</div><img src="static\\img\\001-assistant.png" class="ms-3" alt="" width="50" height="58"></div></div>';
        $("#messagebox").append(assistantHTML);

        // Aplicar "scroll to bottom" da caixa de mensagens após um tempo de espera
        setTimeout(function () {
            var messageBox2 = document.getElementById("messagebox");
            messageBox2.scrollTop = messageBox2.scrollHeight;
        }, 100);
    });
}

// Pressione Enter para aplicar o input introduzido
$("#userMessage").keypress(function (e) {
    if (e.which == 13) {
        getCompletion();
    }
});

// Clique no botão Enviar para aplicar o input introduzido
$("#sendButton").click(function () {
    getCompletion();
});