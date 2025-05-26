const form = document.querySelector('.chat-form');
const chatBox = document.querySelector('.chat-box');

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    try {
        const response = await fetch(form.action, {
            method: 'POST',
            headers: {
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRFToken': getCookie('csrftoken'),
            },
            body: formData,
        });

        if (!response.ok) throw new Error('Error en la petición');

        const html = await response.text();

        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        const nuevoChatBox = tempDiv.querySelector('.chat-box');
        if (nuevoChatBox) {
            chatBox.innerHTML = nuevoChatBox.innerHTML;
            // Esperamos que renderice para mover scroll al final:
            requestAnimationFrame(() => {
                chatBox.scrollTop = chatBox.scrollHeight;
            });
        }

        form.reset();

    } catch (error) {
        alert('Hubo un error enviando la imagen. Intenta de nuevo.');
        console.error(error);
    }
});


// Función para obtener cookie csrf (igual que antes)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            cookie = cookie.trim();
            if (cookie.startsWith(name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
