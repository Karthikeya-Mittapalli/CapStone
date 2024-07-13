document.getElementById('generate-btn').addEventListener('click', function() {
    fetch('/generate-image', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            document.getElementById('generatedImage').src = data.image_url + '?' + new Date().getTime();
        })
        .catch(error => {
            console.error('Error generating image:', error);
        });
});
