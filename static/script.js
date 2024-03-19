function previewImage(event) {
    const input = event.target;
    const preview = document.querySelector('#preview');

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.opacity = 1;
        };

        reader.readAsDataURL(input.files[0]);
    }
}
