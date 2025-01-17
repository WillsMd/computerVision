document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    const imageInput = document.getElementById("imageInput").files[0];
    if (!imageInput) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput);

    const response = await fetch("/api/face-analysis", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        document.getElementById("uploadedImage").src = imageUrl;
    } else {
        alert("Error analyzing face.");
    }
});
