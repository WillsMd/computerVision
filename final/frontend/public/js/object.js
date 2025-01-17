document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    const imageInput = document.getElementById("imageInput").files[0];
    if (!imageInput) {
        alert("Please upload an image.");
        return;
    }

    const formData = new FormData();
    formData.append("image", imageInput);

    const response = await fetch("/api/object-detection", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const result = await response.json();
        const objectsDiv = document.getElementById("results");
        objectsDiv.innerHTML = "";
        result.objects.forEach(obj => {
            objectsDiv.innerHTML += `<p>Label: ${obj.label} | Confidence: ${obj.confidence} | Coordinates: ${obj.coordinates}</p>`;
        });
    } else {
        alert("Error detecting objects.");
    }
});
