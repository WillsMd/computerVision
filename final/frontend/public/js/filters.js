document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    const imageInput = document.getElementById("imageInput").files[0];
    if (!imageInput) {
        alert("Please upload an image.");
        return;
    }

    const filterType = document.querySelector('input[name="filter"]:checked').value;

    const formData = new FormData();
    formData.append("image", imageInput);
    formData.append("filter_type", filterType);

    const response = await fetch("/api/apply-filters", {
        method: "POST",
        body: formData,
    });

    if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        document.getElementById("filteredImage").src = imageUrl;
    } else {
        alert("Error applying filter.");
    }
});
