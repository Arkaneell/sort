let isVerified = false;

async function verifyImageBeforeSubmission() {
    console.log("Verifying image...");

    const fileInput = document.getElementById("camera-input");
    const selectedCategory = document.getElementById("category").value;

    if (!fileInput.files.length) {
        alert("Please upload an image before submitting.");
        return false;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);
    formData.append("category", selectedCategory); // ✅ Added category field

    try {
        const response = await fetch("https://sort-teal.vercel.app/", { // ✅ Fixed port
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        console.log("Server Response:", result);

        if (result.error) {
            alert(`Error: ${result.error}`);
            return false;
        }

        if (result.class.toLowerCase() === selectedCategory.toLowerCase()) {
            isVerified = true;
            alert(`✅ Verified: ${result.class} (Confidence: ${(result.confidence * 100).toFixed(2)}%)`);
            return true;
        } else {
            alert(`❌ Mismatch! Expected: ${selectedCategory}, Detected: ${result.class}`);
            return false;
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error verifying image. Please try again.");
        return false;
    }
}

async function handleSubmit(event) {
    event.preventDefault();
    const verified = await verifyImageBeforeSubmission();
    if (verified) {
        document.getElementById("sell-form").submit();
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("sell-form");
    if (form) form.addEventListener("submit", handleSubmit);
});
