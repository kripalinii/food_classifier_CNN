// Food Image Classifier - Interactive JavaScript

// Global variables
let selectedFile = null

// DOM elements
const fileInput = document.getElementById("fileInput")
const uploadBox = document.getElementById("uploadBox")
const previewSection = document.getElementById("previewSection")
const previewImage = document.getElementById("previewImage")
const predictBtn = document.getElementById("predictBtn")
const loadingSection = document.getElementById("loadingSection")
const resultsSection = document.getElementById("resultsSection")

// Initialize event listeners
document.addEventListener("DOMContentLoaded", () => {
  setupEventListeners()
})

function setupEventListeners() {
  // File input change
  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect)
  }

  // Drag and drop functionality
  if (uploadBox) {
    uploadBox.addEventListener("click", () => fileInput.click())
    uploadBox.addEventListener("dragover", handleDragOver)
    uploadBox.addEventListener("dragleave", handleDragLeave)
    uploadBox.addEventListener("drop", handleDrop)
  }

  // Predict button
  if (predictBtn) {
    predictBtn.addEventListener("click", predictFood)
  }
}

function handleFileSelect(event) {
  const file = event.target.files[0]
  if (file) {
    selectedFile = file
    showPreview(file)
  }
}

function handleDragOver(event) {
  event.preventDefault()
  uploadBox.classList.add("dragover")
}

function handleDragLeave(event) {
  event.preventDefault()
  uploadBox.classList.remove("dragover")
}

function handleDrop(event) {
  event.preventDefault()
  uploadBox.classList.remove("dragover")

  const files = event.dataTransfer.files
  if (files.length > 0) {
    const file = files[0]
    if (isValidImageFile(file)) {
      selectedFile = file
      fileInput.files = files // Update file input
      showPreview(file)
    } else {
      showError("Please upload a valid image file (PNG, JPG, JPEG, GIF)")
    }
  }
}

function isValidImageFile(file) {
  const allowedTypes = ["image/png", "image/jpeg", "image/jpg", "image/gif"]
  return allowedTypes.includes(file.type)
}

function showPreview(file) {
  const reader = new FileReader()
  reader.onload = (e) => {
    previewImage.src = e.target.result
    previewSection.style.display = "block"

    // Hide other sections
    hideSection(loadingSection)
    hideSection(resultsSection)

    // Smooth scroll to preview
    previewSection.scrollIntoView({ behavior: "smooth" })
  }
  reader.readAsDataURL(file)
}

function predictFood() {
  if (!selectedFile) {
    showError("Please select an image first")
    return
  }

  // Show loading
  showSection(loadingSection)
  hideSection(previewSection)
  hideSection(resultsSection)

  // Prepare form data
  const formData = new FormData()
  formData.append("file", selectedFile)

  // Make prediction request
  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      hideSection(loadingSection)

      if (data.error) {
        showError(data.error)
      } else {
        showResults(data)
      }
    })
    .catch((error) => {
      hideSection(loadingSection)
      showError("Something went wrong. Please try again.")
      console.error("Error:", error)
    })
}

function showResults(data) {
  // Update result title
  const resultTitle = document.getElementById("resultTitle")
  if (resultTitle) {
    resultTitle.textContent = `🎯 ${data.confidence_message}`
  }

  // Update confidence meter
  const confidenceFill = document.getElementById("confidenceFill")
  const confidenceText = document.getElementById("confidenceText")
  if (confidenceFill && confidenceText) {
    const confidencePercent = (data.confidence * 100).toFixed(1)
    confidenceFill.style.width = `${confidencePercent}%`
    confidenceText.textContent = `${confidencePercent}% Confidence`

    // Change color based on confidence
    if (data.confidence >= 0.7) {
      confidenceFill.style.background = "linear-gradient(90deg, #48bb78, #38a169)"
    } else if (data.confidence >= 0.5) {
      confidenceFill.style.background = "linear-gradient(90deg, #ed8936, #dd6b20)"
    } else {
      confidenceFill.style.background = "linear-gradient(90deg, #e53e3e, #c53030)"
    }
  }

  // Update chef tip
  const chefTip = document.getElementById("chefTip")
  if (chefTip) {
    chefTip.innerHTML = `<strong>👨‍🍳 Chef's Tip:</strong> ${data.chef_tip}`
  }

  // Update recipe link
  const recipeLink = document.getElementById("recipeLink")
  if (recipeLink) {
    recipeLink.innerHTML = `
            <strong>🔗 Try this recipe:</strong><br>
            <a href="${data.recipe_url}" target="_blank" rel="noopener">
                Get cooking instructions →
            </a>
        `
  }

  // Store smell message for easter egg
  const smellMessage = document.getElementById("smellMessage")
  if (smellMessage) {
    smellMessage.textContent = data.smell_message
  }

  // Show results section
  showSection(resultsSection)
  resultsSection.scrollIntoView({ behavior: "smooth" })
}

function showSmellMessage() {
  const smellMessage = document.getElementById("smellMessage")
  if (smellMessage) {
    smellMessage.style.display = "block"

    // Add a fun animation
    smellMessage.style.opacity = "0"
    smellMessage.style.transform = "translateY(10px)"

    setTimeout(() => {
      smellMessage.style.transition = "all 0.5s ease"
      smellMessage.style.opacity = "1"
      smellMessage.style.transform = "translateY(0)"
    }, 100)
  }
}

function resetApp() {
  // Reset all sections
  hideSection(previewSection)
  hideSection(loadingSection)
  hideSection(resultsSection)

  // Clear file input
  if (fileInput) {
    fileInput.value = ""
  }
  selectedFile = null

  // Scroll back to top
  window.scrollTo({ top: 0, behavior: "smooth" })
}

function showSection(section) {
  if (section) {
    section.style.display = "block"
  }
}

function hideSection(section) {
  if (section) {
    section.style.display = "none"
  }
}

function showError(message) {
  alert(`❌ Error: ${message}`)
}

// Fun loading messages
const loadingMessages = [
  "🧠 AI is thinking... analyzing your delicious photo!",
  "🔍 Examining pixels for food patterns...",
  "🍽️ Consulting the digital chef's cookbook...",
  "📊 Calculating deliciousness levels...",
  "🎯 Identifying your tasty creation...",
]

function updateLoadingMessage() {
  const loadingText = document.querySelector("#loadingSection p")
  if (loadingText) {
    const randomMessage = loadingMessages[Math.floor(Math.random() * loadingMessages.length)]
    loadingText.textContent = randomMessage
  }
}

// Update loading message every 2 seconds
setInterval(updateLoadingMessage, 2000)

// Add some fun interactions
document.addEventListener("keydown", (event) => {
  // Easter egg: Press 'F' for food facts
  if (event.key.toLowerCase() === "f" && event.ctrlKey) {
    const foodFacts = [
      "🍕 The world's largest pizza was 13,580.28 square feet!",
      "🍝 There are over 600 pasta shapes produced worldwide!",
      "🥗 The Caesar salad was invented in 1924 in Mexico!",
      "🍩 Americans consume over 10 billion donuts annually!",
    ]
    const randomFact = foodFacts[Math.floor(Math.random() * foodFacts.length)]
    alert(`Fun Food Fact: ${randomFact}`)
    event.preventDefault()
  }
})

// Console easter egg
console.log(`
🍽️ Food Image Classifier Console Commands:
- Press Ctrl+F for random food facts
- Check out the source code on GitHub!
- Built with ❤️ and lots of 🧠
`)
