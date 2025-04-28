document.addEventListener("DOMContentLoaded", () => {
  // Get the video path from the page
  const videoPath = document.querySelector(".video-path").textContent;

  // Initialize variables to store current job info
  let currentJobId = null;

  // Define the processVideo function globally
  window.processVideo = async function () {
    const statusElement = document.getElementById("processingStatus");
    const statusMessageElement = document.getElementById("statusMessage");
    const processBtn = document.getElementById("processBtn");
    const processedContainer = document.getElementById("processedVideoContainer");
    const processedSource = document.getElementById("processedVideoSource");
    const processedPlayer = document.getElementById("processedVideoPlayer");
    const logsContainer = document.getElementById("processingLogs");
    const logContent = document.getElementById("logContent");

    // Clear previous logs
    logContent.textContent = "";

    // Show processing status and logs
    statusElement.className = "processing-status status-processing";
    statusElement.style.display = "block";
    statusMessageElement.innerHTML =
      '<div class="loading-spinner"></div> Processing video with TAPIR (this may take several minutes)...';
    logsContainer.style.display = "block";
    processBtn.disabled = true;

    try {
      // Start the processing job through socket API
      currentJobId = await api.processVideo(videoPath);

      // Set up log handler for this job
      window.handleJobLog = (jobId, message) => {
        if (jobId !== currentJobId) return;

        if (message.trim() === "") return; // Skip empty messages

        let formattedMessage = message;
        let messageClass = "";

        // Format message based on type
        if (message.startsWith("ERROR:")) {
          formattedMessage = message.substring(6);
          messageClass = "log-error";
        } else if (message.startsWith("DONE:")) {
          formattedMessage = message.substring(5);
          messageClass = "log-success";
        } else if (message.startsWith("Processing ") || message.startsWith("Starting ")) {
          messageClass = "log-info";
        }

        // Add the message to the log
        const logLine = document.createElement("div");
        logLine.textContent = formattedMessage;
        if (messageClass) {
          logLine.className = messageClass;
        }
        logContent.appendChild(logLine);

        // Scroll to bottom
        logContent.scrollTop = logContent.scrollHeight;

        // Check for completion or error
        if (message.startsWith("DONE:") || message.startsWith("ERROR:")) {
          // Check the job status to get the results
          checkJobStatus(currentJobId);
        }
      };
    } catch (error) {
      statusElement.className = "processing-status status-error";
      statusMessageElement.textContent = "Error: " + error.message;
      processBtn.disabled = false;

      // Log the error
      const logLine = document.createElement("div");
      logLine.textContent = "Error: " + error.message;
      logLine.className = "log-error";
      logContent.appendChild(logLine);
    }
  };

  // Function to check job status
  async function checkJobStatus(jobId) {
    const statusElement = document.getElementById("processingStatus");
    const statusMessageElement = document.getElementById("statusMessage");
    const processBtn = document.getElementById("processBtn");
    const processedContainer = document.getElementById("processedVideoContainer");
    const processedSource = document.getElementById("processedVideoSource");
    const processedPlayer = document.getElementById("processedVideoPlayer");

    try {
      const result = await api.checkJobStatus(jobId);

      if (result && result.success) {
        statusElement.className = "processing-status status-success";
        statusMessageElement.innerHTML = `<strong>Success!</strong> Video has been processed.`;

        // Update the processed video player
        processedSource.src = result.output_url;
        processedPlayer.load();
        processedContainer.style.display = "block";

        // If we have tabs interface, create it now
        createTabsIfNeeded(result.output_filename);

        // Switch to the processed video tab
        switchTab("processed");
      } else {
        statusElement.className = "processing-status status-error";
        statusMessageElement.textContent = "Error: " + (result.error || "Unknown error occurred");
      }
    } catch (error) {
      statusElement.className = "processing-status status-error";
      statusMessageElement.textContent = "Error checking job status: " + error.message;
    } finally {
      processBtn.disabled = false;
    }
  }

  // Create tabs if needed
  function createTabsIfNeeded(output_filename) {
    // If tab interface doesn't exist, create it
    const videoTabs = document.querySelector(".video-tabs");
    if (!videoTabs) {
      const container = document.querySelector(".container");
      const h1 = container.querySelector("h1");

      const tabsDiv = document.createElement("div");
      tabsDiv.className = "video-tabs";
      tabsDiv.innerHTML = `
                <div class="tab" onclick="switchTab('original')">Original Video</div>
                <div class="tab active" onclick="switchTab('processed')">Processed Video</div>
            `;

      container.insertBefore(tabsDiv, h1.nextSibling);

      // Create processed video section if it doesn't exist
      if (!document.getElementById("processed-video")) {
        const processedDiv = document.createElement("div");
        processedDiv.id = "processed-video";
        processedDiv.className = "video-info";
        processedDiv.innerHTML = `
                    <h2>Processed: ${output_filename}</h2>
                    <div class="video-path">output/${output_filename}</div>
                    
                    <div class="video-container">
                        <video controls autoplay>
                            <source src="${processedSource.src}" type="video/mp4">
                            Your browser does not support the video tag or this video format.
                        </video>
                    </div>
                `;

        // Insert after original video section
        const originalVideo = document.getElementById("original-video");
        originalVideo.after(processedDiv);
        originalVideo.style.display = "none";
      }
    }
  }

  // Define switchTab function globally
  window.switchTab = function (tabName) {
    // Hide all sections
    document.getElementById("original-video").style.display = "none";
    const processedVideo = document.getElementById("processed-video");
    if (processedVideo) {
      processedVideo.style.display = "none";
    }

    // Show selected section
    document.getElementById(tabName + "-video").style.display = "block";

    // Update tab styling
    const tabs = document.querySelectorAll(".tab");
    tabs.forEach((tab) => {
      tab.classList.remove("active");
      if (tab.textContent.trim().toLowerCase().includes(tabName)) {
        tab.classList.add("active");
      }
    });
  };
});
