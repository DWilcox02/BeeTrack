// Store the session ID and data
const sessionId = "{{ session_id }}";
let selectedPointIndex = null;
let selectedPointColor = null;
let plotlyPlot = null;
let pointPositions = {}; // Keep track of current point positions

// Initialize once page is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Set up the point selection buttons
  document.querySelectorAll(".point-button").forEach((button) => {
    button.addEventListener("click", function () {
      selectPoint(this.dataset.index, this.dataset.color);
    });
  });

  // Initialize overlay and handlers
  initializePlotHandlers();

  // Show initial status
  showStatus("Select a point and click on the image to place it.", "processing");
});

// Initialize plot handlers
function initializePlotHandlers() {
  // Find the plotly container
  const plotlyContainer = document.getElementById("plotly-container");
  if (!plotlyContainer) {
    console.error("Plotly container not found");
    return;
  }

  // Find any Plotly plot inside
  plotlyPlot = document.querySelector(".js-plotly-plot");
  if (!plotlyPlot) {
    // If not found, it might be loading - retry after delay
    console.log("Waiting for Plotly to initialize...");
    setTimeout(initializePlotHandlers, 300);
    return;
  }

  // Add click handler to the overlay
  const overlay = document.getElementById("plot-overlay");
  if (overlay) {
    // Set the overlay to match the plot dimensions
    resizeOverlay();

    // Add event listeners
    window.addEventListener("resize", resizeOverlay);
    overlay.addEventListener("click", handlePlotClick);

    console.log("Plot overlay initialized with click handler");
  } else {
    console.error("Plot overlay element not found");
  }
}

// Resize the overlay to match the plot
function resizeOverlay() {
  const plotContainer = document.querySelector(".plotly-container");
  if (!plotContainer) return;

  const overlay = document.getElementById("plot-overlay");
  if (!overlay) return;

  // Get the dimensions of the plot area (any SVG inside)
  const plotSvg = plotContainer.querySelector("svg");
  if (plotSvg) {
    const rect = plotSvg.getBoundingClientRect();

    // Position the overlay to match the SVG
    overlay.style.width = rect.width + "px";
    overlay.style.height = rect.height + "px";

    // For positioning, use CSS
    overlay.style.position = "absolute";
    overlay.style.top = "0";
    overlay.style.left = "0";

    console.log(`Overlay resized to match plot: ${rect.width}x${rect.height}`);
  } else {
    // As fallback, match container size
    overlay.style.width = "100%";
    overlay.style.height = "100%";
    overlay.style.position = "absolute";
    overlay.style.top = "0";
    overlay.style.left = "0";

    console.log("Overlay sized to 100% of container (fallback)");
  }
}

// Function to select a point
function selectPoint(index, color) {
  // Reset all buttons
  document.querySelectorAll(".point-button").forEach((btn) => {
    btn.classList.remove("active");
  });

  // Set the selected point
  selectedPointIndex = parseInt(index);
  selectedPointColor = color;

  // Highlight the selected button
  document.getElementById("point" + color.charAt(0).toUpperCase() + color.slice(1)).classList.add("active");

  // Show status
  showStatus(`Selected ${color} point. Click on the image to place it.`, "processing");
}

// Function to handle plot clicks with precise coordinate mapping
function handlePlotClick(event) {
  if (selectedPointIndex === null) {
    showStatus("Please select a point color first.", "processing");
    return;
  }

  // Get the Plotly plot element
  const plotlyDiv = document.querySelector(".js-plotly-plot");
  if (!plotlyDiv) {
    console.error("Plotly div not found");
    return;
  }

  // Get the bounding rect of the plot
  const plotRect = plotlyDiv.getBoundingClientRect();

  // Calculate mouse position relative to the plot div
  const mouseX = event.clientX - plotRect.left;
  const mouseY = event.clientY - plotRect.top;

  // Use Plotly's internal conversion function to convert from pixel coordinates to data coordinates
  // This eliminates any issues with calculating margins manually
  try {
    const coordData = plotlyDiv._fullLayout.clickmodeBar
      ? plotlyDiv._fullLayout.clickmodeBar.clickData({
          xpx: mouseX,
          ypx: mouseY,
        })
      : null;

    if (!coordData) {
      // Fallback method if clickmodeBar is not available
      // Get the axes
      const xaxis = plotlyDiv._fullLayout.xaxis;
      const yaxis = plotlyDiv._fullLayout.yaxis;

      // Convert from pixel to data coordinates
      // These are Plotly's internal conversion methods

      // TODO: Actually fix instead of manual forced change
      const dataX = xaxis.p2d(mouseX) - 128.9;
      const dataY = yaxis.p2d(mouseY) - 90.9;

      console.log(`Mouse position on plot: (${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`);
      console.log(`Converted to data coordinates: (${dataX.toFixed(1)}, ${dataY.toFixed(1)})`);

      // Update point position on the server
      updatePoint(selectedPointIndex, dataX, dataY);
    } else {
      console.log(`Using clickmodeBar data:`, coordData);
      // Update point position using the coordinate data from clickmodeBar
      updatePoint(selectedPointIndex, coordData.x, coordData.y);
    }
  } catch (error) {
    console.error("Error converting coordinates:", error);

    // Last resort fallback - manually calculate using plot layout information
    try {
      const layout = plotlyDiv._fullLayout;
      const xaxis = layout.xaxis;
      const yaxis = layout.yaxis;

      // Get plot area dimensions and positions
      const xa = {
        _offset: xaxis._offset,
        _length: xaxis._length,
        range: xaxis.range,
      };

      const ya = {
        _offset: yaxis._offset,
        _length: yaxis._length,
        range: yaxis.range,
      };

      // Calculate relative position within plot area
      const relX = mouseX - xa._offset;
      const relY = mouseY - ya._offset;

      // Convert to data coordinates
      const xFraction = relX / xa._length;
      const yFraction = relY / ya._length;

      const xRange = xa.range[1] - xa.range[0];
      const yRange = ya.range[1] - ya.range[0];

      let dataX = xa.range[0] + xFraction * xRange;
      let dataY = ya.range[0] + (1 - yFraction) * (ya.range[0] - ya.range[1]);

      console.log(
        `Fallback calculation: (${mouseX}, ${mouseY}) -> (${relX}, ${relY}) -> (${dataX.toFixed(1)}, ${dataY.toFixed(
          1
        )})`
      );
      console.log(`Plot area: offset (${xa._offset}, ${ya._offset}), length (${xa._length}, ${ya._length})`);
      console.log(`X range: [${xa.range[0]}, ${xa.range[1]}], Y range: [${ya.range[0]}, ${ya.range[1]}]`);

      // Update point position on the server
      updatePoint(selectedPointIndex, dataX, dataY);
    } catch (fallbackError) {
      console.error("Fallback coordinate conversion also failed:", fallbackError);
      showStatus("Error calculating coordinates. Please try again.", "error");
    }
  }
}

// Function to show status messages
function showStatus(message, type) {
  const pointStatus = document.getElementById("pointStatus");
  if (!pointStatus) return;

  pointStatus.textContent = message;
  pointStatus.style.display = "block";

  // Set the appropriate status type
  pointStatus.className = "processing-status";
  if (type === "success") {
    pointStatus.classList.add("status-success");
  } else if (type === "error") {
    pointStatus.classList.add("status-error");
  } else {
    pointStatus.classList.add("status-processing");
  }
}

// Function to update a point's position
async function updatePoint(pointIndex, x, y) {
  showStatus(`Updating ${selectedPointColor} point position...`, "processing");

  try {
    console.log(`Sending API request for point ${pointIndex} at coordinates (${x}, ${y})`);

    const response = await fetch("/api/update_point", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        point_index: pointIndex,
        x: x,
        y: y,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const result = await response.json();
    console.log("Server response:", result);

    if (result.success) {
      // Store the updated position
      pointPositions[selectedPointColor] = { x: x, y: y };

      // Update the plot using the returned plot data
      if (result.plot_data) {
        updatePlot(result.plot_data);
      }

      showStatus(
        `${selectedPointColor.charAt(0).toUpperCase() + selectedPointColor.slice(1)} point placed at (${x.toFixed(
          1
        )}, ${y.toFixed(1)})`,
        "success"
      );
    } else {
      throw new Error(result.error || "Failed to update point");
    }
  } catch (error) {
    console.error("Error updating point:", error);
    showStatus(`Error: ${error.message}`, "error");
  }
}

// Function to update the plot with new data
function updatePlot(plotData) {
  try {
    // If plotData is a string (JSON), parse it
    if (typeof plotData === "string") {
      plotData = JSON.parse(plotData);
    }

    // Find the Plotly element if not already found
    if (!plotlyPlot) {
      plotlyPlot = document.querySelector(".js-plotly-plot");
      if (!plotlyPlot) {
        console.error("Plotly element not found for updating");
        return;
      }
    }

    // Use Plotly's react method to update the plot
    if (window.Plotly) {
      Plotly.react(plotlyPlot, plotData.data, plotData.layout, {
        displayModeBar: true,
        staticPlot: false,
        scrollZoom: false,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
      });

      console.log("Plot updated with new data");
    } else {
      console.error("Plotly not available for updating plot");
    }

    // Resize the overlay to match the updated plot
    setTimeout(resizeOverlay, 100);
  } catch (error) {
    console.error("Error updating plot:", error);
    showStatus("Error updating the plot visualization", "error");
  }
}

// Function to save all points
async function savePoints() {
  showStatus("Saving point positions...", "processing");

  try {
    const response = await fetch("/api/save_points", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status}`);
    }

    const result = await response.json();

    if (result.success) {
      let pointsMessage = "Points saved:\n";
      result.points.forEach((point) => {
        pointsMessage += `${point.color}: (${point.x.toFixed(1)}, ${point.y.toFixed(1)})\n`;
      });

      showStatus("Point positions saved successfully!", "success");
      console.log(pointsMessage);
    } else {
      throw new Error(result.error || "Unknown error");
    }
  } catch (error) {
    console.error("Error saving points:", error);
    showStatus(`Error: ${error.message}`, "error");
  }
}

async function processVideoWithPoints() {
  const videoPath = "{{ filename }}";
  const statusElement = document.getElementById("processingStatus");
  const statusMessageElement = document.getElementById("statusMessage");
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

  console.log("Here (lowercase)");
  try {
    // Start the processing job
    const response = await fetch("/api/process_video_with_points", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        video_path: videoPath,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      throw new Error(`Failed to start processing: ${response.status} ${response.statusText}`);
    }

    const result = await response.json();
    const jobId = result.job_id;

    if (!jobId) {
      throw new Error("No job ID returned from server");
    }

    // Set up event source for log messages
    const eventSource = new EventSource(`/api/logs/${jobId}`);

    // Handle log messages
    eventSource.onmessage = function (event) {
      const message = event.data;

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
        // Close the event source
        eventSource.close();

        // Check the job status to get the results
        checkJobStatus(jobId);
      }
    };

    // Handle errors
    eventSource.onerror = function (error) {
      console.error("EventSource error:", error);
      eventSource.close();

      statusElement.className = "processing-status status-error";
      statusMessageElement.textContent = "Error: Connection to server lost";
    };
  } catch (error) {
    statusElement.className = "processing-status status-error";
    statusMessageElement.textContent = "Error: " + error.message;

    // Log the error
    const logLine = document.createElement("div");
    logLine.textContent = "Error: " + error.message;
    logLine.className = "log-error";
    logContent.appendChild(logLine);
  }
}
