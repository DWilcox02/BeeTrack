// Store the session ID and data
let selectedPointIndex = null;
let selectedPointColor = null;
let plotlyPlot = null;
let pointPositions = {}; // Keep track of current point positions
let pointRadiusValues = [50, 50, 50, 50];

const backend_url = "http://127.0.0.1:5001";

const appContainer = document.getElementById("app-container")
const sessionId = appContainer.dataset.sessionId;
const marginTop = appContainer.dataset.marginTop;
const marginLeft = appContainer.dataset.marginLeft;
const marginRight = appContainer.dataset.marginRight;
const marginBottom = appContainer.dataset.marginBottom;
const imageWidth = window.appConfig.imageWidth;
const imageHeight = window.appConfig.imageHeight;

let currentJobId = null;

// List of images as data URLs

const images = [];

// Get DOM elements
const timelineSlider = document.getElementById('timelineSlider');
timelineSlider.max = images.length - 1;
const imageNumber = document.getElementById('imageNumber');
const totalImages = document.getElementById("totalImages");
totalImages.textContent = images.length;
const displayedImage = document.getElementById('displayedImage');

const tracks = [];

// Initialize with the first image
try {
    displayedImage.src = images[0];
} catch (error) {
    console.log("No slider images")
}

// Update image when slider value changes
timelineSlider.addEventListener('input', function() {
    const index = parseInt(this.value);
    displayedImage.src = images[index];
    imageNumber.textContent = index;
});

function addTimelineFrame(data) {
  let imageData = data.frame;
  imageData = ensureDataUrlFormat(imageData);
  const frameIndex = data.frame_index;
  images.push(imageData);
  timelineSlider.max = images.length - 1;
  totalImages.textContent = images.length - 1;
}

// Socket event handler for job logs
window.handleJobLog = function (jobId, message) {
  // Only process logs for current job
  if (currentJobId !== null && jobId !== currentJobId) return;

  const logContent = document.getElementById("logContent");
  if (!logContent) return;

  if (message.trim() === "") return; // Skip empty messages

  // Format message based on type
  let formattedMessage = message;
  let messageClass = "";

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

  // Use requestAnimationFrame for reliable scrolling after DOM update
  requestAnimationFrame(() => {
    logContent.scrollTop = logContent.scrollHeight;
  });

  // Check for completion or error
  if (message.startsWith("DONE:")) {
    const statusElement = document.getElementById("processingStatus");
    const statusMessageElement = document.getElementById("statusMessage");

    // Update status to success
    statusElement.className = "processing-status status-success";
    statusMessageElement.innerHTML = "<strong>Success!</strong> Point cloud processing complete.";

    // Re-enable the process button
    const processButton = document.getElementById("processPointCloud");
    if (processButton) {
      processButton.disabled = false;
    }

    // Show validation button if appropriate
    const validationButton = document.getElementById("validationContinue");
    if (validationButton) {
      validationButton.style.display = "block";
    }
  } else if (message.startsWith("ERROR:")) {
    const statusElement = document.getElementById("processingStatus");
    const statusMessageElement = document.getElementById("statusMessage");

    // Update status to error
    statusElement.className = "processing-status status-error";
    statusMessageElement.textContent = "Error: " + formattedMessage;

    // Re-enable the process button
    const processButton = document.getElementById("processPointCloud");
    if (processButton) {
      processButton.disabled = false;
    }
  }
};


// Initialize once page is loaded
document.addEventListener("DOMContentLoaded", function () {
  // Set up the point selection buttons
  document.querySelectorAll(".point-button").forEach((button) => {
    button.addEventListener("click", function () {
      selectPoint(this.dataset.index, this.dataset.color);
    });
  });

  const sendRadiusBtn = document.getElementById("sendRadiusBtn");
  if (sendRadiusBtn) {
    sendRadiusBtn.addEventListener("click", updateRadiusForSelectedPoint);
  }
  const importPointsBtn = document.getElementById("importPointsBtn");
  if (importPointsBtn) {
    importPointsBtn.addEventListener("click", importPointsFromText);
  }

  // Initialize overlay and handlers
  initializePlotHandlers();

  // Show initial status
  showStatus("Select a point and click on the image to place it.", "processing");

  setTimeout(initializeRadiusValues, 500);
});

function importPointsFromText() {
  const inputText = document.getElementById("pointsDataInput").value.trim();

  if (!inputText) {
    showStatus("Please paste valid points data in the text area.", "error");
    return;
  }

  try {
    // Try to parse the input as JSON
    let pointsData;

    // Check if the input starts with a [ and is array-like
    if (inputText.startsWith("[")) {
      // Try to evaluate the JavaScript object notation (safer than eval)
      try {
        pointsData = JSON.parse(inputText.replace(/([{,]\s*)(\w+):/g, '$1"$2":'));
      } catch (e) {
        // If JSON.parse fails, try Function constructor as a safer alternative to eval
        // This allows for JavaScript object notation with unquoted property names
        pointsData = new Function("return " + inputText)();
      }
    } else {
      throw new Error("Input does not appear to be a valid array of points");
    }

    // Validate the parsed data
    if (!Array.isArray(pointsData)) {
      throw new Error("Input must be an array of point objects");
    }

    // Validate each point has the required properties
    for (let i = 0; i < pointsData.length; i++) {
      const point = pointsData[i];
      if (
        typeof point.x !== "number" ||
        typeof point.y !== "number" ||
        typeof point.color !== "string" ||
        typeof point.radius !== "number"
      ) {
        throw new Error(`Invalid point at index ${i}. Each point must have x, y, color, and radius properties.`);
      }
    }

    // Send all points to the backend
    updateAllPoints(pointsData);
  } catch (error) {
    console.error("Error parsing points data:", error);
    showStatus(`Error: ${error.message}`, "error");
  }
}

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

    // console.log("Plot overlay initialized with click handler");
  } else {
    console.error("Plot overlay element not found");
  }
}

function initializeRadiusValues() {
  if (window.pointsData && Array.isArray(window.pointsData)) {
    pointsData.forEach((point, index) => {
      if (point && point.radius !== undefined) {
        pointRadiusValues[index] = parseInt(point.radius, 10);
      }
    });
  }
  console.log("Initialized radius values:", pointRadiusValues);
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

    // console.log(`Overlay resized to match plot: ${rect.width}x${rect.height}`);
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

  // Get current radius for this point from our array or from the pointsData
  let currentRadius = 50; // Default

  // Try to get it from pointsData first if available
  if (
    window.pointsData &&
    window.pointsData[selectedPointIndex] &&
    window.pointsData[selectedPointIndex].radius !== undefined
  ) {
    currentRadius = parseInt(window.pointsData[selectedPointIndex].radius, 10);
    // Update our tracking array
    pointRadiusValues[selectedPointIndex] = currentRadius;
  } else if (pointRadiusValues[selectedPointIndex] !== undefined) {
    // Fall back to our tracking array
    currentRadius = pointRadiusValues[selectedPointIndex];
  }

  // Update the slider value
  const radiusSlider = document.getElementById("radiusRange");
  radiusSlider.value = currentRadius;

  // Update the displayed value
  document.getElementById("radiusValue").textContent = currentRadius;

  // Store as selected radius
  selectedPointRadius = currentRadius;

  // Highlight the selected button
  document.getElementById("point" + color.charAt(0).toUpperCase() + color.slice(1)).classList.add("active");

  // Show status
  showStatus(
    `Selected ${color} point. Click on the image to place it or use the slider to adjust radius.`,
    "processing"
  );
}

function updateRadiusForSelectedPoint() {
  if (selectedPointIndex === null) {
    showStatus("Please select a point first.", "processing");
    return;
  }

  // Get the current slider value
  const newRadius = parseInt(document.getElementById("radiusRange").value, 10);

  // Store the new radius in our array
  pointRadiusValues[selectedPointIndex] = newRadius;

  // Get the last known position of this point
  const point = window.pointsData[selectedPointIndex];

  // Make sure we have valid coordinates
  if (point && point.x !== undefined && point.y !== undefined) {
    // Update the point with the new radius
    updatePoint(selectedPointIndex, point.x, point.y, newRadius);
  } else {
    showStatus("Point position unknown. Place the point first.", "error");
  }
}


// Function to handle plot clicks with precise coordinate mapping
function handlePlotClick(event) {
  if (selectedPointIndex === null) {
    showStatus("Please select a point color first.", "processing");
    return;
  }

  radius = selectedPointRadius;

  // Get the Plotly plot element
  const plotlyDiv = document.querySelector(".js-plotly-plot");
  if (!plotlyDiv) {
    console.error("Plotly div not found");
    return;
  }

  // Get the bounding rect of the plot
  const plotRect = plotlyDiv.getBoundingClientRect();
  const plotWidth = plotRect.width;
  const plotHeight = plotRect.height;

  // Calculate mouse position relative to the plot div
  const mouseX = event.clientX - plotRect.left;
  const mouseY = event.clientY - plotRect.top;

  // Use Plotly's internal conversion function to convert from pixel coordinates to data coordinates
  // This eliminates any issues with calculating margins manually
  try {

      // Convert from pixel to data coordinates
      // These are Plotly's internal conversion methods

      // TODO: Actually fix instead of manual forced change
      const dataX = (mouseX - marginLeft)/(plotWidth - marginLeft - marginRight) * imageWidth;
      const dataY = (mouseY - marginTop)/(plotHeight - marginTop - marginBottom) * imageHeight;

      console.log(`Mouse position on plot: (${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`);
      console.log(`Converted to data coordinates: (${dataX.toFixed(1)}, ${dataY.toFixed(1)})`);

      // Update point position on the server
      updatePoint(selectedPointIndex, dataX, dataY, radius);
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
      updatePoint(selectedPointIndex, dataX, dataY, radius);
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
async function updatePoint(pointIndex, x, y, radius) {
  showStatus(`Updating ${selectedPointColor} point position...`, "processing");

  try {
    showStatus(`Updating ${selectedPointColor} point position...`, "processing");

    try {
      console.log(`Sending socket request for point ${pointIndex} at coordinates (${x}, ${y})`);

      // Emit the update_point event with the data
      socket.emit("update_point", {
        session_id: sessionId,
        point_index: pointIndex,
        x: x,
        y: y,
        radius: radius
      });

      // The response will be handled by the socket.on('update_point_response') handler above
    } catch (error) {
      console.error("Error emitting socket event:", error);
      showStatus(`Error: ${error.message}`, "error");
    }
  } catch (error) {
    console.error("Error updating point:", error);
    showStatus(`Error: ${error.message}`, "error");
  }
}

// Function to update all points at once
async function updateAllPoints(points) {
  showStatus("Updating all point positions...", "processing");

  try {
    // Emit the update_all_points event with the data
    socket.emit("update_all_points", {
      session_id: sessionId,
      points: points,
    });

    // Update the frontend visualization immediately
    updatePlot(points);

    showStatus("Points updated successfully!", "success");
  } catch (error) {
    console.error("Error updating points:", error);
    showStatus(`Error: ${error.message}`, "error");
  }
}


function ensureDataUrlFormat(imageData) {
  // If the image data doesn't start with the data URL prefix, add it
  if (!imageData.startsWith("data:image")) {
    return `data:image/jpeg;base64,${imageData}`;
  }
  return imageData;
}

function pointToCircle(point) {
  return {
    type: "circle",
    xref: "x",
    yref: "y",
    x0: parseFloat(point["x"]) - parseFloat(point["radius"]),
    y0: parseFloat(point["y"]) - parseFloat(point["radius"]),
    x1: parseFloat(point["x"]) + parseFloat(point["radius"]),
    y1: parseFloat(point["y"]) + parseFloat(point["radius"]),
    line: {
      color: point["color"],
    },
  };
}


// Function to update the plot with new data
function updatePlot(newPoints, frameData=null) {
  try {
    // Find the Plotly element if not already found
    if (!plotlyPlot) {
      plotlyPlot = document.querySelector(".js-plotly-plot");
      if (!plotlyPlot) {
        console.error("Plotly element not found for updating");
        return;
      }
    }

    // Create new shapes array
    const newCircles = newPoints.map(pointToCircle)

    // Update the global pointsData
    window.pointsData = newPoints;

    // Use Plotly's react method to update the plot with new data
    if (window.Plotly) {
      // Get the current layout
      let currentLayout = plotlyPlot.layout || plotData.layout;

      if (!Array.isArray(newPoints) || currentLayout.shapes.length != newPoints.length) {
        console.error("newPoints must be an array of the same length as current number of points");
        return;
      }
      
      // Update the image in the layout if frameData is provided
      if (frameData && frameData.frame) {
        imageData = ensureDataUrlFormat(frameData.frame);
        
        // If we have a new image, update it in the layout
        if (!currentLayout.images) {
          currentLayout.images = [];
        }
        
        // If an image already exists, update it, otherwise add a new one
        if (currentLayout.images.length > 0) {
          console.log("Updating existing image in layout");
          currentLayout.images[0] = {
            source: imageData,
            x: 0,
            y: 0,
            sizex: frameData.width,
            sizey: frameData.height,
            sizing: 'stretch',
            layer: 'below',
            xref: 'x',
            yref: 'y',
          };
        } else {
          console.log("Adding new image to layout");
          currentLayout.images.push({
            source: frameData.frame,
            x: 0,
            y: 0,
            sizex: frameData.width,
            sizey: frameData.height,
            sizing: 'stretch',
            layer: 'below',
            xref: 'x',
            yref: 'y',
          });
        }
        
        // Update title to show current frame
        currentLayout.title = `Frame Analysis`;
      }

      console.log(currentLayout.shapes)
      for (let i = 0; i < currentLayout.shapes.length; i++) {
        currentLayout.shapes[i] = newCircles[i];
      }

      Plotly.react(plotlyPlot, [], currentLayout, {
        displayModeBar: true,
        staticPlot: false,
        scrollZoom: false,
        displaylogo: false,
        modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "resetScale2d"],
      });

      console.log("Plot updated with new points:", newPoints);
      if (frameData) {
        console.log("Plot updated with new frame image");
      }
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
  const statusElement = document.getElementById("processingStatus");
  const statusMessageElement = document.getElementById("statusMessage");
  const logsContainer = document.getElementById("processingLogs");
  const logContent = document.getElementById("logContent");

  const processPointCloudButton = document.getElementById("processPointCloud");
  if (processPointCloudButton) {
    processPointCloudButton.disabled = true;
  }

  // Clear previous logs
  logContent.textContent = "";

  // Show processing status and logs
  statusElement.className = "processing-status status-processing";
  statusElement.style.display = "block";
  statusMessageElement.innerHTML =
    '<div class="loading-spinner"></div> Processing video with TAPIR (this may take several minutes)...';
  logsContainer.style.display = "block";

  try {
    // Generate a unique job ID if server doesn't provide one
    currentJobId = "pointcloud_" + new Date().getTime();

    socket.emit("process_video_with_points", {
      session_id: sessionId,
      job_id: currentJobId, // Pass the job ID to the server
    });
  } catch (error) {
    statusElement.className = "processing-status status-error";
    statusMessageElement.textContent = "Error: " + error.message;

    // Log the error
    const logLine = document.createElement("div");
    logLine.textContent = "Error: " + error.message;
    logLine.className = "log-error";
    logContent.appendChild(logLine);

    // Re-enable the process button
    if (processPointCloudButton) {
      processPointCloudButton.disabled = false;
    }
  }
}

async function sendValidationContinue() {
  console.log("User validated the request");

  // Hide the validation button
  document.getElementById("validationContinue").style.display = "none";

  // Get the request ID
  const requestId = window.pendingValidationRequestId;

  if (requestId) {
    // Send the response back as a new event
    socket.emit("validation_response", {
      request_id: requestId,
      validated: true,
      timestamp: new Date().toISOString(),
    });

    // Clear the stored request ID
    window.pendingValidationRequestId = null;
  } else {
    console.warn("No pending validation request ID found");
  }
}


async function downloadTracksCSV() {
  console.log("Downloading CSV Tracks")
}

function trackJSONtoTableRow(track) {
  /*
  track: {
    "frame": int,
    "x": float,
    "y": float,
    "bodypart": string
  }
  */

  return `<tr>
    <td>${track.frame}</td>
    <td>${track.x}</td>
    <td>${track.y}</td>
    <td>${track.bodypart}</td>
  </tr>`;
}

function addTracks(new_tracks) {
  /* 
  new_tracks: [
    {
      "frame": int,
      "x": float,
      "y": float,
      "bodypart": string
    }
  ]
  */
  Array.prototype.push.apply(tracks, new_tracks);
  const tracksTable = document.getElementById("tracksTable");
  const tbody = tracksTable.querySelector("tbody");

  new_tracks.forEach((track) => {
    tbody.innerHTML += trackJSONtoTableRow(track);
  });
}