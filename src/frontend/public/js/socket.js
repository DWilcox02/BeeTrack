// Connect to Flask backend
const socket = io("http://127.0.0.1:5001");

// Handle connection
socket.on("connect", () => {
  console.log("Connected to Flask backend");
});

// Handle disconnection
socket.on("disconnect", () => {
  console.log("Disconnected from Flask backend");
});

// Socket API wrapper functions
const api = {
  // Get all videos
  getVideos: () => {
    return new Promise((resolve, reject) => {
      socket.emit("get_videos", {}, (response) => {
        if (response.success) {
          resolve(response.data);
        } else {
          reject(new Error(response.error || "Failed to get videos"));
        }
      });
    });
  },

  // Process video
  processVideo: (videoPath) => {
    return new Promise((resolve, reject) => {
      socket.emit("process_video", { video_path: videoPath }, (response) => {
        if (response.success) {
          resolve(response.job_id);
        } else {
          reject(new Error(response.error || "Failed to start processing"));
        }
      });
    });
  },

  // Process video with points
  processVideoWithPoints: (videoPath, sessionId) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "process_video_with_points",
        {
          video_path: videoPath,
          session_id: sessionId,
        },
        (response) => {
          if (response.success) {
            resolve(response.job_id);
          } else {
            reject(new Error(response.error || "Failed to start processing with points"));
          }
        }
      );
    });
  },

  startNewSession: (session_id, points, video_path, frame_width, frame_height) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "start_new_session",
        {
          session_id: session_id,
          points: points,
          video_path: video_path,
          frame_width: frame_width,
          frame_height: frame_height,
        },
        (response) => {
          if (response.success) {
            resolve(response.session_id);
          } else {
            reject(new Error(response.error || "Failed to start new session"));
          }
        }
      );
    });
  },

  // Update point
  updatePoint: (sessionId, pointIndex, x, y) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "update_point",
        {
          session_id: sessionId,
          point_index: pointIndex,
          x: x,
          y: y,
        },
        (response) => {
          if (response.success) {
            resolve(response.plot_data);
          } else {
            reject(new Error(response.error || "Failed to update point"));
          }
        }
      );
    });
  },

  // Save points
  savePoints: (sessionId) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "save_points",
        {
          session_id: sessionId,
        },
        (response) => {
          if (response.success) {
            resolve(response.points);
          } else {
            reject(new Error(response.error || "Failed to save points"));
          }
        }
      );
    });
  },

  // Check job status
  checkJobStatus: (jobId) => {
    return new Promise((resolve, reject) => {
      socket.emit("job_status", { job_id: jobId }, (response) => {
        if (response.success) {
          resolve(response.result);
        } else {
          reject(new Error(response.error || "Failed to get job status"));
        }
      });
    });
  },

  // Update all points
  updateAllPoints: (sessionId, points) => {
    return new Promise((resolve, reject) => {
      socket.emit(
        "update_all_points",
        {
          session_id: sessionId,
          points: points,
        },
        (response) => {
          if (response.success) {
            resolve(response.points);
          } else {
            reject(new Error(response.error || "Failed to update all points"));
          }
        }
      );
    });
  }
};

// -------------------------------------------------------------------------
// Socket.IO Events
// -------------------------------------------------------------------------

// Listen for job updates
socket.on("job_log", (data) => {
  window.handleJobLog(data.job_id, data.message);
});

// Socket event handlers for point updates
socket.on('update_point_response', (result) => {
  console.log("Socket response:", result);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points);
    }

    showStatus("success");
  }
});


socket.on("update_points_with_frame", (result) => {
  console.log("Updating points after interval");
  console.log("New points: " + result.points);
  console.log("New frame index: " + result.frameData.frame_idx);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points, result.frameData);
    }

    showStatus("success");
  }
})


socket.on("validation_request", (data, callback) => {
  requestValidation(data);
});


socket.on("update_all_points_response", (result) => {
  console.log("Socket response for all points update:", result);

  if (result.success) {
    // Update the plot using the returned plot data
    if (result.points) {
      updatePlot(result.points);
    }

    showStatus("All points updated successfully!", "success");
  } else {
    showStatus(`Error: ${result.error || "Failed to update points"}`, "error");
  }
});


socket.on("add_timeline_frame", (data) => {
  if (data.frame && data.frame_index) {
    addTimelineFrame(data);
  } else {
    console.error("Error with data: " + data)
  }
});


socket.on("add_tracks", (data) => {
  if (data.new_tracks) {
    addTracks(data.new_tracks)
  }
})


socket.on("add_validation", (data) => {
  if (data.validation_point) {
    addValidation(data.validation_point);
  }
})


socket.on("stop_job_success", (data) => {
  console.log("Stop job success:", data);
  showStatus(`Stop signal sent successfully for job ${data.job_id}`, "processing");

  const statusElement = document.getElementById("processingStatus");
  const statusMessageElement = document.getElementById("statusMessage");

  // Update status to success
  statusElement.className = "processing-status status-success";
  statusMessageElement.innerHTML = `Process Stopped`;


  resetProcessingButtons();
});


socket.on("stop_job_error", (data) => {
  console.error("Stop job error:", data);
  showStatus(`Error stopping job: ${data.error}`, "error");

  // Re-enable the stop button if there's an error
  const stopButton = document.getElementById("stopProcessing");
  if (stopButton) {
    stopButton.disabled = false;
    stopButton.textContent = "Stop Processing";
  }
});


socket.on("process_complete", (result) => {
  const statusElement = document.getElementById("processingStatus");
  const statusMessageElement = document.getElementById("statusMessage");

  // Update status to success
  statusElement.className = "processing-status status-success";
  statusMessageElement.innerHTML = `<strong>Success!</strong> Point cloud processing complete. View output video at <a href="/output/${result.output_filename}" target="_blank">${result.output_filename}</a>`;

  resetProcessingButtons();

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
});

// Export the API for use in other scripts
window.api = api;
